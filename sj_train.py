import time
import torch.backends.cudnn as cudnn

import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from sj_models import Encoder, Decoder, Attention
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from datasets import CaptionDataset
import math
import json
from scipy.misc import imread, imresize, imshow
# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 1 #0
epochs = 100  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 20 #20? #32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-5 # 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 0.1 #5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
  # path to checkpoint, None if none

learning_rate_decay_start = 20
learning_rate_decay_every = 50

optim_alpha = 0.8
optim_beta=0.999
optim_epsilon=1e-8

cnn_optim_alpha = 0.8
cnn_optim_beta=0.999


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    
    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    allaccs = AverageMeter()
    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        encoded_image, global_img_features = encoder(imgs)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(encoded_image, global_img_features, caps, caplens)

        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        scores = scores.data
        targets = targets.data
        
        # Calculate loss
        loss = criterion(scores, targets)

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        all_score = accuracy(scores, targets, imgs.shape[0])
        allaccs.update(all_score, sum(decode_lengths))
        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'
                  'All Accuracy {all_score.val:.3f} ({all_score.avg:.3f})\t'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs,
                                                                          all_score = allaccs))


def validate(val_loader, encoder, decoder, criterion):
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    allaccs = AverageMeter()
    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            encoded_image, global_img_features = encoder(imgs)
                
            scores, caps_sorted, decode_lengths, sort_ind = decoder(encoded_image, global_img_features, caps, caplens)

            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            scores = scores.data
            targets = targets.data

            # Calculate loss
            loss = criterion(scores, targets)

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            
            all_score = accuracy(scores, targets, imgs.shape[0])
            allaccs.update(all_score, sum(decode_lengths))
            
            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'
                      'All Accuracy {all_score.val:.3f} ({all_score.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs,all_score = allaccs))

            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4

data_folder = '/NAS_Storage1/leo8544/a-PyTorch-Tutorial-to-Image-Captioning/caption data/' 
#data_name = 'coco_5_cap_per_img_5_min_word_freq'  
data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'
word_map_file='./caption data/WORDMAP_'+data_name+'.json'
checkpoint = None
#checkpoint = 'BEST_132_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
#checkpoint = 'My_132_no_att_losscheckpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar' 
#checkpoint = 'New_132_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
checkpoint = 'BEST_New_132_checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar'
#checkpoint = 'BEST_132_checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar'

with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

if checkpoint is None:
    decoder = Decoder(attention_dim=attention_dim,
                                        embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr,betas = (optim_alpha,optim_beta),eps=optim_epsilon)
    encoder = Encoder()
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr,betas = (cnn_optim_alpha,cnn_optim_beta)) if fine_tune_encoder else None

else:
   checkpoint = torch.load(checkpoint)
   start_epoch = checkpoint['epoch'] + 1
   epochs_since_improvement = checkpoint['epochs_since_improvement']
   best_bleu4 = checkpoint['bleu-4']
   decoder = checkpoint['decoder']
   decoder_optimizer = checkpoint['decoder_optimizer']
   encoder = checkpoint['encoder']
   encoder_optimizer = checkpoint['encoder_optimizer']
   if fine_tune_encoder is True and encoder_optimizer is None:
       encoder.fine_tune(fine_tune_encoder)
       encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

decoder = decoder.to(device)
encoder = encoder.to(device)

criterion = nn.CrossEntropyLoss().to(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([normalize])
train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

image_path1 ='./test_imgs/test1.jpg'
image_path2 ='./test_imgs/test2.jpg'

img = imread(image_path1)
if len(img.shape) == 2:
    img = img[:, :, np.newaxis]
    img = np.concatenate([img, img, img], axis=2)
    
img = imresize(img, (256, 256))
img = img.transpose(2, 0, 1)
img = img / 255.
img = torch.FloatTensor(img).to(device)
image1 = transform(img)  # (3, 256, 256)
image1 = image1.unsqueeze(0)  # (1, 3, 256, 256)

img = imread(image_path2)
if len(img.shape) == 2:
    img = img[:, :, np.newaxis]
    img = np.concatenate([img, img, img], axis=2)
img = imresize(img, (256, 256))
img = img.transpose(2, 0, 1)
img = img / 255.
img = torch.FloatTensor(img).to(device)
image2 = transform(img)  # (3, 256, 256)
image2 = image2.unsqueeze(0)  # (1, 3, 256, 256)

for epoch in range(start_epoch, epochs):

    if epochs_since_improvement == 20:
        break
    
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) / learning_rate_decay_every
        decay_factor = math.pow(0.5, frac)
        adjust_learning_rate(decoder_optimizer, decay_factor)

    train(train_loader=train_loader,
          encoder=encoder,
          decoder=decoder,
          criterion=criterion,
          encoder_optimizer=encoder_optimizer,
          decoder_optimizer=decoder_optimizer,
          epoch=epoch)

    recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)
        
    
    
    # Check if there was an improvement
    is_best = recent_bleu4 > best_bleu4
    best_bleu4 = max(recent_bleu4, best_bleu4)
    if not is_best:
        epochs_since_improvement += 1
        print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
    else:
        epochs_since_improvement = 0
    
    
        
    encoded_image, global_img_features = encoder(image1)
    seq, alphas, betas = decoder.caption_image_beam_search(encoded_image, global_img_features, 
                                                           word_map, rev_word_map)
    decoder.visualize_with_att(image_path1, seq, alphas, betas, rev_word_map)
    
    encoded_image, global_img_features = encoder(image2)
    seq, alphas, betas = decoder.caption_image_beam_search(encoded_image, global_img_features,
                                                           word_map, rev_word_map)
    decoder.visualize_with_att(image_path2, seq, alphas, betas, rev_word_map)
  

    # Save checkpoint
    save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, recent_bleu4, is_best,'New_132_')





