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


torch.cuda.set_device(5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

data_folder = '/NAS_Storage1/leo8544/a-PyTorch-Tutorial-to-Image-Captioning/caption data/' 
#data_name = 'coco_5_cap_per_img_5_min_word_freq'  
data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'
word_map_file='./caption data/WORDMAP_'+data_name+'.json'

checkpoint = 'BEST_132_checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar'

with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

decoder = decoder.to(device)
encoder = encoder.to(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([normalize])

img_list = []
path_list = []
for i in range(10):
    path = './test_imgs/test'+str(i)+'.jpg'
    path_list.append(path)
    img = imread(path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
        
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    image = transform(img)  # (3, 256, 256)
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    img_list.append(image)

for i in range(len(path_list)):
    encoded_image, global_img_features = encoder(img_list[i])
    seq, alphas, betas = decoder.caption_image_beam_search(encoded_image, global_img_features, 
                                                               word_map, rev_word_map)
    decoder.visualize_with_att(path_list[i], seq, alphas, betas, rev_word_map)
    
# DataLoader
loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
references = list()
hypotheses = list()

beam_size = 3
# For each image
for i, (image, caps, caplens, allcaps) in enumerate(
        tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
    
    k = beam_size

    image = image.to(device)  # (1, 3, 256, 256)
    encoded_image, global_img_features = encoder(image)
    # (1, enc_image_size, enc_image_size, encoder_dim)
    
    enc_image_size = math.sqrt(encoded_image.size(1))
    
    encoder_dim = encoded_image.size(2)

    num_pixels = encoded_image.size(1)

    encoded_image = encoded_image.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    seqs = k_prev_words  # (k, 1)

    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    complete_seqs = list()
    complete_seqs_scores = list()

    step = 1
    
    h = torch.zeros(k, decoder.embed_dim).to(device)
    m = torch.zeros(k, decoder.embed_dim).to(device)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    for t in range(50): 
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        # squeeze for looking with just one word
        V, vg = decoder.project_features(encoded_image,global_img_features)
        vg = vg.expand_as(embeddings)
            
        inputX = torch.cat((embeddings,vg), dim = 1)
            
        h, m, s = decoder.LSTM(inputX, (h,m))
            
        c_hat, alpha, beta = decoder.attention(V, h, s)
            
        #alpha = alpha.view(-1, encoded_image_size, encoded_image_size)
        # for attention on img 
            
        scores = decoder.final_layer(c_hat+h)
        scores = decoder.log_softmax(scores)
        scores = top_k_scores.expand_as(scores) + scores
         
        if t == 0: # at the first time, all k scores are same
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True) 
            # flatten and get top k words
            
        prev_word_inds = top_k_words // decoder.vocab_size  # for sorting by the order of best k
        next_word_inds = top_k_words % decoder.vocab_size  # for sorting by the order of best word index
    
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  
        #seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],dim=1)  
        #seqs_betas = torch.cat([seqs_betas[prev_word_inds], beta[prev_word_inds].unsqueeze(1)], dim=1)
        # update seqs, alphas and betas based on topping best k 
    
        # remove ended seqs
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        
        
        # update ended seqs
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            #complete_seqs_alpha.extend(seqs_alpha[complete_inds])
            #complete_seqs_betas.extend(seqs_betas[complete_inds])
            complete_seqs_scores.extend(top_k_scores[complete_inds])
                
            k -= len(complete_inds)  
                
            # Proceed with incomplete sequences
        if k == 0:
            break
            
        seqs = seqs[incomplete_inds]
        #seqs_alpha = seqs_alpha[incomplete_inds]
        #seqs_betas = seqs_betas[incomplete_inds]
        # update seqs, alphas and betas 
            
        h = h[prev_word_inds[incomplete_inds]]
        m = m[prev_word_inds[incomplete_inds]]
        encoded_image = encoded_image[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            # update based on topping best k 
            
        # Proceed with incomplete sequences
        if k == 0:
            break
       
    try:
        best_seq_index = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[best_seq_index]
    except:
        best_seq_index = 0
        seq = seqs[best_seq_index]
    # get index of best seq
    #alphas = complete_seqs_alpha[best_seq_index].cpu().detach().numpy()
    ##betas = complete_seqs_betas[best_seq_index].cpu().detach().numpy()
    # References
    img_caps = allcaps[0].tolist()
    img_captions = list(
        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
            img_caps))  # remove <start> and pads
    references.append(img_captions)

        # Hypotheses
    hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

    assert len(references) == len(hypotheses)

# Calculate BLEU-4 scores
bleu4 = corpus_bleu(references, hypotheses)
