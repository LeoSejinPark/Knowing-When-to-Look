import torch
from torch import nn
import torchvision

import skimage.transform

from scipy.misc import imread, imresize, imshow
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=8):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        ## AdaptiveAvgPool2d can return set output size regardless of input size
        ## For getting global_img_features
        
        self.fine_tune()

    def forward(self, images):
        encoded_images = self.resnet(images)  
        # resnet output (batch_size, 2048, image_size/32, image_size/32)
        batch_size = encoded_images.shape[0]
        features = encoded_images.shape[1]
    
        global_img_features = self.avg_pool(encoded_images).reshape(batch_size, features) 
        # (14)
        
        encoded_images = encoded_images.permute(0, 2, 3, 1) 
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # -> ecah pixel of encoded image has feature vector(2048 dim)
        
        encoded_images = encoded_images.view(batch_size, -1, features)
        # Flatten image ->  # (batch_size, Num pixcel, features)
                                
        return encoded_images, global_img_features

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
            

class Decoder(nn.Module):

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        
        super(Decoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim) 
        
        self.Cell = nn.LSTMCell(embed_dim * 2,decoder_dim)
        self.attention = Attention(decoder_dim, attention_dim)
        
        self.project_g = nn.Linear(self.encoder_dim, embed_dim)  
        #projection for global feature vector
        self.project_i = nn.Linear(self.encoder_dim, embed_dim)  
        #projection for encoded image
        
        self.project_x = nn.Linear(embed_dim * 2, decoder_dim)
        self.project_h = nn.Linear(decoder_dim, decoder_dim)
        self.final_layer = nn.Linear(decoder_dim, vocab_size) 
        self.init_weights() 
        
        
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.log_softmax = nn.LogSoftmax()
        
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.final_layer.bias.data.fill_(0)
        self.final_layer.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    
    def project_features(self,encoded_image,global_img_features):
        V = self.relu(self.project_i(encoded_image))
        #V is projected spatial image feature (15)
        vg = self.relu(self.project_g(global_img_features))
        #vg is projected global feature (16)
        
        return V, vg
    
    def LSTM(self, x_t, previous_cell):
        h_t_1, m_t_1 = previous_cell
        h_t, m_t = self.Cell(x_t, (h_t_1, m_t_1))
        g_t = self.sigmoid(self.project_x(x_t) + self.project_h(h_t_1))
        #(9)
        s_t =  g_t * self.tanh(m_t)
        #(10)
        return h_t, m_t, s_t
    
    def forward(self, encoded_image, global_img_features, caps, caplens):
      
        batch_size = encoded_image.size(0)
        vocab_size = self.vocab_size
        
        caplens, sort_ind = caplens.squeeze(1).sort(dim=0, descending=True)
        encoded_image = encoded_image[sort_ind]
        caps = caps[sort_ind]
        global_img_features = global_img_features[sort_ind]
        ## sort by caplens, because of training

        V, vg = self.project_features(encoded_image,global_img_features)
        
        embedded_caps = self.embedding(caps)  # (batch_size, max_caption_length, embed_dim)
        # max_caption_length = start + (len(max cap)==50) + end
        vg = vg.unsqueeze(1).expand_as(embedded_caps)
        # for concatenating with embedded_caps
        inputX = torch.cat((embedded_caps,vg), dim = 2)
        # x = [w;v^g]
        
        h_t = torch.zeros(batch_size, self.embed_dim).to(device)
        m_t = torch.zeros(batch_size, self.embed_dim).to(device)
        decode_lengths = (caplens - 1).tolist() # remove start

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths]) # check num of not ended samples
            x_t = inputX[:batch_size_t, t, :] #get not ended x_t
            h_t, m_t, s_t = self.LSTM(x_t, (h_t[:batch_size_t], m_t[:batch_size_t])) #update h,m,s
            c_t_hat,alpha_t, beta_t = self.attention(V[:batch_size_t],h_t,s_t) #get c_t
            preds = self.final_layer(self.dropout(c_t_hat+h_t)) #predict next word
            predictions[:batch_size_t, t, :] = preds
            
        return predictions, caps, decode_lengths, sort_ind 
    
    
    def caption_image_beam_search(self, encoded_image, global_img_features, word_map, rev_word_map, beam_size=3):
        
        k = beam_size
        
        encoded_image_size = int(math.sqrt(encoded_image.size(1))) 
        
        num_pixels = encoded_image.size(1)

        encoded_image = encoded_image.expand(k, num_pixels, self.encoder_dim)
        # for k beam search
        
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  
        # initialize k seqs with <start>
        
        seqs = k_prev_words  
    
        top_k_scores = torch.zeros(k, 1).to(device)  
    
        seqs_alpha = torch.ones(k, 1, encoded_image_size, encoded_image_size).to(device)  
        seqs_betas = torch.ones(k,1,1).to(device) 
        
        # for ended seqs
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()
        complete_seqs_betas = list() 
    
        # Start decoding
        h = torch.zeros(k, self.embed_dim).to(device)
        m = torch.zeros(k, self.embed_dim).to(device)
        
        max_len = 50
        for t in range(0,max_len):
           
            embeddings = self.embedding(k_prev_words).squeeze(1)
            # squeeze for looking with just one word
            V, vg = self.project_features(encoded_image,global_img_features)
            vg = vg.expand_as(embeddings)
            
            inputX = torch.cat((embeddings,vg), dim = 1)
            
            h, m, s = self.LSTM(inputX, (h,m))
            
            c_hat, alpha, beta = self.attention(V, h, s)
            
            alpha = alpha.view(-1, encoded_image_size, encoded_image_size)
            # for attention on img 
            
            scores = self.final_layer(c_hat+h)
            scores = self.log_softmax(scores)
            scores = top_k_scores.expand_as(scores) + scores
            # get k scores
            
            if t == 0: # at the first time, all k scores are same
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True) 
                # flatten and get top k words
            
            prev_word_inds = top_k_words // self.vocab_size  # for sorting by the order of best k
            next_word_inds = top_k_words % self.vocab_size  # for sorting by the order of best word index
    
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],dim=1)  
            seqs_betas = torch.cat([seqs_betas[prev_word_inds], beta[prev_word_inds].unsqueeze(1)], dim=1)
            # update seqs, alphas and betas based on topping best k 
    
            # remove ended seqs
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
    
            # update ended seqs
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds])
                complete_seqs_betas.extend(seqs_betas[complete_inds])
                complete_seqs_scores.extend(top_k_scores[complete_inds])
                
                k -= len(complete_inds)  
                
            # Proceed with incomplete sequences
            if k == 0:
                break
            
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            seqs_betas = seqs_betas[incomplete_inds]
            # update seqs, alphas and betas 
            
            h = h[prev_word_inds[incomplete_inds]]
            m = m[prev_word_inds[incomplete_inds]]
            encoded_image = encoded_image[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            # update based on topping best k 
                
        best_seq_index = complete_seqs_scores.index(max(complete_seqs_scores))
        # get index of best seq
        alphas = complete_seqs_alpha[best_seq_index].cpu().detach().numpy()
        betas = complete_seqs_betas[best_seq_index].cpu().detach().numpy()
        
        seq = complete_seqs[best_seq_index]
        generated_words = [rev_word_map[seq[i]] for i in range(len(seq))][1:-1]
        generated_sentence = ' '.join([word for word in generated_words])
        
        print("Image said that \" ",generated_sentence,"\" ")
        
        return seq, alphas, betas
        
    def visualize_with_att(self, image_path, seq, alphas, betas, rev_word_map):
        image = Image.open(image_path)
        image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
    
        words = [rev_word_map[ind] for ind in seq]
    
        for t in range(len(words)):
            plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)
            plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=8)
            plt.text(250, 1, '%.2f' % (1-(betas[t].item())), color='red', backgroundcolor='white', fontsize=8)
            plt.imshow(image)
            current_alpha = alphas[t, :]
            alpha = skimage.transform.resize(current_alpha, [14 * 24, 14 * 24])
            if t == 0 or t == (len(words) -1):
                plt.imshow(alpha, alpha=0)
            else:
                plt.imshow(alpha, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
            
        plt.show()

class Attention(nn.Module):
    def __init__(self, hidden_size, att_dim):
        super(Attention,self).__init__()
        self.project_h = nn.Linear(hidden_size, att_dim)
        self.project_s = nn.Linear(hidden_size, att_dim)
        self.project_V = nn.Linear(hidden_size, att_dim)
        
        self.project_att  = nn.Linear(att_dim,1)
        self.project_att_hat  = nn.Linear(att_dim,1)
        
        self.alphas = nn.Linear(att_dim, 1)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  
        

    def forward(self, V, h_t, s_t):
        WV = self.project_V(V)
        Wh = self.project_h(h_t) 
        z_t = self.project_att(self.tanh(WV+Wh.unsqueeze(1))).squeeze(-1)
        # (6)
        alpha_t = self.softmax(z_t)
        # (7)
        
        c_t = (V * alpha_t.unsqueeze(-1)).sum(dim=1)
        # (8)
        
        Ws = self.project_s(s_t)
        z_t_hat = self.project_att_hat(self.tanh(Ws+Wh))        
        alpha_t_hat = self.softmax(torch.cat([z_t,z_t_hat],dim = 1))
        # (12)
        beta_t = alpha_t_hat[:,-1].unsqueeze(1)
        
        c_t_hat = beta_t*s_t + (1-beta_t)*c_t
        # (11)

        return c_t_hat, alpha_t, beta_t
