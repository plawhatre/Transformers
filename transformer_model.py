from random import randint
import torch.nn as nn
from positional_encoding import PositionalEncoding
from encoder_layer import StackedEncoder
from decoder_layer import StackedDecoder
from sentence_embedding import SentenceEmbedding
import torch.nn.functional as F
import torch
import torch.optim as optim
import os
from datetime import datetime


class Transformer(nn.Module):
    def __init__(self, 
                 batch_size, 
                 max_seq_len, 
                 d_model, 
                 Nx, 
                 inp_dim, 
                 d_hidden, 
                 num_heads, 
                 p_drop,
                 src_vocab,
                 dst_vocab):
        super().__init__()
        self.src_vocab = src_vocab
        self.dst_vocab = dst_vocab
        self.src_sent_encode = SentenceEmbedding(batch_size, max_seq_len, d_model, src_vocab)
        self.dst_sent_encode = SentenceEmbedding(batch_size, max_seq_len, d_model, dst_vocab)
        self.encoder = StackedEncoder(Nx, inp_dim, d_model, d_hidden, num_heads, p_drop, eps=1e-5)
        self.decoder = StackedDecoder(Nx, inp_dim, d_model, d_hidden, num_heads, p_drop, eps=1e-5)
        self.linear = nn.Linear(d_model, len(dst_vocab))
        self.params = {'batch_size': batch_size,
                        'max_seq_len': max_seq_len,
                        'd_model': d_model,
                        'Nx': Nx,
                        'inp_dim': inp_dim,
                        'd_hidden': d_hidden,
                        'num_heads': num_heads,
                        'p_drop': p_drop,
                        'src_vocab': src_vocab,
                        'dst_vocab': dst_vocab}
        print("\x1B[32mModel created\x1B[0m")


    def forward(self, src_lang_sent, dst_lang_sent):
        # Add START and END token
        dst_lang_sent = self.dst_sent_encode.add_start_end_token(dst_lang_sent)

        # Sentence encoding
        x, _ = self.src_sent_encode(src_lang_sent)
        y, y_token = self.dst_sent_encode(dst_lang_sent)

        # masking
        encoder_mask = self.src_sent_encode.create_encoder_mask(src_lang_sent)
        decoder_mask = self.dst_sent_encode.create_decoder_mask(dst_lang_sent)
        encoder_decoder_mask = self.src_sent_encode.\
            create_encoder_decoder_mask(src_lang_sent, dst_lang_sent)
        
        x_out = self.encoder(x , encoder_mask)
        out = self.decoder(y, x_out, decoder_mask, encoder_decoder_mask)
        out = self.linear(out)

        # true translation for loss computation
        token_mask = (y_token!=0) * 1
        ignore_padding_mask = (y_token == 0)
        
        y_onehot_padded = F.one_hot(y_token - token_mask, num_classes=len(self.dst_vocab)) 
        y_onehot = torch.where(ignore_padding_mask.unsqueeze(-1), 
                               torch.zeros_like(y_onehot_padded), 
                               y_onehot_padded)
        return out, y_onehot
    
    def train_time_inference(self, index_sent, dst_lang_sent, output, vocab_keys, vocab_values):
        pred_sent = F.softmax(output[index_sent], dim =-1)
        pred_sent = (torch.max(pred_sent, axis=-1).indices.numpy() + 1).tolist()
        pred_train_sample = " ".join(
            [vocab_keys[val] for val in [vocab_values.index(word) 
                for word in pred_sent]]
            )
        print("--"*5 + "ORIGINAL" + "--"*5 ,"\n", '\x1B[32m', dst_lang_sent[index_sent], '\x1B[0m')
        print("--"*5 + "PREDICTED" + "--"*5 ,"\n", '\x1B[36m', pred_train_sample, '\x1B[0m')
    
    def translate(self, src_lang_sent):
        inference_batch_size = len(src_lang_sent)
        self.src_sent_encode.batch_size = inference_batch_size
        self.src_sent_encode.pos_encoding.batch_size = inference_batch_size
        self.dst_sent_encode.batch_size = inference_batch_size
        self.dst_sent_encode.pos_encoding.batch_size = inference_batch_size
        vocab_keys = list(self.dst_vocab.keys())
        vocab_values = list(self.dst_vocab.values())

        # Sentence encoding
        x, _ = self.src_sent_encode(src_lang_sent)
        # masking
        encoder_mask = self.src_sent_encode.create_encoder_mask(src_lang_sent)
        x_out = self.encoder(x, encoder_mask)

        
        i = 0
        dst_lang_sent = ["START "] * self.src_sent_encode.batch_size
        next_token_lst = ['placeholder'] * inference_batch_size

        while True:
            i += 1
            decoder_mask = self.dst_sent_encode.create_decoder_mask(dst_lang_sent)
            encoder_decoder_mask = self.src_sent_encode.\
                create_encoder_decoder_mask(src_lang_sent, dst_lang_sent)

            y, _ = self.dst_sent_encode(dst_lang_sent)
            out = self.decoder(y , x_out, decoder_mask, encoder_decoder_mask)
            out = F.softmax(self.linear(out), dim=-1)

            next_token_ind = (torch.max(out[:,i, :], axis=-1).indices.numpy() + 1).tolist()

            for idx, sent in enumerate(dst_lang_sent): 
                if next_token_lst[idx] == 'END':
                    continue
                else:
                    next_token = vocab_keys[vocab_values.index(next_token_ind[idx])]
                    next_token_lst[idx] = next_token
                    dst_lang_sent[idx] = sent + next_token + " "

            terminate_loop = all(last_token == 'END' for last_token in next_token_lst)

            if terminate_loop or i >= (out.shape[1] - 1):
                break

        return dst_lang_sent
    
    def create_checkpoint(self, epoch, optimizer, path='./model', filename=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        if filename is None:
            filename = f"checkpoint_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if not os.path.exists(path):
            os.mkdir(path)
        
        torch.save(self.params, f"{path}/model_attributes.pt")
        torch.save(checkpoint, f"{path}/{filename}.pt")

    @staticmethod
    def load_checkpoint(path='./model', filename='final_model'):
        loaded_params = torch.load(f"{path}/model_attributes.pt")
        model = Transformer(**loaded_params)
        optimizer = optim.Adam(model.parameters())
        
        checkpoint = torch.load(f"{path}/{filename}.pt")
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, epoch
        
    def train(self, epochs, train_loader, optimizer, criterion):
        # Dict keys and vals
        vocab_keys = list(self.dst_vocab.keys())
        vocab_values = list(self.dst_vocab.values())

        for epoch in range(epochs):
            running_loss = 0.0
            for iteration, data in enumerate(train_loader, 0):
                # fecthing the batch sample
                src_lang_sent, dst_lang_sent = data

                # setting grads to zero
                optimizer.zero_grad()

                # forward 
                output , y_true = self(src_lang_sent, dst_lang_sent)
                
                # backward
                loss = criterion(output, y_true.float())
                loss.backward()

                #  Update params
                optimizer.step()
                
                # stats during training
                running_loss += loss.item()
                num_print_after_iter = 10
                if iteration % num_print_after_iter == (num_print_after_iter - 1):
                    print(f"\x1B[35m[Epoch: {epoch}, Iteration: {iteration}], Loss: {running_loss/num_print_after_iter}\x1B[0m")
                    running_loss = 0.0 

                    # preds 
                    index_sent = randint(0, len(src_lang_sent) - 1)
                    self.train_time_inference(index_sent, 
                                         dst_lang_sent, 
                                         output, 
                                         vocab_keys, 
                                         vocab_values)
        return epoch

