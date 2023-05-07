import torch.nn as nn
from positional_encoding import PositionalEncoding
from encoder_layer import StackedEncoder
from decoder_layer import StackedDecoder
from sentence_embedding import SentenceEmbedding
import torch.nn.functional as F
import torch
import os


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

        while True:
            i += 1
            decoder_mask = self.dst_sent_encode.create_decoder_mask(dst_lang_sent)
            encoder_decoder_mask = self.src_sent_encode.\
                create_encoder_decoder_mask(src_lang_sent, dst_lang_sent)

            y, _ = self.dst_sent_encode(dst_lang_sent)
            out = self.decoder(y , x_out, decoder_mask, encoder_decoder_mask)
            out = F.softmax(self.linear(out), dim=-1)

            next_token_ind = (torch.max(out[:,i, :], axis=-1).indices.numpy() + 1).tolist()
            next_token_lst = ['placeholder', 'placeholder']

            for idx, sent in enumerate(dst_lang_sent): 
                if next_token_lst[idx] == 'STOP':
                    continue
                else:
                    next_token = vocab_keys[vocab_values.index(next_token_ind[idx])]
                    next_token_lst[idx] = next_token
                    dst_lang_sent[idx] = sent + next_token + " "

            terminate_loop = all(last_token == 'STOP' for last_token in next_token_lst)

            if terminate_loop or i >= (out.shape[1] - 1):
                break

        return dst_lang_sent
            

    def save_model(self, path='./model'):
        if not os.path.exists(path):
            os.mkdir(path)

        torch.save(self.params, path+'/nmt_params.pt') 
        torch.save(self.state_dict(), path+'/nmt.pt')

    @staticmethod
    def load_model(path):
        loaded_params = torch.load(path+'/nmt_params.pt')
        model = Transformer(**loaded_params)
        model.load_state_dict(torch.load(path+'/nmt.pt'))
        model.eval()
        return model
