import torch.nn as nn
from positional_encoding import PositionalEncoding
from encoder_layer import StackedEncoder
from decoder_layer import StackedDecoder
from sentence_embedding import SentenceEmbedding
import torch.nn.functional as F
import torch


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


    def forward(self, src_lang_sent, dst_lang_sent):
        # Add START and END token
        dst_lang_sent = self.dst_sent_encode.add_start_end_token(dst_lang_sent)

        # Sentence encoding
        x, _ = self.src_sent_encode(src_lang_sent)
        y, y_token = self.dst_sent_encode(dst_lang_sent)

        # positional encoding
        x_encoding = PositionalEncoding(*x.shape)()
        y_encoding = PositionalEncoding(*y.shape)()

        # masking
        encoder_mask = self.src_sent_encode.create_encoder_mask(src_lang_sent)
        decoder_mask = self.dst_sent_encode.create_decoder_mask(dst_lang_sent)
        encoder_decoder_mask = self.src_sent_encode.\
            create_encoder_decoder_mask(src_lang_sent, dst_lang_sent)
        
        x_out = self.encoder(x + x_encoding, encoder_mask)
        out = self.decoder(y + y_encoding, x_out, decoder_mask, encoder_decoder_mask)
        out = self.linear(out)

        # true translation for loss computation
        token_mask = (y_token!=0) * 1
        ignore_padding_mask = (y_token == 0)
        
        y_onehot_padded = F.one_hot(y_token - token_mask, num_classes=len(self.dst_vocab)) 
        y_onehot = torch.where(ignore_padding_mask.unsqueeze(-1), 
                               torch.zeros_like(y_onehot_padded), 
                               y_onehot_padded)
        return out, y_onehot