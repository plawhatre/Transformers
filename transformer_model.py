import torch.nn as nn
from positional_encoding import PositionalEncoding
from encoder_layer import StackedEncoder
from decoder_layer import StackedDecoder
from sentence_embedding import SentenceEmbedding


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
                 get_src_vocab,
                 get_dst_vocab):
        super().__init__()
        self.src_sent_encode = SentenceEmbedding(batch_size, max_seq_len, d_model, get_src_vocab)
        self.dst_sent_encode = SentenceEmbedding(batch_size, max_seq_len, d_model, get_dst_vocab)
        self.encoder = StackedEncoder(Nx, inp_dim, d_model, d_hidden, num_heads, p_drop, eps=1e-5)
        self.decoder = StackedDecoder(Nx, inp_dim, d_model, d_hidden, num_heads, p_drop, eps=1e-5)
        self.linear = nn.Linear(d_model, len(get_dst_vocab))


    def forward(self, src_lang_sent, dst_lang_sent):
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
        return out, y_token