from positional_encoding import PositionalEncoding
import torch.nn as nn
import torch

NEG_INFTY = -1e9

class SentenceEmbedding(nn.Module):
    def __init__(self, batch_size, max_seq_len, d_model, vocab):
        super().__init__()
        self.vocab = vocab
        self.batch_size = batch_size
        self.vocab_size = len(vocab) + 1
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(batch_size, max_seq_len, d_model)
        self.dropout = nn.Dropout(p=0.1)

    def tokenize(self, batch_sent):
        X = []
        for sent in batch_sent:
            idx_sent = [self.vocab.get(word, self.vocab["UNKN"]) for word in sent.split()]
        
            len_sent = len(idx_sent)
            if  len_sent < self.max_seq_len:
                idx_sent = idx_sent + [0]*(self.max_seq_len - len_sent)
            else:
                idx_sent = idx_sent[:self.max_seq_len]
            
            X.append(idx_sent)

        return torch.LongTensor(X)
        
    @staticmethod
    def add_start_end_token(batch_sent):
        added_tokens_sent = ()
        for idx, sent in enumerate(batch_sent):
            added_tokens_sent += ("START " + sent + " END",)
        return added_tokens_sent

    def forward(self, x):
        y_token = self.tokenize(x)
        y = self.embedding(y_token)
        y += self.pos_encoding()
        y = self.dropout(y)
        return y, y_token
    
    def create_encoder_mask(self, batch_src_sent):
        encoder_mask = torch.full((self.batch_size,self.max_seq_len, self.max_seq_len), 
                                  NEG_INFTY)
        for idx, sent in enumerate(batch_src_sent):
            len_sent = len(sent.split())
            encoder_mask[idx, :(len_sent), :(len_sent)] = 0

        return encoder_mask

    def create_decoder_mask(self, batch_dst_sent):
        decoder_mask = torch.full((self.batch_size,self.max_seq_len, self.max_seq_len), 
                                  NEG_INFTY)
        decoder_mask = torch.triu(decoder_mask, diagonal=1)

        for idx, sent in enumerate(batch_dst_sent):
            len_sent = len(sent.split())
            decoder_mask[idx, (len_sent+1):, :] = NEG_INFTY
            decoder_mask[idx, :, (len_sent+1):] = NEG_INFTY

        return decoder_mask

    def create_encoder_decoder_mask(self, batch_src_sent, batch_dst_sent):
        encoder_decoder_mask = torch.full((self.batch_size,self.max_seq_len, self.max_seq_len), 
                                  NEG_INFTY)
        for idx, src_dst_sent in enumerate(zip(batch_src_sent, batch_dst_sent)):
            src_sent, dst_sent = src_dst_sent[0], src_dst_sent[1] 
            len_src_sent = len(src_sent.split())
            len_dst_sent = len(dst_sent.split())
            encoder_decoder_mask[idx, :(len_dst_sent), :(len_src_sent)] = 0

        return encoder_decoder_mask