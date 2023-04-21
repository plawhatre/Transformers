from positional_encoding import PositionalEncoding
import torch.nn as nn
import torch

class SentenceEmbedding(nn.Module):
    def __init__(self, batch_size, max_seq_len, d_model, vocab):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(batch_size, max_seq_len, d_model)
        self.dropout = nn.Dropout(p=0.1)

    def tokenize(self, batch_sent):
        X = []
        for sent in batch_sent:
            idx_sent = [self.vocab[word] for word in sent.split()]
        
            len_sent = len(idx_sent)
            if  len_sent < self.max_seq_len:
                idx_sent = idx_sent + [0]*(self.max_seq_len - len_sent)
            
            X.append(idx_sent)
        return torch.LongTensor(X)

    def forward(self, x):
        y = self.tokenize(x)
        y = self.embedding(y)
        y += self.pos_encoding()
        y = self.dropout(y)
        return y
