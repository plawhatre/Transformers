from positional_encoding import PositionalEncoding
import torch.nn as nn

class SentenceEmbedding(nn.Module):
    def __init__(self, batch_size, max_seq_len, d_model, vocab):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(batch_size, max_seq_len, d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        y = self.embedding(x)
        y += self.pos_encoding()
        y = self.dropout(y)
        return y
