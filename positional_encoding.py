import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
  def __init__(self, batch_size, max_seq_len, inp_dim):
    super().__init__()
    self.batch_size = batch_size
    self.max_seq_len = max_seq_len
    self.inp_dim = inp_dim

  def forward(self):
    even_i = torch.arange(0, self.inp_dim, 2).float()
    odd_i = torch.arange(1, self.inp_dim, 2).float()
    even_denominator = torch.pow(1e4, even_i/self.inp_dim)
    odd_denominator = torch.pow(1e4, (even_i - 1)/self.inp_dim)

    pos = torch.arange(self.max_seq_len, dtype=torch.float).reshape(self.max_seq_len, 1)

    even_PE = torch.sin(pos/even_denominator)
    odd_PE = torch.cos(pos/odd_denominator)

    stacked = torch.stack([even_PE, odd_PE], axis=-1)
    pe = torch.flatten(stacked, start_dim=1, end_dim=2)
    pe = pe.expand(self.batch_size, self.max_seq_len, self.inp_dim)

    return pe