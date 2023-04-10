import torch

class PositionalEncoding:
  def __init__(self, max_seq_len, d_model):
    self.max_seq_len = max_seq_len
    self.d_model = d_model

  def __call__(self):
    even_i = torch.arange(0, self.d_model, 2).float()
    odd_i = torch.arange(1, self.d_model, 2).float()
    even_denominator = torch.pow(1e4, even_i/self.d_model)
    odd_denominator = torch.pow(1e4, (even_i - 1)/self.d_model)

    pos = torch.arange(self.max_seq_len, dtype=torch.float).reshape(self.max_seq_len, 1)

    even_PE = torch.sin(pos/even_denominator)
    odd_PE = torch.cos(pos/odd_denominator)

    stacked = torch.stack([even_PE, odd_PE], axis=-1)
    pe = torch.flatten(stacked, start_dim=1, end_dim=2)

    return pe