import torch.nn as nn
from attention import Attention

class MultiHeadAttention(nn.Module, Attention):
  def __init__(self, inp_dim, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.inp_dim = inp_dim
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads

    self.qkv_layer = nn.Linear(inp_dim, 3*d_model)
    self.linear_layer = nn.Linear(d_model, d_model)

  def forward(self, x, mask=None):
    batch_size, seq_length, _ = x.size()

    qkv = self.qkv_layer(x)
    qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
    qkv = qkv.permute([0, 2, 1, 3])
    q, k, v = qkv.chunk(3, axis=-1)

    values, attention = self.scaled_dot_product(q, k, v, mask)
    values = values.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.num_heads * self.head_dim)
    values = self.linear_layer(values)
    return values