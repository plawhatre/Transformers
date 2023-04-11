import torch.nn as nn
from attention import Attention

class MultiHeadCrossAttention(nn.Module, Attention):
  def __init__(self, inp_dim, d_model, num_heads):
    super(MultiHeadCrossAttention, self).__init__()
    self.inp_dim = inp_dim
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads

    self.kv_layer = nn.Linear(d_model, 2 * d_model, bias=False)
    self.q_layer = nn.Linear(d_model, d_model, bias=False)
    self.linear_layer = nn.Linear(d_model, d_model)

  def forward(self, x, encoder_out, mask=None):
    batch_size, seq_length, _ = x.size()

    kv = self.kv_layer(encoder_out)
    q = self.q_layer(x)
    kv = kv.reshape(batch_size, seq_length, self.num_heads, 2 * self.head_dim)
    q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
    kv = kv.permute(0, 2, 1, 3)
    q = q.permute(0, 2, 1, 3) 
    k, v = kv.chunk(2, axis=-1)

    values, attention = self.scaled_dot_product(q, k, v, mask)
    values = values.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.num_heads * self.head_dim)
    values = self.linear_layer(values)
    return values