import torch.nn as nn
from multihead_attention import MultiHeadAttention
from layer_normalization import LayerNormalization
from feedforward_net import FeedForwardNetwork

class EncoderLayer(nn.Module):
  def __init__(self, inp_dim, d_model, d_hidden, num_heads, p_drop, eps=1e-5):
    super(EncoderLayer, self).__init__()
    self.attention = MultiHeadAttention(inp_dim, d_model, num_heads)
    self.dropout_attention = nn.Dropout(p=p_drop)
    self.add_norm_attention = LayerNormalization(d_model, eps)
    self.ffn = FeedForwardNetwork(d_model, d_hidden)
    self.dropout_ffn = nn.Dropout(p=p_drop) 
    self.add_norm_ffn = LayerNormalization(d_model, eps)

  def forward(self, x, encoder_mask):
    x_residual = x.clone()
    x = self.attention(x, mask=encoder_mask)
    x = self.dropout_attention(x)
    x = self.add_norm_attention(x + x_residual)

    x_residual = x.clone()
    x = self.ffn(x)
    x = self.dropout_ffn(x)
    x = self.add_norm_ffn(x + x_residual)

    return x

class SequentialEncoder(nn.Sequential):
  def forward(self, *args):
    x, encoder_out = args
    for module in self._modules.values():
      x = module(x, encoder_out)
    return x

class StackedEncoder(nn.Module):
  def __init__(self, Nx, inp_dim, d_model, d_hidden, num_heads, p_drop, eps=1e-5):
    super(StackedEncoder, self).__init__()
    self.layers = SequentialEncoder(*[EncoderLayer(inp_dim, d_model, d_hidden, num_heads, p_drop, eps=eps) for _ in range(Nx)])

  def forward(self, x, encoder_mask):
    return self.layers(x, encoder_mask)