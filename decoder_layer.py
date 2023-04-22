import torch.nn as nn
from layer_normalization import LayerNormalization
from feedforward_net import FeedForwardNetwork
from multihead_attention import MultiHeadAttention
from multihead_cross_attention import MultiHeadCrossAttention

class DecoderLayer(nn.Module):
  def __init__(self, inp_dim, d_model, d_hidden, num_heads, p_drop, eps=1e-5):
    super(DecoderLayer, self).__init__()
    self.attention = MultiHeadAttention(inp_dim, d_model, num_heads)
    self.dropout_attention = nn.Dropout(p=p_drop)
    self.add_norm_attention = LayerNormalization(d_model, eps)

    self.cross_attention = MultiHeadCrossAttention(inp_dim, d_model, num_heads)
    self.dropout_cross_attention = nn.Dropout(p=p_drop)
    self.add_norm_cross_attention = LayerNormalization(d_model, eps)

    self.ffn = FeedForwardNetwork(d_model, d_hidden)
    self.dropout_ffn = nn.Dropout(p=p_drop) 
    self.add_norm_ffn = LayerNormalization(d_model, eps)

  def forward(self, x, encoder_out, decoder_mask, encoder_decoder_mask):
    batch_size, seq_length, _ = x.size()
    x_residual = x.clone()
    x = self.attention(x, mask=decoder_mask)
    x = self.dropout_attention(x)
    x = self.add_norm_attention(x + x_residual)

    x_residual = x.clone()
    x = self.cross_attention(x,
                             encoder_out, 
                             mask=encoder_decoder_mask)
    x = self.dropout_cross_attention(x)
    x = self.add_norm_cross_attention(x + x_residual)


    x_residual = x.clone()
    x = self.ffn(x)
    x = self.dropout_ffn(x)
    x = self.add_norm_ffn(x + x_residual)

    return x
  
class SequentialDecoder(nn.Sequential):
  def forward(self, *args):
    x, encoder_out, decoder_mask, encoder_decoder_mask = args
    for module in self._modules.values():
      x = module(x, encoder_out, decoder_mask, encoder_decoder_mask)
    return x

class StackedDecoder(nn.Module):
  def __init__(self, Nx, inp_dim, d_model, d_hidden, num_heads, p_drop, eps=1e-5):
    super(StackedDecoder, self).__init__()
    self.layers = SequentialDecoder(*[DecoderLayer(inp_dim, d_model, d_hidden, num_heads, p_drop, eps=1e-5) for _ in range(Nx)])

  def forward(self, x, encoder_out, decoder_mask, encoder_decoder_mask):
    return self.layers(x, encoder_out, decoder_mask, encoder_decoder_mask)