import torch
import math
import torch.nn.functional as F

class Attention:
  @staticmethod
  def create_mask(batch_size, seq_length):
    mask = torch.full((batch_size,seq_length, seq_length), -math.inf)
    mask = torch.triu(mask, diagonal=1)
    return mask
    
  @staticmethod
  def scaled_dot_product(q, k, v, mask=None):
    d_k= k.size()[-1]
    scaled = q@k.transpose(-2,-1) / math.sqrt(d_k)

    if mask is not None:
      scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = attention@v
    return values, attention
