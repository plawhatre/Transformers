import torch
import math
import torch.nn.functional as F

class Attention:  
  @staticmethod
  def scaled_dot_product(q, k, v, mask=None):
    d_k= k.size()[-1]
    scaled = q@k.transpose(-2,-1) / math.sqrt(d_k)

    if mask is not None:
      mask = mask.unsqueeze(1).repeat(1, scaled.shape[1],1, 1)
      scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = attention@v
    return values, attention
