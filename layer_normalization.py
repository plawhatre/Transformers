import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
  def __init__(self, d_model, eps=1e-5):
    super(LayerNormalization, self).__init__()
    self.parameter_shape = d_model
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(self.parameter_shape))
    self.beta = nn.Parameter(torch.zeros(self.parameter_shape))

  def forward(self, x):
    means = x.mean(dim=[-1, -2], keepdim=True)
    vars = ((x - means)**2).mean(dim=[-1, -2], keepdim=True)
    stds = (vars + self.eps).sqrt()
    y = (x - means) / stds
    out = self.gamma * y + self.beta
    return out