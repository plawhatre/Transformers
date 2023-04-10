import torch.nn as nn

class FeedForwardNetwork(nn.Module):
  def __init__(self, d_model, d_hidden):
    super(FeedForwardNetwork, self).__init__()
    self.layer1 = nn.Linear(d_model, d_hidden)
    self.actv = nn.ReLU()
    self.layer2 = nn.Linear(d_hidden, d_model)

  def forward(self, x):
    x = self.layer1(x)
    x = self.actv(x)
    x = self.layer2(x)

    return x