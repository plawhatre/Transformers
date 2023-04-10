import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 
import yaml

if __name__ == '__main__':
    # Read yaml file and load params
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    batch_size = params['global']['batch_size']
    seq_length = params['global']['seq_length']
    inp_dim = params['global']['inp_dim']
    d_model = params['global']['d_model']
    d_hidden = params['global']['d_hidden']
    num_heads = params['global']['num_heads']
    max_seq_len = params['global']['max_seq_len']
    p_drop = params['global']['p_drop']
    Nx = params['global']['Nx']

    # Sentence for encoder block
    x = torch.randn((batch_size, seq_length, inp_dim))
    # Sentence for decoder block
    y = torch.randn((batch_size, seq_length, inp_dim))


