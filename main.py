import torch
import yaml
from positional_encoding import PositionalEncoding
from encoder_layer import StackedEncoder
from decoder_layer import StackedDecoder

if __name__ == '__main__':
    # Read yaml file and load params
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    batch_size = params['global']['batch_size']
    inp_dim = params['global']['inp_dim']
    d_model = params['global']['d_model']
    d_hidden = params['global']['d_hidden']
    num_heads = params['global']['num_heads']
    max_seq_len = params['global']['max_seq_len']
    p_drop = params['global']['p_drop']
    Nx = params['global']['Nx']

    # Sentence for encoder block
    x = torch.randn((batch_size, max_seq_len, inp_dim))
    # Sentence for decoder block
    y = torch.randn((batch_size, max_seq_len, inp_dim))

    # positional encoding
    pos_encoding = PositionalEncoding(batch_size, max_seq_len, inp_dim)

    # forward pass
    encoder = StackedEncoder(Nx, inp_dim, d_model, d_hidden, num_heads, p_drop, eps=1e-5)
    decoder = StackedDecoder(Nx, inp_dim, d_model, d_hidden, num_heads, p_drop, eps=1e-5)
    
    x_out = encoder(x+pos_encoding())
    out = decoder(y+pos_encoding(), x_out)
    print(out)


