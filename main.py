import yaml
from glob import glob

from torch.utils.data import DataLoader
from text_dataset import TextDataset
from transformer_model import Transformer

if __name__ == '__main__':
    # Read yaml file and load params
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    inp_dim = params['global']['inp_dim']
    d_model = params['global']['d_model']
    d_hidden = params['global']['d_hidden']
    num_heads = params['global']['num_heads']
    p_drop = params['global']['p_drop']
    Nx = params['global']['Nx']

    batch_size = params['training']['batch_size']
    lang = str(params['training']['lang'])
    sent_limit = params['training']['sent_limit']
    max_seq_len = params['training']['max_seq_len']

    # Load data
    src_folder = glob(f"data/v2/*{lang}/*en")[0]
    dst_folder = glob(f"data/v2/*{lang}/*{lang}")[0]

    with open(src_folder, 'r') as f:
        src_sent = f.readlines() 

    with open(dst_folder, 'r') as f:
        dst_sent = f.readlines() 


    src_sent = src_sent[:sent_limit]
    dst_sent = dst_sent[:sent_limit]

    src_sent = [sent.strip('\n') for sent in src_sent]
    dst_sent = [sent.strip('\n') for sent in dst_sent]

    train_dataset = TextDataset(src_sent, dst_sent, max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # temporary
    src_lang_sent, dst_lang_sent =  next(iter(train_loader))
    model = Transformer(batch_size, max_seq_len, d_model, Nx, 
                        inp_dim, d_hidden, num_heads, p_drop, 
                        train_dataset.get_src_vocab, train_dataset.get_dst_vocab)
    out = model(src_lang_sent, dst_lang_sent)

    print(out)


