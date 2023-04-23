import yaml
from glob import glob

from torch.utils.data import DataLoader
from text_dataset import TextDataset
from transformer_model import Transformer
import torch.optim as optim
import torch.nn as nn

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
    epochs = params['training']['epochs']
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

    # Dataset
    train_dataset = TextDataset(src_sent, dst_sent, max_seq_len)
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              drop_last=True)

    # Model, Criterion, Optimizer
    model = Transformer(batch_size, max_seq_len, d_model, Nx, 
                        inp_dim, d_hidden, num_heads, p_drop, 
                        train_dataset.get_src_vocab, train_dataset.get_dst_vocab)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters() ,lr=1e-4, betas=(0.9, 0.98), eps=1e-9)


    # Training
    for epoch in range(epochs):
        running_loss = 0.0
        for iteration, data in enumerate(train_loader, 0):
            # fecthing the batch sample
            src_lang_sent, dst_lang_sent = data

            # setting grads to zero
            optimizer.zero_grad()

            # forward 
            output , y_true = model(src_lang_sent, dst_lang_sent)
            
            # backward
            loss = criterion(output, y_true.float())
            loss.backward()

            #  Update params
            optimizer.step()
            
            # stats during training
            running_loss += loss.item()
            if iteration % 5 == 0:
                print(f"[Epoch: {epoch}, Iteration: {iteration}], Loss: {running_loss/10}")
                running_loss = 0.0

    print("Training Finished")