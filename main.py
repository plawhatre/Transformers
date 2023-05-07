import os
import yaml
from glob import glob
from random import randint

from torch.utils.data import DataLoader
from text_dataset import TextDataset
from transformer_model import Transformer
import torch
import torch.nn.functional as F
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
    lr = float(params['training']['lr'])
    beta1 = params['training']['beta1']
    beta2 = params['training']['beta2']
    eps = float(params['training']['eps'])
    training_flag = params['training']['flag']
    save_model = params['training']['save_model']
    save_path = params['training']['save_path']

    inference_flag = params['inference']['flag']
    load_path = params['inference']['load_path']


    if training_flag:
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
        optimizer = optim.Adam(model.parameters() ,lr=lr, betas=(beta1, beta2), eps=eps)

        # Dict keys and vals
        vocab_keys = list(model.dst_vocab.keys())
        vocab_values = list(model.dst_vocab.values())

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
                num_print_after_iter = 10
                if iteration % num_print_after_iter == (num_print_after_iter - 1):
                    print(f"[Epoch: {epoch}, Iteration: {iteration}], Loss: {running_loss/num_print_after_iter}")
                    running_loss = 0.0 

                    # preds 
                    index_sent = randint(0, len(src_lang_sent))
                    pred_sent = F.softmax(output[index_sent], dim =-1)
                    pred_sent = torch.max(pred_sent, axis=-1).indices.numpy().tolist()
                    pred_train_sample = " ".join(
                        [vocab_keys[val] for val in [vocab_values.index(word) 
                            for word in pred_sent]]
                        )
                    print("--"*5 + "ORIGINAL" + "--"*5 ,"\n", src_lang_sent[index_sent])
                    print("--"*5 + "PREDICTED" + "--"*5 ,"\n", pred_train_sample)

        print("Training Finished")

        model.save_model(save_path)

    # Translate
    if inference_flag:
        model = Transformer.load_model(load_path)
        translated_out = model.translate(["The court has fixed a hearing for February 12", 
                                          "fixed a hearing"])
        for sentence in translated_out:
            print(sentence)