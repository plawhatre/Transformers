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

    print("\x1B[33mParams loaded\x1B[0m")

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

        print("\x1B[34mStarting training\x1B[0m")

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
                    print(f"\x1B[35m[Epoch: {epoch}, Iteration: {iteration}], Loss: {running_loss/num_print_after_iter}\x1B[0m")
                    running_loss = 0.0 

                    # preds 
                    index_sent = randint(0, len(src_lang_sent) - 1)
                    model.train_time_inference(index_sent, 
                                         dst_lang_sent, 
                                         output, 
                                         vocab_keys, 
                                         vocab_values)
        print("\x1B[36mTraining Finished\x1B[0m")

        save_model_name = 'final_model'        
        model.create_checkpoint(epoch, optimizer, save_path, save_model_name)
        print(f"\x1B[33mModel saved at {save_path}\x1B[0m")

    # Translate
    if inference_flag:
        saved_model_name = 'final_model'
        model, _, _ = Transformer.load_checkpoint(load_path, saved_model_name)
        print(f"\x1B[34mModel loaded from {load_path}\x1B[0m")

        translate_sent = ["The court has fixed a hearing for February 12",
             "Please select the position where the track should be",
             "Joseph dreamed a dream, and he told it to his brothers"]

        print("\x1B[35mTranslating input text\x1B[0m")
        translated_out = model.translate(translate_sent)
        for idx, sentence in enumerate(translated_out):
            if idx %2 == 0:
                print(f"\x1B[32m{sentence}\x1B[0m")
            else:
                print(f"\x1B[36m{sentence}\x1B[0m")