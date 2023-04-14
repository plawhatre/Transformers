from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, src_sent, dst_sent):
        self.src_sent = src_sent
        self.dst_sent = dst_sent

    def __len__(self):
        return len(self.src_sent)

    def __getitem__(self, idx):
        return self.src_sent[idx], self.dst_sent[idx]