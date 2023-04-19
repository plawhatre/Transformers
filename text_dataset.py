from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, src_sent, dst_sent):
        self.src_sent = src_sent
        self.dst_sent = dst_sent
        self.src_vocab = self._build_vocab(self.src_sent)
        self.dst_vocab = self._build_vocab(self.dst_sent)

    def _build_vocab(self, sentences):
        vocab = {}
        unique_id = 0
        for sentence in sentences:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = unique_id
                    unique_id += 1
        return vocab

    def __len__(self):
        return len(self.src_sent)

    def __getitem__(self, idx):
        src_sent = [self.src_vocab[word] for word in self.src_sent[idx].split()]
        dst_sent = [self.dst_vocab[word] for word in self.dst_sent[idx].split()]
        return src_sent, dst_sent