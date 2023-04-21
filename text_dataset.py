from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, src_sent, dst_sent, max_sent_length=50):
        self.src_sent = src_sent
        self.dst_sent = dst_sent
        self.max_sent_length = max_sent_length

    @property
    def get_src_vocab(self):
        return TextDataset._build_vocab(self.src_sent)

    @property
    def get_dst_vocab(self):
        return TextDataset._build_vocab(self.dst_sent)

    @staticmethod
    def _build_vocab(sentences):
        vocab = {}
        unique_id = 1
        for sentence in sentences:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = unique_id
                    unique_id += 1
        return vocab

    def __len__(self):
        return len(self.src_sent)

    def __getitem__(self, idx):
        return self.src_sent[idx], self.dst_sent[idx]