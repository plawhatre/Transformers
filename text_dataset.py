from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, src_sent, dst_sent, max_sent_length=50, start_token=True, end_token=True):
        self.src_sent = src_sent
        self.dst_sent = dst_sent
        self.max_sent_length = max_sent_length
        self.start_token = start_token
        self.end_token = end_token
        print("\x1B[32mDataset created\x1B[0m")

    @property
    def get_src_vocab(self):
        return TextDataset._build_vocab(self.src_sent, self.start_token, self.end_token)

    @property
    def get_dst_vocab(self):
        return TextDataset._build_vocab(self.dst_sent, self.start_token, self.end_token)

    @staticmethod
    def _build_vocab(sentences, start_token, end_token):
        vocab = {}
        if start_token:
            vocab['START'] = 1
            unique_id = 2
        else:
            unique_id = 1

        for sentence in sentences:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = unique_id
                    unique_id += 1

        vocab['UNKN'] = unique_id

        if end_token:
            vocab['END'] = unique_id + 1
            
        return vocab

    def __len__(self):
        return len(self.src_sent)

    def __getitem__(self, idx):
        return self.src_sent[idx], self.dst_sent[idx]