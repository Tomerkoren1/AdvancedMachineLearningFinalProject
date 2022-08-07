import torch

class PairSentencesDataset(torch.utils.data.Dataset):
    '''
    Return 2 sentences and semantic similarity label between the 2 sentences
    '''
    def __init__(self, sentences1, sentences2, labels):
        self.sentences1  = sentences1
        self.sentences2  = sentences2
        self.labels = labels

    def __getitem__(self, idx):
        sen1 = self.sentences1[idx].lower()
        sen2 = self.sentences2[idx].lower()
        label = self.labels[idx]
        return sen1, sen2, label

    def __len__(self):
        return len(self.labels)