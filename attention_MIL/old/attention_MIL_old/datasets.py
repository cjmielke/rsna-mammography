import numpy as np
import tables
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, labelDF):
        #self.args = args
        self.labelDF = labelDF
        self.tensorCache = dict()           # for positive tensors only

    def getTensor(self, row, item):
        if not row.cancer:
            return torch.load(row.tensorFile)
        else:
            tensor = self.tensorCache.get(item, None)
            if tensor is None:
                tensor = torch.load(row.tensorFile)
                self.tensorCache[item] = tensor
            return tensor

    def __getitem__(self, item):
        row = self.labelDF.iloc[item]
        #print(item)
        #print(row)
        #tensor = torch.load(row.tensorFile)
        tensor = self.getTensor(row, item)
        cancer = np.array([row.cancer]).astype('float32')
        #cancer = row.cancer*1.0
        return tensor, cancer

    def __len__(self):
        return len(self.labelDF)
    '''
    def __iter__(self):
        self.labelDF = self.labelDF.sample(frac=1)
        #for rn, row in self.labelDF.sample(frac=1).iterrows():
        for i in range(len(self)):
            yield self[i]

    '''


class EmbeddingDatasetH5(Dataset):
    def __init__(self, labelDF):
        #self.args = args
        self.labelDF = labelDF
        self.h5 = tables.file

    def getTensor(self, row, item):
        if not row.cancer:
            return torch.load(row.tensorFile)
        else:
            tensor = self.tensorCache.get(item, None)
            if tensor is None:
                tensor = torch.load(row.tensorFile)
                self.tensorCache[item] = tensor
            return tensor

    def __getitem__(self, item):
        row = self.labelDF.iloc[item]
        #print(item)
        #print(row)
        #tensor = torch.load(row.tensorFile)
        tensor = self.getTensor(row, item)
        cancer = np.array([row.cancer]).astype('float32')
        #cancer = row.cancer*1.0
        return tensor, cancer

    def __len__(self):
        return len(self.labelDF)
    '''
    def __iter__(self):
        self.labelDF = self.labelDF.sample(frac=1)
        #for rn, row in self.labelDF.sample(frac=1).iterrows():
        for i in range(len(self)):
            yield self[i]

    '''



def collate(batch):
    embs = [item[0] for item in batch]
    lbls = [item[1] for item in batch]
    #print(embs[0])     # tensor already, presumably from torch.load

    #embeddings = torch.cat(embs, 0)
    #embeddings = torch.stack(embs)                  # cant do this, ragged!
    embeddings = embs                                 # can we just return a list?
    #labels = torch.cat(lbls, 0)
    labels = torch.from_numpy(np.concatenate(lbls))
    #print(embeddings.shape, labels.shape)
    #print(labels)
    return embeddings, labels
