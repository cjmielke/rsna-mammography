from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T


def makeSplit(labelDF):
    '''
    Splits dataset into predefined training/validation patients
    '''
    trainPatients = pd.read_csv('/fast/rsna-breast/trainingSplit.csv')
    valPatients = pd.read_csv('/fast/rsna-breast/validationSplit.csv')

    if 'patient_id' in labelDF.columns:
        trainDF = labelDF[labelDF.patient_id.isin(set(trainPatients.patient_id))]
        valDF = labelDF[labelDF.patient_id.isin(set(valPatients.patient_id))]
        # verify no overlap between patients
        assert len(set(trainDF.patient_id).intersection(set(valDF.patient_id))) == 0
    elif 'ptID' in labelDF.columns:
        trainDF = labelDF[labelDF.ptID.isin(set(trainPatients.patient_id))]
        valDF = labelDF[labelDF.ptID.isin(set(valPatients.patient_id))]
        # verify no overlap between patients
        assert len(set(trainDF.ptID).intersection(set(valDF.ptID))) == 0
    else:
        raise ValueError

    print(f'Train/Val shapes : {trainDF.shape} / {valDF.shape}')

    return trainDF, valDF


def getDatasets():
    #labelDF = pd.read_feather('/fast/rsna-breast/tables/tile_224_stats_sorted.feather')     # DUMB!
    labelDF = pd.read_csv('/fast/rsna-breast/train.csv')[['patient_id', 'image_id', 'cancer']]
    trainDF, valDF = makeSplit(labelDF)

    class_count = list(trainDF.cancer.value_counts())
    print(f'class_count : {class_count}')
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    print(f'class_weights : {class_weights}')
    lbl = list(trainDF.cancer)
    #print(lbl)
    #c

    class_weights_all = class_weights[lbl]
    trainDF['weight'] = class_weights_all


    # these are dataframes of tiles
    # FIXME - consider weighted samping to increase preponderance of cancer pts
    # TODO - to accomplish this, load train.csv to get labels and merge on image_id ...

    return trainDF, valDF



class EmbeddingDataset(Dataset):
    def __init__(self, labelDF, encoder, noise=0):
        #self.args = args
        self.noise = noise
        self.encoder = encoder
        self.labelDF = labelDF.astype(int)
        self.tensorCache = dict()           # for positive tensors only

    def getTensor(self, row, item):
        #print(row)
        tensorFile = f'/fast/rsna-breast/features/224/{self.encoder}/{row.patient_id}/{row.image_id}.pt'
        return torch.load(tensorFile)

    def __getitem__(self, item):
        row = self.labelDF.iloc[item]
        tensor = self.getTensor(row, item)
        #cancer = np.array([row.cancer]).astype('float32')
        if self.noise:
            # because this is in-place - cached tensors are broken slowly over time, which are all the positive cases!
            #tensor += self.noise*torch.randn(tensor.shape)
            # this should create a new one, and leave the cached tensor alone
            tensor = tensor + self.noise * torch.randn(tensor.shape)
        #cancer = row.cancer*1.0
        #return tensor, cancer

        return tensor

    def __len__(self):
        return len(self.labelDF)




def collate(batch):

    #embs = [item[0] for item in batch]
    embs = batch
    embeddings = embs                                 # can we just return a list?
    #labels = torch.cat(lbls, 0)
    #return embeddings, embeddings
    return embeddings


if __name__=='__main__':
    getTables()








