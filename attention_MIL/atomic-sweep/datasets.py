from glob import glob
from random import shuffle

import numpy as np
import pandas as pd
import tables
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
import pyarrow as pa
from misc import getPtImgIDs


def readArrow(fileName):
    source = pa.memory_map(fileName, 'r')
    table = pa.ipc.RecordBatchStreamReader(source).read_all()
    rows = []
    for i in range(table.num_columns):
        rows.append(table.column(i).to_numpy())
    return np.stack(rows)


def makeSplit(labelDF):
    '''
    labelDF = labelDF.sample(frac=1)  # shuffle!
    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7)
    split = splitter.split(labelDF, groups=labelDF['patient_id'])
    train_inds, val_inds = next(split)

    trainDF = labelDF.iloc[train_inds]
    valDF = labelDF.iloc[val_inds]
    '''
    trainPatients = pd.read_csv('/fast/rsna-breast/trainingSplit.csv')
    valPatients = pd.read_csv('/fast/rsna-breast/validationSplit.csv')

    trainDF = labelDF[labelDF.patient_id.isin(set(trainPatients.patient_id))]
    valDF = labelDF[labelDF.patient_id.isin(set(valPatients.patient_id))]

    # verify no overlap between patients
    assert len(set(trainDF.patient_id).intersection(set(valDF.patient_id))) == 0
    print(f'Train/Val shapes : {trainDF.shape} / {valDF.shape}')

    return trainDF, valDF



def getH5Dataset(h5file):
    labelDF = h5file.replace('.h5','_train.csv')
    #labelDF = f'/fast/rsna-breast/features/{encoder}_train.csv'
    labelDF = pd.read_csv(labelDF)
    trainDF, valDF = makeSplit(labelDF)
    return labelDF, trainDF, valDF

def getTensorFileDataset(encoder):
    # gather tensor files and join into dataframe - then split into train and validation
    labelDF = pd.read_csv('/fast/rsna-breast/train.csv')
    tensorFiles = glob(f'/fast/rsna-breast/features/{encoder}/*/*.pt')
    #tensorFiles = glob(f'/fast/rsna-breast/features/{encoder}/*/*.arrow')
    rows = []
    for tf in tensorFiles:
        ptID, imgID = getPtImgIDs(tf)
        #FIXME - this is silly, why add tensorFile to the dataframe? Just uses memory when it can be interpolated later
        #rows.append(dict(patient_id=ptID, image_id=imgID, tensorFile=tf))
        rows.append(dict(patient_id=ptID, image_id=imgID, tensorFile=1))
    fileDF = pd.DataFrame(rows)
    labelDF = labelDF.merge(fileDF, on=['patient_id', 'image_id'])
    missing = labelDF[labelDF.tensorFile.isna()]
    assert len(missing)==0

    #trainDF, valDF = makeSplit(labelDF)

    return makeSplit(labelDF)


class EmbeddingDataset(Dataset):
    def __init__(self, labelDF, args, noise=0, tensorDrop=0.0):
        self.tensorDrop = tensorDrop
        self.args = args
        self.noise = noise
        self.labelDF = labelDF
        self.tensorCache = dict()           # for positive tensors only

    def getTensor(self, row, item):
        # FIXME - interpolate tensorFile on the fly!
        #tensorFile = f'/fast/rsna-breast/features/{self.encoder}/{row.patient_id}/{row.image_id}.arrow'
        tensorFile = f'/fast/rsna-breast/features/{self.args.encoder}/{row.patient_id}/{row.image_id}.pt'

        T = torch.load(tensorFile)

        if self.args.normTensors:
            T = T.detach()
            T = T/T.max()           # cheap normalization

        return T


        # disable cache for now ....
        if not row.cancer:
            return torch.load(tensorFile)
            #return torch.from_numpy(readArrow(tensorFile))
        else:
            tensor = self.tensorCache.get(item, None)
            if tensor is None:
                tensor = torch.load(tensorFile)
                #tensor = torch.from_numpy(readArrow(tensorFile))
                self.tensorCache[item] = tensor
            return tensor

    def __getitem__(self, item):
        row = self.labelDF.iloc[item]
        #print(item)
        #print(row)
        #tensor = torch.load(row.tensorFile)
        tensor = self.getTensor(row, item)
        cancer = np.array([row.cancer]).astype('float32')

        if self.tensorDrop:
            keepFrac = 1.0-self.args.tensorDrop
            keep = int(keepFrac*len(tensor))
            #print(f'keeping {keep}/{tensor.shape}')
            tensorList = [t for t in tensor]
            shuffle(tensorList)
            tensor = torch.stack(tensorList[:keep])
            #print(f'new tensor : {tensor.shape}')

        if self.noise:
            # because this is in-place - cached tensors are broken slowly over time, which are all the positive cases!
            #tensor += self.noise*torch.randn(tensor.shape)
            # this should create a new one, and leave the cached tensor alone
            #tensor = tensor + self.noise * torch.randn(tensor.shape)
            tensor += self.noise * torch.randn(tensor.shape)
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
    def __init__(self, labelDF, H5):
        #self.args = args
        self.labelDF = labelDF
        self.h5 = H5
        self.ca = self.h5.root.tensors

    def __getitem__(self, item):
        row = self.labelDF.iloc[item]
        startIDX, endIDX = row.startIDX, row.startIDX+row.nTiles
        tensorBag = self.ca[startIDX:endIDX,:]
        tensorBag = tensorBag.astype(np.float32)
        tensorBag = tensorBag/255.0
        #print(tensorBag.shape)
        assert len(tensorBag)==row.nTiles
        tensorBag = torch.from_numpy(tensorBag)
        #crash
        cancer = np.array([row.cancer]).astype('float32')
        #cancer = row.cancer*1.0
        return tensorBag, cancer

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
