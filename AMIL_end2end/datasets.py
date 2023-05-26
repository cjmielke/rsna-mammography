import sys
from glob import glob
from typing import List

import numpy as np
import pandas as pd
import tables
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import Dataset
import pyarrow as pa
import torchvision.transforms as T



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



class TensorBagDataset(Dataset):

    def __init__(self, imagesDF, tilesDF, training=True, limit=32):
        self.limit = limit
        self.training = training
        self.imagesDF = imagesDF
        self.tilesDF = tilesDF
        self.imageTilesDFs = {}
        for imgID, imgTilesDF in tilesDF.groupby('imgID'):
            self.imageTilesDFs[imgID] = imgTilesDF

    def __len__(self):
        return len(self.imagesDF)

    def __getitem__(self, item):
        imgRow = self.imagesDF.iloc[item]
        imgTilesDF = self.imageTilesDFs[imgRow.image_id]
        # the return should be a stack of tiles (possibly limited) and a target
        if self.training:
            # test at most 32 tiles. 16 selected randomly from the 32 tiles with highest attention score
            # 16 other tiles selected at random from the whole bag
            #topAttnTiles = imgTilesDF.sort_values('attention', ascending=False).head(self.limit*2).sample(frac=0.5)
            topAttnTiles = imgTilesDF.sort_values('attention', ascending=False).head(self.limit)
            #print(imgTilesDF.shape)
            if len(imgTilesDF) > self.limit:
                everythingElse = imgTilesDF.sample(frac=1).head(self.limit//2)
                #DF = imgTilesDF.sample(frac=1).head(limit)
                DF = pd.concat([topAttnTiles.head(self.limit//2), everythingElse]).drop_duplicates()
            else:
                DF = topAttnTiles#.sample(num=self.limit)
        else:           # Validation set
            # test limited number
            #DF = imgTilesDF.head(128)               # test all validation tiles, to a reasonable limit
            # actually, bad idea. Some validation images have hundreds of tiles. Better prioritize high-attention
            DF = imgTilesDF.sort_values('attention', ascending=False).head(32)

        #print(DF)

        images = []
        for idx, row in DF.iterrows():
            R = row.astype(int)
            fn = f'/fast/rsna-breast/tiles/224/{R.ptID}/{R.imgID}_{R.row}_{R.col}.png'
            img = Image.open(fn)#.convert('RGB')
            images.append(img)

        return images, imgRow.cancer


def getDatasets(args):
    imagesDF = pd.read_csv('/fast/rsna-breast/train.csv')
    trainImages, valImages = makeSplit(imagesDF)

    #tilesDF = pd.read_feather('/fast/rsna-breast/tables/tile_224_stats_sorted.feather')
    #tilesDF = pd.read_feather('/fast/rsna-breast/tables/attn_scores_all_giddy_rain.feather')
    #tilesDF = pd.read_feather('/fast/rsna-breast/tables/attn_scores_all_kind_sweep_53.feather')

    # tiles re-generated
    #tilesDF = pd.read_feather('/fast/rsna-breast/tables/attn_scores_sweet_laughter_413.feather')
    tilesDF = pd.read_feather('/fast/rsna-breast/tables/attn_scores_xcit_glad_shape_220.feather')


    trainTiles = tilesDF[tilesDF.imgID.isin(trainImages.image_id)]
    valTiles = tilesDF[tilesDF.imgID.isin(valImages.image_id)]

    trainDataset = TensorBagDataset(trainImages, trainTiles, training=True, limit=args.tileLimit)
    valDataset = TensorBagDataset(valImages, valTiles, training=False)

    return trainDataset, valDataset




imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

normalize = { 'mean': [0.5], 'std': [0.25] }

class GenericCollate(nn.Module):

    def __init__(self, training=True):
        super(GenericCollate, self).__init__()

        if training:
            transform = [
                # T.RandomAffine(90.0),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                T.RandomAffine(90, shear=(-20.0, 20.0), scale=(0.75, 1.25)),
                # T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                # T.RandomApply([color_jitter], p=cj_prob),
                # T.RandomGrayscale(p=random_gray_scale),
                # GaussianBlur(
                #    kernel_size=kernel_size * input_size_,
                #    prob=gaussian_blur),
            ]
        else:
            transform = []

        transform += [
            T.ToTensor(),
            #T.Normalize(mean=imagenet_normalize['mean'], std=imagenet_normalize['std']),
            T.Normalize(mean=normalize['mean'], std=normalize['std'])
        ]

        transform = T.Compose(transform)

        self.transform = transform

    def forward(self, batch: List[tuple]):
        batch_size = len(batch)

        # list of transformed images
        #transforms = [self.transform(batch[i][0]).unsqueeze_(0) for i in range(batch_size)]
        imgBags = []
        for item in batch:
            pils = item[0]
            transforms = [self.transform(pil).unsqueeze_(0) for pil in pils]
            transformed = torch.cat(transforms, 0)
            imgBags.append(transformed)

        # list of labels
        labels = torch.LongTensor([item[1] for item in batch])

        # tuple of transforms
        #transformed = torch.cat(transforms, 0)

        #print(transformed.shape)

        #return transformed, labels

        return imgBags, labels

'''
def collate(batch):
    images = [item[0] for item in batch]
    lbls = [item[1] for item in batch]
    print(images[0])
    #print(embs[0])     # tensor already, presumably from torch.load

    #embeddings = torch.cat(embs, 0)
    #embeddings = torch.stack(embs)                  # cant do this, ragged!
    #labels = torch.cat(lbls, 0)
    labels = torch.from_numpy(np.concatenate(lbls))
    return images, labels

'''