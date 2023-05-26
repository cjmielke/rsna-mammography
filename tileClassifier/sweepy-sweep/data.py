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



'''
        class_count = list(trainDF.cancer.value_counts())
        class_count[1] = args.sampler * class_count[1]      # overweight the positive cases. If sampler=1.0, then its trained 50/50
        print(class_count)
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        print(class_weights)
        class_weights_all = class_weights[labels]
        print(class_weights_all)
        sampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all), replacement=True)

'''


def getTablesV1(args):
    # has 224234 tiles
    attenDF = pd.read_csv('/fast/rsna-breast/tables/attn_scores_giddy_rain.csv')#.set_index(['imgID', 'row', 'col'])

    # has 114131 tiles - since less images were analyzed ....
    # FIXME - need to experiment with this in a manner most appropriate! Some patients are labelled "cancer" and "nocancer" accross different images!
    #attenDF = pd.read_csv('/fast/rsna-breast/tables/attn_scores_img_giddy_rain.csv')#.set_index(['imgID', 'row', 'col'])

    # FIXME - experiment 3 .... could score the entire tiles dataset (not just from positive cases) and weigh accordingly ....

    tilesDF = pd.read_feather('/fast/rsna-breast/tables/tile_224_stats_sorted.feather')
    tilesDF = tilesDF[tilesDF['max'] > 50]

    labelsDF = pd.read_csv('/fast/rsna-breast/train.csv')
    healthyPatients = set(labelsDF[labelsDF.cancer == 0].patient_id)
    cancerPatients = set(labelsDF[labelsDF.cancer == 1].patient_id)

    healthyPatientTiles = tilesDF[tilesDF.ptID.isin(healthyPatients)]
    cancerPatientTiles = tilesDF[tilesDF.ptID.isin(cancerPatients)]     # dont need this really, unless I need min/max/mean/std columns

    #healthyPatientTiles =

    #cancerPatientTiles = cancerPatientTiles.set_index(['ptID','imgID','row','col'])

    print(healthyPatientTiles.shape, cancerPatientTiles.shape)
    print(attenDF.shape)

    # for simplicity, since I dont need those other columns, no need to join
    cancerPatientTiles = attenDF

    # setting weights .... lots to think about! For a first pass, a 50/50 split seems reasonable
    # HOWEVER, we want the high-scoring tiles from the cancer set to have the highest probability

    healthyPatientTiles['attention'] = 1
    # cancerPatientTiles have actual attention values that aren't uniform,
    # but should be shifted. IE, tiles with attention=0 should have same weight as non-cancer tiles!
    cancerPatientTiles['attention'] += 1
    #cancerPatientTiles['attention'] = 1         # FIXME - just for testing....

    # to get weight, we'll scale by preponderance
    numHealthyTiles = len(healthyPatientTiles)
    numCancerPatientTiles = len(cancerPatientTiles)

    healthyPatientTiles['weight'] = healthyPatientTiles['attention'] / numHealthyTiles
    cancerPatientTiles['weight'] = cancerPatientTiles['attention'] / numCancerPatientTiles

    healthyPatientTiles['cancer'] = 0       # totally true
    cancerPatientTiles['cancer'] = 1        # FIXME - this is wrong. Many patients have some images labelled cancer, and other images labelled not

    mergedDF = pd.concat([healthyPatientTiles, cancerPatientTiles])#.sample(frac=1, random_state=13)
    mergedDF['weight'] = mergedDF['weight'] * numHealthyTiles
    print(mergedDF)
    print(mergedDF[['cancer','weight']].value_counts())
    print(mergedDF['weight'].sort_values())
    print(mergedDF['weight'].astype(int).value_counts())

    mergedDF['target'] = mergedDF.cancer
    mergedDF = mergedDF.sample(frac=1)          # give it a shuffle (shouldn't matter though)

    for c in ['min', 'max', 'mean', 'std']: del mergedDF[c]

    trainDF, valDF = makeSplit(mergedDF)

    return trainDF, valDF



# builds a dataset of noncancer/cancer tiles, where the cancer tiles are those with attention > threshold
# task performed well (75% training and validation accuracy predicting cancer), but these features performed worse
# in the MIL model than base efficientnet. I suspect it was too easy a task?
def getTablesV2(args):
    # has 224234 tiles
    attenDF = pd.read_csv('/fast/rsna-breast/tables/attn_scores_giddy_rain.csv')#.set_index(['imgID', 'row', 'col'])

    # FIXME - need to experiment with this in a manner most appropriate! Some patients are labelled "cancer" and "nocancer" accross different images!
    # has 114131 tiles - since less images were analyzed ....
    #attenDF = pd.read_csv('/fast/rsna-breast/tables/attn_scores_img_giddy_rain.csv')#.set_index(['imgID', 'row', 'col'])

    # FIXME - experiment 3 .... could score the entire tiles dataset (not just from positive cases) and weigh accordingly ....

    #attenDF['cancer'] = 0.0                                 # low-attention tiles probably dont have cancer in them!
    #attenDF[attenDF.attention >= args.attnThresh]['cancer'] = 1.0     # can tune this threshold via hyperparameter tuning!

    attenDF = attenDF[attenDF.attention >= args.attnThresh]     # only grab subset
    # we could include the low-attention tiles in this set along with the non-cancer patients, but
    # 1) we already have so many negative tiles
    # 2) can't risk mis-labelling a cancer tile
    attenDF['cancer'] = 1

    tilesDF = pd.read_feather('/fast/rsna-breast/tables/tile_224_stats_sorted.feather')
    tilesDF = tilesDF[tilesDF['max'] > 50]

    labelsDF = pd.read_csv('/fast/rsna-breast/train.csv')
    healthyPatients = set(labelsDF[labelsDF.cancer == 0].patient_id)
    cancerPatients = set(labelsDF[labelsDF.cancer == 1].patient_id)

    healthyPatientTiles = tilesDF[tilesDF.ptID.isin(healthyPatients)]
    cancerPatientTiles = tilesDF[tilesDF.ptID.isin(cancerPatients)]     # dont need this really, unless I need min/max/mean/std columns

    cancerPatientTiles = attenDF                    # for simplicity, since I dont need those other columns, no need to join

    # setting weights .... lots to think about! For a first pass, a 50/50 split seems reasonable
    # HOWEVER, we want the high-scoring tiles from the cancer set to have the highest probability

    healthyPatientTiles['attention'] = 1
    # cancerPatientTiles have actual attention values that aren't uniform,
    # but should be shifted. IE, tiles with attention=0 should have same weight as non-cancer tiles!
    cancerPatientTiles['attention'] += 1
    #cancerPatientTiles['attention'] = 1         # FIXME - just for testing....

    # to get weight, we'll scale by preponderance
    numHealthyTiles = len(healthyPatientTiles)
    numCancerPatientTiles = len(cancerPatientTiles)

    healthyPatientTiles['weight'] = healthyPatientTiles['attention'] / numHealthyTiles
    cancerPatientTiles['weight'] = cancerPatientTiles['attention'] / numCancerPatientTiles

    healthyPatientTiles['cancer'] = 0       # totally true
    #cancerPatientTiles['cancer'] = 1        # FIXME - this is wrong. Many patients have some images labelled cancer, and other images labelled not

    mergedDF = pd.concat([healthyPatientTiles, cancerPatientTiles])#.sample(frac=1, random_state=13)
    mergedDF['weight'] = mergedDF['weight'] * numHealthyTiles
    print(mergedDF)
    print(mergedDF[['cancer','weight']].value_counts())
    print(mergedDF['weight'].sort_values())
    print(mergedDF['weight'].astype(int).value_counts())

    mergedDF = mergedDF.sample(frac=1)          # give it a shuffle (shouldn't matter though)

    for c in ['min', 'max', 'mean', 'std']: del mergedDF[c]

    mergedDF['target'] = mergedDF.cancer
    trainDF, valDF = makeSplit(mergedDF)

    print(f'Negative tiles overall : {len(mergedDF[mergedDF.cancer==0])}')
    print(f'Positive tiles overall : {len(mergedDF[mergedDF.cancer==1])}')

    return trainDF, valDF



# task 3 - instead of predicting cancer tiles, try predicting high-attention tiles in both cancer and noncancer images
def getTablesAttn(args):  #FIXME - unfinished
    # has 5,251,862 tiles !
    attenDF = pd.read_csv('/fast/rsna-breast/tables/attn_scores_all_giddy_rain.csv')#.set_index(['imgID', 'row', 'col'])

    # thresh=0.05  197171 tiles
    # thresh=0.1  111,988 tiles
    # thresh=0.5  18,195 tiles

    attenDF['target'] = 1*(attenDF.attention >= args.attnThresh)
    labels = attenDF.target

    class_count = list(attenDF.target.value_counts())

    print(class_count)
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    print(class_weights)
    class_weights_all = class_weights[labels]

    attenDF['weight'] = class_weights_all

    trainDF, valDF = makeSplit(attenDF)

    print(f'Negative tiles overall : {len(attenDF[attenDF.target==0])}')
    print(f'Positive tiles overall : {len(attenDF[attenDF.target==1])}')

    return trainDF, valDF


def getNewAttnsWithRaws(attnThresh=None, rawSigma=None):
    # has 5,251,862 tiles !
    attenDF = pd.read_feather('/fast/rsna-breast/tables/attn_scores_all_kind_sweep_53.feather')#.set_index(['imgID', 'row', 'col'])
    labels = pd.read_csv('/fast/rsna-breast/train.csv')[['image_id', 'cancer']]
    attenDF = attenDF.merge(labels, left_on='imgID', right_on='image_id')

    if rawSigma:
        thresh = (attenDF.raw.mean() + float(rawSigma) * float(attenDF.raw.std()))
        print(f'Dataset size before filtering raw attention by {rawSigma} sigma (thresh : {thresh}) : {attenDF.shape}')
        attenDF['thresh'] = attenDF.raw >= thresh
        attenDF = attenDF[attenDF.raw >= thresh]
        print(thresh)
        print(f'Dataset size after : {attenDF.shape}')
        print(attenDF[['thresh', 'cancer']].value_counts())

    ''' for Sigma==2.0
    thresh  cancer
    True    0         198524
            1           5555
    '''

    #attenDF['target'] = 1*(attenDF.attention >= attnThresh)
    #attenDF['target'] = 1*(attenDF.raw >= 0)
    attenDF['target'] = attenDF.cancer

    labels = list(attenDF.target)
    class_count = list(attenDF.target.value_counts())
    print(class_count)
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    print(class_weights)
    class_weights_all = class_weights[labels]
    attenDF['weight'] = class_weights_all

    trainDF, valDF = makeSplit(attenDF)

    print(f'Negative tiles overall : {len(attenDF[attenDF.target==0])}')
    print(f'Positive tiles overall : {len(attenDF[attenDF.target==1])}')

    print(valDF)

    return trainDF, valDF



class TileDataset(Dataset):
    def __init__(self, df, argparse):
        self.args = argparse
        self.df = df.reset_index()
        #print(self.df)
        #print(df.columns.to_series()[np.isnan(df).any()])          # finding a nan column ....
        self.df = self.df.astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        R = self.df.iloc[item]
        imgFile = f'/fast/rsna-breast/tiles/224/{R.ptID}/{R.imgID}_{R.row}_{R.col}.png'
        img = Image.open(imgFile)
        if not self.args.colorize:
            img = img.convert('RGB')
        #target = R.cancer
        target = R.target
        return img, target, R.raw





















imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

class CollateFunction(nn.Module):

    def __init__(self, argparse):
        super().__init__()
        self.args = argparse
        input_size = 224
        min_scale = 0.8

        if argparse.colorize:
            normalize = dict(mean=[0.5], std=[0.25])
        else:
            normalize = imagenet_normalize

        transform = [
            #T.RandomAffine(90.0),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            T.RandomAffine(90, shear=(-20.0, 20.0), scale=(0.5, 1.5)),
            #T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            #T.RandomApply([color_jitter], p=cj_prob),
            #T.RandomGrayscale(p=random_gray_scale),
            #GaussianBlur(
            #    kernel_size=kernel_size * input_size_,
            #    prob=gaussian_blur),
            T.ToTensor()
        ]

        if normalize:
            transform += [
             T.Normalize(
                mean=normalize['mean'],
                std=normalize['std'])
             ]

        transform = T.Compose(transform)
        self.transform = transform



    def forward(self, batch: List[tuple]):

        transforms = [self.transform(item[0]) for item in batch]
        transforms = torch.stack(transforms)
        #labels = torch.LongTensor([item[1] for item in batch])
        labels = np.asarray([item[1] for item in batch]).astype(np.float32)
        labels = torch.from_numpy(labels)


        rawAttn = np.asarray([item[2] for item in batch]).astype(np.float32)
        rawAttn = torch.from_numpy(rawAttn)
        #labels = torch.cat([item[1] for item in batch]).float()
        #print(labels)
        #print(transforms.shape, labels.shape)
        #c

        #print(labels)
        return transforms, labels, rawAttn







if __name__=='__main__':
    getNewAttnsWithRaws(rawSigma=2.0)



