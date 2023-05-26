import sys
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms as T
import pandas as pd






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



imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}



class MammDataset(Dataset):
    def __init__(self, df, args, validation=False):
        self.args = args
        self.df = df

        if validation:
            transform = [
                #T.Resize(512)
                #T.RandomCrop((1024, 832), pad_if_needed=True, fill=0, padding_mode='constant')
            ]
        else:
            transform = [
                #T.RandomAffine(90.0),
                #T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                T.RandomAffine(20, shear=(-20.0, 20.0), scale=(0.85, 1.15)),
                #T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                #T.RandomCrop((768//2,1024//2), pad_if_needed=True, fill=0, padding_mode='constant')
                #T.Resize(512),
                #T.RandomCrop((1024, 832), pad_if_needed=True, fill=0, padding_mode='constant')
                #T.RandomApply([color_jitter], p=cj_prob),
                #T.RandomGrayscale(p=random_gray_scale),
                #GaussianBlur(
                #    kernel_size=kernel_size * input_size_,
                #    prob=gaussian_blur),
                #T.ToTensor()
            ]

        if self.args.colorizer:
            normalize = { 'mean': [0.5], 'std': [0.25] }
        else:
            normalize = imagenet_normalize
        transform += [
            T.ToTensor(),
            T.Normalize( mean=normalize['mean'], std=normalize['std'] )
        ]

        transform = T.Compose(transform)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        pngFile = f'/fast/rsna-breast/{self.args.dataset}/{int(row.patient_id)}/{int(row.image_id)}.png'
        pil = Image.open(pngFile)
        if pil.mode == 'L' and not self.args.colorizer:
            pil = pil.convert('RGB')
        tensor = self.transform(pil)
        return tensor, row.cancer

def collate(batch):
    imgs = torch.stack([i[0] for i in batch])
    labels = torch.from_numpy(np.asarray([i[1] for i in batch])).float()
    return imgs, labels

def getDataloaders(args):
    labelsDF = pd.read_csv('/fast/rsna-breast/train.csv')#[['image_id', 'cancer']]
    trainDF, valDF = makeSplit(labelsDF)

    trainData = MammDataset(trainDF, args)
    #valData = MammDataset(trainDF, args, validation=True)           # FUUUUUUUCK
    valData = MammDataset(valDF, args, validation=True)

    class_count = list(trainDF.cancer.value_counts())
    class_count[1] *= args.sampler         # overweight
    print(f'class_count : {class_count}')
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    print(f'class_weights : {class_weights}')
    lbl = list(trainDF.cancer)
    class_weights_all = class_weights[lbl]

    weighted_sampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all),
                                             replacement=True)

    trainLoader = DataLoader(trainData, batch_size=args.bs, num_workers=4, collate_fn=collate, sampler=weighted_sampler, persistent_workers=True)
    valLoader = DataLoader(valData, batch_size=8, num_workers=4, collate_fn=collate, persistent_workers=True, shuffle=True)




    return trainLoader, valLoader


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default='png8')
    args = parser.parse_args()

    trainL, valL = getDataloaders(args)
    for batch in trainL:
        imgs, lbls = batch
        print(imgs.shape)
        #print(batch)

