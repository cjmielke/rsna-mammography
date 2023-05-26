import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T
from PIL import Image

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


def getData(sampler=8):
    labelDF = pd.read_csv('/fast/rsna-breast/train.csv').sample(frac=1).reset_index()
    trainDF, valDF = makeSplit(labelDF)

    #class_count = list(trainDF.cancer.value_counts())
    class_count = list(trainDF.biopsy.value_counts())
    class_count[1] = sampler * class_count[1]
    print(class_count)
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    print(class_weights)
    labels = list(trainDF.cancer)
    class_weights_all = class_weights[labels]
    print(class_weights_all)
    trainSampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all), replacement=True)

    return trainDF, trainSampler, valDF


transform = T.Compose([
    T.RandomAffine(90, shear=(-20.0, 20.0), scale=(0.5, 1.5)),
    T.RandomCrop((4096, 3000), pad_if_needed=True),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.25])
])


class FullDataset(Dataset):
    def __init__(self, df, transform=transform):
        self.transform = transform
        self.df = df
        #self.df = pd.read_csv('/fast/rsna-breast/train.csv').sample(frac=1).reset_index()

    def __len__(self):
        return len(self.df)

    def crop(self, pil, f=2):
        img = np.asarray(pil)

        av = np.mean(img, axis=0)
        mi = np.min(img, axis=0)
        ma = np.max(img, axis=0)
        img = img[:, (((av - mi) > f) + ((av - ma) > f))]

        av = np.mean(img, axis=1)
        mi = np.min(img, axis=1)
        ma = np.max(img, axis=1)
        img = img[(((av - mi) > f) + ((av - ma) > f)), :]

        return Image.fromarray(img)

    def __getitem__(self, item):
        R = self.df.iloc[item]
        fn = f'/fast/rsna-breast/pngfull/{int(R.patient_id)}/{int(R.image_id)}.png'
        pil = Image.open(fn)
        pil = self.crop(pil)
        if self.transform:
            tensor = self.transform(pil)
        else: tensor = pil                      # wrong
        return tensor, R.cancer, R


