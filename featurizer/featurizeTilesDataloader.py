# DataLoader version of this is WAAAAY slower than my tileStatistics pipeline, which just reads with PIL
# Maybe the torch transforms are slow? So confusing


from typing import List

from PIL import Image
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms as T

# there are 54k unique patientID_imageID pairs
# Can distribute these to workers, which then perform a glob to get all tiles
# for now, lets try writing a single-GPU version

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}


class GenericCollate(nn.Module):

    def __init__(self):
        super(GenericCollate, self).__init__()

        transform = [
            T.ToTensor(),
            T.Normalize(
                mean=imagenet_normalize['mean'],
                std=imagenet_normalize['std']
            )
        ]

        transform = T.Compose(transform)

        self.transform = transform

    def forward(self, batch: List[tuple]):

        batch_size = len(batch)

        # list of transformed images
        #transforms = [self.transform(batch[i % batch_size][0]).unsqueeze_(0) for i in range(batch_size)]
        #transforms = [self.transform(batch[i][0]).unsqueeze_(0) for i in range(batch_size)]
        transforms = [self.transform(batch[i][0]).unsqueeze_(0) for i in range(batch_size)]
        # list of labels
        #labels = torch.LongTensor([item[1] for item in batch])
        # list of filenames
        fnames = [item[1] for item in batch]

        # tuple of transforms
        transformed = torch.cat(transforms, 0)

        return transformed, fnames


class TileDataset(Dataset):
    """ Very similar to the dataset class used to train lightly models """

    def __init__(self, df):
        self.df = df
        print(f'Filtered {len(self.df)} records out of {len(df)}')
        print('building file list')
        #self.files = [f'/fast/rsna-breast/tiles/224/{int(R.ptID)}/{int(R.imgID)}_{int(R.row)}_{int(R.col)}.png' for rn,R in self.df.iterrows()]
        #self.files = [f'/fast/rsna-breast/tiles/224/{int(pt)}/{int(img)}_{int(R)}_{int(C)}.png' for
        #              pt,img,R,C in self.df[['ptID', 'imgID', 'row', 'col']].to_records(index=False)]
        self.recs = self.df[['ptID', 'imgID', 'row', 'col']].to_records(index=False)
        print('done')

    def __getitem__(self, item):
        #R = self.df.iloc[item]
        #fn = f'/fast/rsna-breast/tiles/224/{int(R.ptID)}/{int(R.imgID)}_{int(R.row)}_{int(R.col)}.png'
        #fn = self.files[item]
        pt, img, R, C = self.recs[item]
        fn = f'/fast/rsna-breast/tiles/224/{int(pt)}/{int(img)}_{int(R)}_{int(C)}.png'
        img = Image.open(fn).convert('RGB')
        #target = R.ptID
        return img, fn

    def __len__(self):
        return len(self.df)


'''
dataset = TileDataset()
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    #collate_fn=collate_fn,
    shuffle=True,
    drop_last=False,
    num_workers=1,
)
'''


if __name__=='__main__':
    # load up database of all tiles and filter for the ones we want
    #df = pd.read_feather('/fast/rsna-breast/tile_224_stats.feather')
    df = pd.read_feather('/fast/rsna-breast/tile_224_stats_sorted.feather')
    df = df[df['max'] > 50]
    #df = df.sort_values(['ptID', 'imgID'])

    dataset = TileDataset(df)
    dataloader = torch.utils.data.DataLoader( dataset, batch_size=16, num_workers=8,
        collate_fn=GenericCollate(),
        shuffle=False, drop_last=False
    )

    for tensor, fnames in tqdm(dataloader):
        #print(tensor.shape)
        pass

    #for dfImg in df.groupby(['ptID', 'imgID']):         # takes a while!
    #    print(dfImg.shape)

    '''
    for patient in tqdm(df.ptID.unique()):
        patientDF = df[df.ptID==patient]
        for img in patientDF.imgID.unique():
            imgTiles = patientDF[patientDF.imgID==img]
            print(patient, img, len(imgTiles))
    '''
