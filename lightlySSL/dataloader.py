import os.path
import random
import time
from glob import glob

import PIL.Image
import pandas as pd
import torch
from lightly.data import LightlyDataset
from torch.utils.data import WeightedRandomSampler


def filterTileDF(df):
    new = []
    missingTiles = 0
    tiles = set(glob('/fast/rsna-breast/tiles/224/*/*_*_*.png'))
    print(f'Fouund {len(tiles)} tiles with glob')
    #df['combined'] = df['bar'].astype(str) + '_' + df['foo'] + '_' + df['new']
    #df['filename'] =
    for rn, R in df.iterrows():
        tile = f'/fast/rsna-breast/tiles/224/{int(R.ptID)}/{int(R.imgID)}_{int(R.row)}_{int(R.col)}.png'
        #if  os.path.exists(tile):
        if tile in tiles:
            new.append(R)
        else:
            missingTiles += 1

    print(f'Missing tiles : {missingTiles}')            # 359,164 !!!!!
    return pd.DataFrame(new)

class TileLightlyDataset(LightlyDataset):

    def __init__(self, RGB=True):
        start = time.time()
        self.RGB = RGB
        df = pd.read_feather('/fast/rsna-breast/tables/tile_224_stats_sorted.feather')
        labels = pd.read_csv('/fast/rsna-breast/train.csv')
        took = time.time() - start
        print(f'took {took} seconds to load dataloader')
        #for col in ['ptID', 'imgID', 'row', 'col']:
        #    df[col] = df[col].astype('int')
        self.df = df[df['max']>20]
        self.df = self.df.merge(labels, how='inner', left_on=['ptID', 'imgID'], right_on=['patient_id', 'image_id'])

        # weigh equally tiles from cancer mammograms to tiles from noncamcer
        '''
        class_count = list(self.df.cancer.value_counts())
        #class_count[1] =
        print(f'class_count : {class_count}')
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        print(f'class_weights : {class_weights}')
        lbl = self.df.cancer
        class_weights_all = class_weights[lbl]

        weighted_sampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all),
                                                 replacement=True)
        '''

        # weigh by high-attention tiles .... could also use a threshold.... or could maybe square the normalized attention
        # this attentionset trained sweepy_sweep ..... which performed amazingly on MIL
        #attenDF = pd.read_feather('/fast/rsna-breast/tables/attn_scores_all_kind_sweep_53.feather')#.set_index(['imgID', 'row', 'col'])
        #attenDF = pd.read_feather('/fast/rsna-breast/tables/attn_scores_all_classic_sweep_9.feather')#.set_index(['imgID', 'row', 'col'])

        # new dataset was made here
        #attenDF = pd.read_feather('/fast/rsna-breast/tables/attn_scores_sweet_laughter_413.feather')
        attenDF = pd.read_feather('/fast/rsna-breast/tables/attn_scores_xcit_glad_shape_220.feather')

        #attenDF = filterTileDF(attenDF)
        labels = labels[['image_id', 'cancer']]
        attenDF = attenDF.merge(labels, left_on='imgID', right_on='image_id')
        self.df = attenDF

        # simply weight by raw attention
        class_weights_all = self.df.raw
        class_weights_all = class_weights_all - class_weights_all.min()

        '''
        # stratify sampling on tiles from cancer
        class_count = list(self.df.cancer.value_counts())
        #class_count[1] =
        print(f'class_count : {class_count}')
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        print(f'class_weights : {class_weights}')
        lbl = list(self.df.cancer)
        class_weights_all = class_weights[lbl]
        '''

        weighted_sampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all),
                                                 replacement=True)
        self.sampler = weighted_sampler

        #self.df = df
        #T = {k:int for k in ['ptID', 'imgID', 'row', 'col']}
        #self.df = self.df.astype(T)
        print(f'Filtered {len(self.df)} records out of {len(df)}')

    def __getitem__(self, item):
        R = self.df.iloc[item]
        fn = f'/fast/rsna-breast/tiles/224/{int(R.ptID)}/{int(R.imgID)}_{int(R.row)}_{int(R.col)}.png'
        try: img = PIL.Image.open(fn)
        except:
            print('file missing, substituting')
            return self.__getitem__(random.randint(0, len(self)))
        if self.RGB:
            img = img.convert('RGB')
        #target = R.ptID
        target = R.cancer
        #target = 0
        return img, target, fn

    def __len__(self):
        return len(self.df)



if __name__=='__main__':
    ds = TileLightlyDataset()
    print(ds.df.head().T)
