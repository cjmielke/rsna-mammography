import os
import sys

import cv2
import torch
from einops import rearrange
from torch.utils.data import Dataset

from config import Config



class RSNAData(Dataset):
    def __init__(self, df, img_folder, augments=None, is_test=False):
        self.df = df
        self.is_test = is_test
        self.augments = augments
        self.img_folder = img_folder

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.df['img_name'][idx])
        img = cv2.imread(img_path)
        img = cv2.resize(img, Config['IMG_SIZE'])
        if self.augments:
            img = self.augments(image=img)['image']
        img = torch.tensor(img, dtype=torch.float)
        # Rearrange the image dimensions so that channels are first in format
        # This is because Swin Transformer Model requires Channels (c) to come first
        img = rearrange(img, 'h w c -> c h w')

        if not self.is_test:
            target = self.df['cancer'][idx]
            target = torch.tensor(target, dtype=torch.float)
            return (img, target)

        return (img)

    def __len__(self):
        return len(self.df)



