import argparse
import os
from datetime import timedelta

import numpy
import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from skimage.transform import resize
from torch import nn
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from nyuModel import ModifiedDenseNet121
import torch.nn.functional as F

from data import getData, FullDataset
import pytorch_lightning as pl
from PIL import Image

#model = ModifiedDenseNet121(num_classes=4)
#o = model(torch.randn(2, 3, 224, 224))
#print(f'output : {o.shape}')
# sys.exit()

dev = torch.device('cuda')




def collate(batch):
    imgs = [item[0] for item in batch]
    lbls = np.asarray([item[1] for item in batch])
    #rows = pd.DataFrame([item[2] for item in batch])
    rows = [item[2] for item in batch]

    imgs = torch.stack(imgs)
    lbls = torch.from_numpy(lbls).float()

    return imgs, lbls, rows

from torchvision import transforms as T
transform = T.Compose([
    #T.RandomAffine(90, shear=(-20.0, 20.0), scale=(0.5, 1.5)),
    T.RandomCrop((4096, 3000), pad_if_needed=True),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.25])
])


if __name__=='__main__':

    labelDF = pd.read_csv('/fast/rsna-breast/train.csv').sample(frac=1).reset_index()

    #labelDF = labelDF[labelDF.cancer==1]

    dataset = FullDataset(labelDF, transform=transform)
    dataloader = DataLoader(dataset, batch_size=12, collate_fn=collate, num_workers=32, persistent_workers=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", default='14')
    args = parser.parse_args()

    dev = args.gpu
    if ',' not in dev: dev = [int(dev)]


    encoder = ModifiedDenseNet121(num_classes=4)
    model = torch.nn.DataParallel(encoder)#.to(dev)

    print('Model:', type(model))
    print('Devices:', model.device_ids)
    model = model.cuda()

    #encoder.eval()

    nExpectedBatches = len(dataset) // 14

    def norm(img):
        img = img - img.min()
        # print(map.min())
        img = 255 * img / img.max()
        img = img.astype(np.uint8)
        return img


    with torch.no_grad():
        for imgs, lbls, rows in tqdm(dataloader, total=nExpectedBatches):
            maps, smallImgs = model(imgs.cuda())
            maps = maps.detach().cpu().permute(0,2,3,1).numpy()
            smallImgs = smallImgs.detach().cpu().numpy()
            #print(maps.shape, maps.min(), maps.max())

            for row, map, img in zip(rows, maps, smallImgs):
                outPath = f'/fast/rsna-breast/heatmaps/{int(row.patient_id)}/'
                if not os.path.exists(outPath):
                    os.makedirs(outPath)
                outFile = os.path.join(outPath, f'{int(row.image_id)}.png')

                print(map.min(), map.max())

                img = img[0]
                img = norm(img)
                #map = norm(map)
                #map = map - map.min()
                #map = np.clip(map, 0, 255).astype(np.uint8)

                h,w, _ = map.shape
                #print('before resize', img.shape)
                img = resize(img, (4*h,4*w), anti_aliasing=True)
                map = resize(map, (4*h,4*w), anti_aliasing=True)
                img = norm(img)

                #map[:,:,0] = norm(map[:,:,0])
                #map[:,:,1] = norm(map[:,:,1])

                #map = map+20
                map = np.clip(map*4, 0, 255).astype(np.uint8)

                #print(map.shape, img.shape)
                map = numpy.stack([map[:,:,0], map[:,:,1], img], axis=2)
                #print(map.shape)


                #print(map)
                #print(map.shape)
                pil = Image.fromarray(map.astype(np.uint8))
                pil.save(outFile)
