"""
Combines ideas from several approaches
A 3x3 region of 224 tiles is built around a known-high-attention tile
from either the cancer or healthy set

This region is 3*224 pixels square. Random cropping selects a 2*224 region

Network uses a small number of convolutional layers to learn a color-transform
of this BW image into RGB space, and does a single maxPooling to bring it
to a 224 tile

From there, auxilliary classifier (any timm model) is used to classify
"""
import argparse
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import BCELoss
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchmetrics import Accuracy

from data import makeSplit
from torchvision import transforms as T
import pytorch_lightning as pl

from models import getModel




def pfbeta(labels, preds, beta=1):
    labels = labels.cpu()
    #preds = preds.cpu()

    preds = preds.clip(0, 1)
    y_true_count = labels.sum()

    ctp = preds[labels==1].sum()
    cfp = preds[labels==0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.0





class RegionDataset(Dataset):
    def __init__(self, df, argparse=None, validation=False):
        self.args = argparse
        self.df = df.reset_index()
        self.df = self.df.astype(int)

        if validation:
            transform = [
                #T.RandomResizedCrop(size=224 * 2),
                #T.CenterCrop(size=2*224)
                T.CenterCrop(size=224)
            ]
        else:
            transform = [
                # T.RandomAffine(90.0),
                T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
                T.RandomAffine(90, shear=(-20.0, 20.0), scale=(0.95, 1.05)),
                #T.CenterCrop(size=2*224),
                T.CenterCrop(size=224),
                #T.RandomResizedCrop(size=224*2),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                #T.RandomApply([color_jitter], p=cj_prob),
                #T.RandomGrayscale(p=random_gray_scale),
                #GaussianBlur(
                #    kernel_size=kernel_size * input_size_,
                #    prob=gaussian_blur),
            ]

        #normalize = dict(mean=[0.5], std=[0.25])
        transform += [
            T.ToTensor(),
            #T.Normalize(mean=normalize['mean'], std=normalize['std'])      # FIXME
        ]

        transform = T.Compose(transform)
        self.transform = transform


    def __len__(self):
        return len(self.df)

    def getTile(self, ptID, imgID, row, col):
        fn = f'/fast/rsna-breast/tiles/224/{int(ptID)}/{int(imgID)}_{int(row)}_{int(col)}.png'
        pil = Image.open(fn)
        return pil

    def assembleRegion(self, R):
        # lazy filesystem implementation
        R = R.astype(int)
        dst = Image.new('L', (224 * 3, 224 * 3))

        missing = 0
        def paste(r, c):
            tile = self.getTile(R.ptID, R.imgID, R.row + r, R.col + c)
            dst.paste(tile, (224 * (1 + c), 224 * (1 + r)))

        for r in [-1, 0, 1]:
            for c in [-1, 0, 1]:
                try: paste(r, c)
                except FileNotFoundError:
                    missing += 1

        #print(f'missing {missing} tiles')

        return dst

    def __getitem__(self, item):
        R = self.df.iloc[item]
        regionIMG = self.assembleRegion(R)
        regionIMG = self.transform(regionIMG)
        target = R.target.astype(np.float32)
        return regionIMG, target, R.raw






class RegionModel(pl.LightningModule):
    def __init__(self, ARGS):
        super().__init__()
        self.args = ARGS

        #k1 = 5
        #self.colorize = nn.Conv2d(1, 3, (k1, k1), padding='same')

        f = self.args.filters
        k = self.args.kernel

        # colorizer is a module that maps a 224*2 sized BW image to RGB 224

        layers = [
            nn.Conv2d(1, f, (k, k), padding='same'),
            nn.ReLU(),
            nn.Conv2d(f, 3, (k, k), padding='same'),
            #nn.ReLU(),
            #nn.MaxPool2d((2, 2)),
            nn.Sigmoid()
        ]
        self.colorizer = nn.Sequential(*layers)
        # encoder is RGB image at 1/4 the size!

        '''
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight.data)
                xavier_uniform_(m.bias.data)
        self.colorizer.apply(weights_init)
        '''

        self.classifier, embSize = getModel(ARGS.classifier, num_classes=1)

        #self.criterion = MSELoss()
        self.bce = BCELoss()
        #self.f1criterion = F1loss()
        self.accuracy = Accuracy()

    def buildDecoder(self, f, k):           # fixme - not sure I want this
        decoderLayers = [
            nn.ConvTranspose2d(3, f2, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(f2, f, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(f, 1, (3, 3), padding='same'),
        ]
        self.decoder = nn.Sequential(*decoderLayers)




    def wandb_log(self, **kwargs):
        for k,v in kwargs.items(): self.log(k, v)

    def forward(self, x):
        print(x.min(), x.mean(), x.max())
        sys.exit()
        rgb = self.colorizer(x)

        if self.args.skipcon:
            #p = torch.max_pool2d(x, 2)
            p=x
            rgbBase = torch.cat([p,p,p], dim=1)
            #print(rgbBase.shape, rgb.shape)
            #sys.exit()
            rgb = rgb + rgbBase

        #print(f'rgb shape {rgb.shape}')
        preds = self.classifier(rgb).squeeze(1)
        preds = torch.sigmoid(preds)
        return rgb, preds


    def training_step(self, batch, batch_index):
        imgs, labels, rawAttn = batch#[0]
        rgb, preds = self.forward(imgs)
        loss = self.bce(preds, labels)
        acc = self.accuracy(preds, labels.int())
        f = pfbeta(labels, preds)
        self.wandb_log(train_loss=loss, train_pfbeta=f, train_acc=acc)
        return dict(loss=loss)

    '''    
    def training_epoch_end(self, outputs) -> None:
        acc = torch.stack([item['acc'] for item in outputs]).mean()
        loss = torch.stack([item['loss'] for item in outputs]).mean()
        self.wandb_log(train_epoch_acc=acc, train_epoch_loss=loss)
    '''

    def validation_step(self, batch, batch_index):
        imgs, labels, rawAttn = batch#[0]
        rgbs, preds = self.forward(imgs)
        loss = self.bce(preds, labels)
        acc = self.accuracy(preds, labels.int())
        f = pfbeta(labels, preds)

        self.wandb_log(val_loss=loss, val_pfbeta=f, val_acc=acc)

        if batch_index <= 16:
            resizeT = T.Compose([T.ToPILImage(), T.Resize(224), T.ToTensor()])
            #imgT = T.ToPILImage()

            img, rgb = imgs[0], rgbs[0]
            img = resizeT(img)
            img = torch.cat([img, img, img])          # bw to rgb

            def norm(i):
                i = i-i.min()
                i = i/i.max()
                return i

            img, rgb = norm(img.cpu()), norm(rgb.cpu())

            i=torch.stack([img, rgb])
            self.logger.log_image('progress', [i])

        return dict(val_loss=loss)

    '''
    def validation_epoch_end(self, outputs) -> None:
        loss = torch.stack([item['loss'] for item in outputs]).mean()
        acc = torch.stack([item['acc'] for item in outputs]).mean()
        #loss = torch.mean(loss)
        #acc = self.accuracy(preds, cancer.int())
        #score = pfbeta(cancer, preds)
        #print(cancer)
        #print(preds)
        #self.log('val_accuracy', acc, batch_size=len(batch), on_epoch=True)
        self.wandb_log(val_epoch_acc=acc, val_epoch_loss=loss)
    '''


    def configure_optimizers(self):
        if self.args.opt == 'sgd':
            optim = torch.optim.SGD(self.parameters(), lr=self.args.lr)
        elif self.args.opt == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        else:
            raise ValueError
        return optim





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", default='0')
    parser.add_argument("--bs", default=8, type=int)
    parser.add_argument("--classifier", default='deit3_small_patch16_224')
    # parser.add_argument("--attnThresh", default=0.8, type=float, help='minimum attention score from MIL model to consider a tile cancerous')
    # parser.add_argument("--sigma", default=2.0, help="number of sigma raw attention score to threshold")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--opt", default='sgd')

    parser.add_argument("--sampler", default=1.0, type=float)
    # parser.add_argument("--lr", default=0.01, type=float)
    # parser.add_argument("--reducedDim", default=64, type=int)
    # parser.add_argument("--hiddenAttn", default=0, type=int)
    # parser.add_argument("--opt", default='sgd')
    parser.add_argument("-wandb", default='mamm-region-classifier')
    # parser.add_argument("-target", default='attentionWithRaw')
    # parser.add_argument("-l1norm", action='store_true')
    # parser.add_argument("--colorize", default=1, type=int)
    # parser.add_argument("--pool", default=1, type=int)
    # parser.add_argument("--poolFilt", default=8, type=int)

    parser.add_argument("--topN", default=5, type=int)
    parser.add_argument("--skipcon", default=1, type=int)

    parser.add_argument("--kernel", default=5, type=int)
    parser.add_argument("--filters", default=32, type=int)

    parser.add_argument("-notes", default=None)
    # parser.add_argument("-attnset", default=choice(['kind', 'classy', 'avg']))
    # parser.add_argument("-poolWeights", default="VGG")
    args = parser.parse_args()










    # best attentionSet we have yet
    attns = pd.read_feather('/fast/rsna-breast/tables/attn_scores_all_atomic_sweep_134.feather')

    # lots of tasks possible ....
    # tiles with raw attention score > threshold
    # top 1-5 tiles from each image - since I havent tried this yet, lets do it ...

    # attns = attns.sort_values('attention', ascending=False)
    topEachImage = attns.sort_values(['imgID', 'attention'], ascending=False).groupby('imgID').head(args.topN)
    # merge in cancer labels at the imageID level
    labels = pd.read_csv('/fast/rsna-breast/train.csv')[['image_id', 'cancer']]
    topEachImage = topEachImage.merge(labels, left_on='imgID', right_on='image_id')
    topEachImage['target'] = topEachImage.cancer

    trainDF, valDF = makeSplit(topEachImage)

    trainDataset = RegionDataset(trainDF)
    valDataset = RegionDataset(trainDF)




    class_count = list(trainDF.cancer.value_counts())
    class_count[1] = args.sampler * class_count[1]  # overweight the positives
    # class_count[1] =
    print(f'class_count : {class_count}')
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    print(f'class_weights : {class_weights}')
    lbl = list(trainDF.cancer)
    class_weights_all = class_weights[lbl]

    weighted_sampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all),
                                             replacement=True)


    trainDataloader = DataLoader(trainDataset, batch_size=args.bs, persistent_workers=True, num_workers=2, sampler=weighted_sampler)





    lbl = list(valDF.cancer)
    class_weights_all = class_weights[lbl]
    valSampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all),
                                             replacement=True)

    valDataloader = DataLoader(valDataset, batch_size=16, persistent_workers=True, num_workers=2,
                               #shuffle=True,
                               sampler=valSampler
                               )

    model = RegionModel(args)










    wandb_logger = WandbLogger(project=args.wandb, notes=args.notes)
    del args.notes
    wandb_logger.log_hyperparams(args)

    run, name = None, 'foo'


    dev = args.gpu
    if ',' not in dev: dev = [int(dev)]

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/fast/rsna-breast/checkpoints/regionClassifier/{args.classifier}_{wandb_logger.experiment.name}/",
        save_top_k=2, monitor="train_loss", train_time_interval=timedelta(minutes=10)
    )
    trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback],
                         logger=wandb_logger,
                         accelerator='gpu', devices=dev,
                         #val_check_interval=0.1,
                         #val_check_interval=1,
                         log_every_n_steps=100,
                         limit_train_batches=32*8,
                         limit_val_batches=32*4,
                         num_sanity_val_steps=0
                         )
    trainer.fit(model=model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)
    #trainer.fit(model=model, train_dataloaders=[trainDataloader])












