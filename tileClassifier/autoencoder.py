import sys

import numpy as np
import pytorch_lightning as pl
import timm
import torch
from torch import nn
from torch.nn import BCELoss, MSELoss
from torchmetrics import Accuracy
import torch.nn.functional as F
from torchvision import transforms as T


class AutoencoderModel(pl.LightningModule):
    def __init__(self, argparse):
        super().__init__()
        self.args = argparse

        #k1 = 5
        #self.colorize = nn.Conv2d(1, 3, (k1, k1), padding='same')

        f = self.args.filters
        k = self.args.kernel

        if self.args.levels == 2:
            self.build2level(f, k)
        elif self.args.levels == 3:
            self.build3level(f, k)

        if self.args.classifier:
            self.buildClassifier(f*3, 3)

        self.criterion = MSELoss()
        self.bce = BCELoss()
        #self.f1criterion = F1loss()
        self.accuracy = Accuracy()

    def build2level(self, f, k):
        # f = 64
        # k = 3
        # first stage at 224 image size
        layers = [
            nn.Conv2d(1, f, (k, k), padding='same'),
            nn.ReLU(),
            nn.Conv2d(f, f, (k, k), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        ]

        # second stage at 112 tensor size
        f2 = 2 * f
        layers += [
            nn.Conv2d(f, f2, (k, k), padding='same'),
            nn.ReLU(),
            nn.Conv2d(f2, 3, (k, k), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Tanh()
        ]
        self.encoder = nn.Sequential(*layers)
        # encoder is RGB image at 1/4 the size!


        decoderLayers = [
            nn.ConvTranspose2d(3, f2, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(f2, f, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(f, 1, (3, 3), padding='same'),
        ]
        self.decoder = nn.Sequential(*decoderLayers)



    def build3level(self, f, k):
        # f = 64
        # k = 3
        # first stage at 224 image size
        layers = [
            nn.Conv2d(1, f, (k, k), padding='same'),
            nn.ReLU(),
            nn.Conv2d(f, f, (k, k), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        ]

        # second stage at 112 tensor size
        f2 = 2 * f
        layers += [
            nn.Conv2d(f, f2, (k, k), padding='same'),
            nn.ReLU(),
            nn.Conv2d(f2, f2, (k, k), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        ]

        # third stage at 56 tensor size
        f3 = 2 * f2
        layers += [
            nn.Conv2d(f2, f3, (k, k), padding='same'),
            nn.ReLU(),
            nn.Conv2d(f3, 3, (k, k), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Tanh()
        ]

        # output is 3x28x28
        self.encoder = nn.Sequential(*layers)
        # encoder is RGB image at 1/4 the size!


        decoderLayers = [
            nn.ConvTranspose2d(3, f3, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(f3, f2, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(f2, f, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(f, 1, (3, 3), padding='same'),
        ]
        self.decoder = nn.Sequential(*decoderLayers)

    def buildClassifier(self, f, k):
        k=3
        # for level==3, input is 28x28
        layers = [
            nn.Conv2d(3, f, (k,k), stride=1),       # output is 26x26
            nn.ReLU(),
            nn.Conv2d(f, f, (k, k), stride=1),      # output is 24x24
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),                   # 12x12
            nn.Conv2d(f, f//2, (k, k), stride=1),      # output is 10x10
            nn.ReLU(),
            nn.Conv2d(f//2, f//2, (k, k), stride=1),      # output is 8x8
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 4x4
        ]
        layers += [
            #nn.MaxPool2d((2, 2)),  # now were at 1x1
            nn.Flatten(),
            nn.Linear(16*f//2, f // 2),
            nn.BatchNorm1d(f // 2),
            nn.ReLU(),
            nn.Linear(f // 2, 1),
            nn.Sigmoid()
        ]
        self.classifier = nn.Sequential(*layers)

    def wandb_log(self, **kwargs):
        for k,v in kwargs.items(): self.log(k, v)

    def forward(self, x):
        '''
        if self.args.colorize:
            x = self.colorizer(x)
            #print(x.shape)
            #print('\n\n')
        '''

        encoding = self.encoder(x)
        #print(encoding.shape)
        reproduced = self.decoder(encoding)
        if self.args.classifier:
            preds = self.classifier(encoding).squeeze(1)
            #print(encoding.shape, preds.shape)
            #sys.exit()
            return encoding, reproduced, preds
        #print(reproduced.mean())

        return encoding, reproduced, None


    def training_step(self, batch, batch_index):
        imgs, labels, rawAttn = batch#[0]
        encoding, reproduced, preds = self.forward(imgs)
        loss = self.criterion(reproduced, imgs)
        if preds is not None:
            predLoss = self.bce(preds, labels)
            self.wandb_log(train_pred_loss=predLoss)
            loss += predLoss

        self.wandb_log(train_loss=loss)
        return dict(loss=loss)

    '''    
    def training_epoch_end(self, outputs) -> None:
        acc = torch.stack([item['acc'] for item in outputs]).mean()
        loss = torch.stack([item['loss'] for item in outputs]).mean()
        self.wandb_log(train_epoch_acc=acc, train_epoch_loss=loss)
    '''

    def validation_step(self, batch, batch_index):
        imgs, labels, rawAttn = batch#[0]
        encoding, reproduced, preds = self.forward(imgs)
        loss = self.criterion(reproduced, imgs)
        if preds is not None:
            predLoss = self.bce(preds, labels)
            self.wandb_log(val_pred_loss=predLoss)
            loss += predLoss

        self.wandb_log(val_loss=loss)
        #self.logger.log_image('reproduced', [imgs[0], reproduced[0]])
        #self.logger.log_image('encoding', [encoding[0]])

        encT = [T.ToPILImage(), T.Resize(224), T.ToTensor()]
        encT = T.Compose(encT)
        #imgT = T.ToPILImage()

        img, rep, enc = imgs[0], reproduced[0], encoding[0]
        img = torch.cat([img,img,img], dim=0)
        rep = torch.cat([rep, rep, rep], dim=0)

        def norm(i):
            i = i-i.min()
            i = i/i.max()
            return i

        img = norm(img.cpu())
        rep = norm(rep.cpu())
        enc = norm(enc.cpu())

        #print(img.shape)
        enc = encT(enc)
        i=torch.stack([img, rep, enc])
        #print(i.shape)
        self.logger.log_image('progress', [i])


        return dict(loss=loss)

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
    encoder = sys.argv[1]
    print(encoder)
    model = timm.create_model(encoder, pretrained=True, num_classes=0, )

    o = model.forward_features(torch.randn(2, 3, 224, 224))
    print(f'shape at 224: {o.shape}')

    o = model.forward_features(torch.randn(2, 3, 224//2, 224//2))
    print(f'shape at {224/2}: {o.shape}')

    o = model.forward_features(torch.randn(2, 3, 224//4, 224//4))
    print(f'shape at {224/4}: {o.shape}')
