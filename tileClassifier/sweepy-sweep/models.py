import sys

import numpy as np
import pytorch_lightning as pl
import timm
import torch
from torch import nn
from torch.nn import BCELoss, MSELoss
from torchmetrics import Accuracy

def getModel(encoder):
    model = timm.create_model(encoder, pretrained=True, num_classes=0)
    o = model(torch.randn(2, 3, 224, 224))
    print(f'Original shape: {o.shape}')
    assert len(o.shape)==2
    embSize = o.shape[1]
    #o = model.forward_features(torch.randn(2, 3, 224, 224))
    #print(f'Unpooled shape: {o.shape}')
    return model, embSize


class ColorizerModel(nn.Module):
    def __init__(self, k=5, n=3):
        super().__init__()
        assert n >= 1
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        layers = [nn.Conv2d(1, 3, (k, k), padding='same')]
        for x in range(n - 1):
            layers.append(nn.Conv2d(3, 3, (k, k), padding='same'))
        self.conv = nn.Sequential(*layers)

    def forward(self, bwImage):
        # model basically uses 3 convolutional filters to add residual color information on top of original image
        rgbBase = torch.cat([bwImage, bwImage, bwImage], dim=1)
        C = self.conv(bwImage)
        #return rgbBase, C
        #print(rgbBase.shape, C.shape)
        return rgbBase + C






class Model(pl.LightningModule):
    def __init__(self, argparse):
        super().__init__()
        #self.model = getModel(encoder)
        self.args = argparse

        self.colorizer = ColorizerModel(n=self.args.colorize)

        self.encoder, embSize = getModel(self.args.encoder)


        regressionLayers = [
            nn.BatchNorm1d(embSize),
            nn.ReLU(),
            nn.Linear(embSize, 1, bias=True)
        ]
        self.rawAttn = nn.Sequential(*regressionLayers)
        self.regLoss = MSELoss()

        classifierLayers = [
            #nn.BatchNorm1d(embSize),
            #nn.ReLU(),
            #nn.Linear(embSize, embSize//2),
            nn.BatchNorm1d(embSize),
            nn.ReLU(),
            nn.Linear(embSize, 1, bias=True)
        ]
        self.classifier = nn.Sequential(*classifierLayers)
        self.criterion = BCELoss()
        #self.f1criterion = F1loss()
        self.accuracy = Accuracy()

    def wandb_log(self, **kwargs):
        for k,v in kwargs.items(): self.log(k, v)

    def forward(self, x):
        #print(x.min(), x.mean(), x.max(), x.shape)
        #preds = self.model(x)
        if self.args.colorize:
            x = self.colorizer(x)
            #print(x.shape)
            #print('\n\n')


        embeddings = self.encoder(x)
        preds = self.classifier(embeddings)
        #print(preds)
        preds = torch.sigmoid(preds)
        preds = preds.squeeze(1)    # why?
        #print(preds)

        rawAttn = self.rawAttn(embeddings)      # regression of attention score

        #print('preds', preds.min(), preds.max(), preds.shape)
        return preds, rawAttn

    def training_step(self, batch, batch_index):
        imgs, labels, rawAttn = batch#[0]
        preds, rawAttnPred = self.forward(imgs)
        binLoss = self.criterion(preds, labels)
        regLoss = self.regLoss(rawAttnPred, rawAttn)
        acc = self.accuracy(preds, labels.int())

        loss = binLoss #+ regLoss

        self.wandb_log(train_loss=binLoss, train_acc=acc)
        return dict(loss=loss, acc=acc)

    def training_epoch_end(self, outputs) -> None:
        acc = torch.stack([item['acc'] for item in outputs]).mean()
        loss = torch.stack([item['loss'] for item in outputs]).mean()
        self.wandb_log(train_epoch_acc=acc, train_epoch_loss=loss)

    def validation_step(self, batch, batch_index):
        imgs, labels, rawAttn = batch#[0]
        preds, rawAttnPred = self.forward(imgs)
        binLoss = self.criterion(preds, labels)
        regLoss = self.regLoss(rawAttnPred, rawAttn)
        acc = self.accuracy(preds, labels.int())

        loss = binLoss #+ regLoss

        self.wandb_log(val_loss=loss, val_acc=acc)
        return dict(loss=loss, acc=acc)

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
