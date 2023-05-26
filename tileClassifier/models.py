import sys

import numpy as np
import pytorch_lightning as pl
import timm
import torch
from torch import nn
from torch.nn import BCELoss, MSELoss
from torchmetrics import Accuracy
import torch.nn.functional as F

def getModel(encoder, num_classes=0):
    model = timm.create_model(encoder, pretrained=True, num_classes=num_classes)
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
        assert n>=1
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


class PoolingColorizer(nn.Module):
    def __init__(self, k=3, f=8):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        # first stage at 224 image size
        layers = [
            nn.Conv2d(1, f, (k, k), padding='same'),
            nn.ReLU(),
            nn.Conv2d(f, f, (k, k), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        ]

        # second stage at 112 tensor size
        f2 = 2*f
        layers += [
            nn.Conv2d(f, f2, (k, k), padding='same'),
            nn.ReLU(),
            nn.Conv2d(f2, f2, (k, k), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        ]

        # third stage, 56x56
        # finally we map channels to 3 colors with depthwise kernels ...
        # tanh activation here might make things easier to normalize layer?
        layers += [
            nn.Conv2d(f2, 3, (1, 1), padding='same'),
            #nn.Tanh()
        ]

        #for x in range(n - 1):
        self.conv = nn.Sequential(*layers)

    def forward(self, bwImage):
        # model basically uses 3 convolutional filters to add residual color information on top of original image
        downSized = F.max_pool2d(bwImage, 4)
        rgbBase = torch.cat([downSized, downSized, downSized], dim=1)
        C = self.conv(bwImage)
        out = C + rgbBase
        out = torch.tanh(out)
        return out



class VGGPooler(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        f = 64
        # first stage at 224 image size
        layers = [
            nn.Conv2d(3, f, (k, k), padding='same'),
            nn.ReLU(),
            nn.Conv2d(f, f, (k, k), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        ]

        # second stage at 112 tensor size
        f2 = 2*f
        layers += [
            nn.Conv2d(f, f2, (k, k), padding='same'),
            nn.ReLU(),
            nn.Conv2d(f2, f2, (k, k), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        ]
        self.features = nn.Sequential(*layers)

        self.loadWeights()

    def loadWeights(self):
        vgg16 = timm.create_model('vgg16', pretrained=True, num_classes=0)
        weights = vgg16.state_dict()
        missing_keys, unexpected_keys = self.load_state_dict(weights, strict=False)
        assert len(missing_keys) == 0

    def forward(self, rgbImg):
        f = self.features(rgbImg)
        return f




class PoolingColorizerVGG(nn.Module):
    def __init__(self, k1=5, k2=1):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)


        self.colorize = nn.Conv2d(1, 3, (k1, k1), padding='same')

        self.features = VGGPooler()

        # third stage, 56x56
        # finally we map channels to 3 colors with depthwise kernels ...
        # tanh activation here might make things easier to normalize layer?
        rgbLayers = [
            nn.Conv2d(128, 3, (k2, k2), padding='same'),
            #nn.Tanh()
        ]
        self.rgb = nn.Sequential(*rgbLayers)

    def forward(self, bwImage):
        # model basically uses 3 convolutional filters to add residual color information on top of original image
        downSized = F.max_pool2d(bwImage, 4)
        rgbBase = torch.cat([downSized, downSized, downSized], dim=1)


        colored = self.colorize(bwImage)
        rgb = torch.cat([bwImage, bwImage, bwImage], dim=1)
        x = colored+rgb         # skip connection
        #print(x.shape)

        #print(f'feats {f.shape}')
        f = self.features(x)
        C = self.rgb(f)
        #return rgbBase, C
        #print(rgbBase.shape, C.shape)
        out = C + rgbBase
        out = torch.tanh(out)
        return out



class Model(pl.LightningModule):
    def __init__(self, argparse):
        super().__init__()
        #self.model = getModel(encoder)
        self.args = argparse

        if self.args.colorize:
            if self.args.pool:
                #self.colorizer = PoolingColorizer(k=self.args.kernel, f=self.args.poolFilt)
                self.colorizer = PoolingColorizerVGG(k1=self.args.kernel)
            else:
                self.colorizer = ColorizerModel(n=self.args.colorize)

        self.encoder, embSize = getModel(self.args.encoder)


        self.density = nn.Sequential(
            nn.BatchNorm1d(embSize),
            nn.ReLU(),
            nn.Linear(embSize, 5, bias=True)
        )
        self.densityLoss = nn.CrossEntropyLoss(ignore_index=0)


        ageLayers = [
            nn.BatchNorm1d(embSize),
            nn.ReLU(),
            nn.Linear(embSize, 10, bias=True)
        ]
        self.age = nn.Sequential(*ageLayers)
        self.ageLoss = nn.CrossEntropyLoss()


        regressionLayers = [
            nn.BatchNorm1d(embSize),
            nn.ReLU(),
            nn.Linear(embSize, 1, bias=True)
        ]
        self.rawAttn = nn.Sequential(*regressionLayers)
        self.regLoss = MSELoss()

        self.biopsy = nn.Sequential(
            nn.BatchNorm1d(embSize),
            nn.ReLU(),
            nn.Linear(embSize, 1, bias=True)
        )
        self.biopsyLoss = BCELoss()

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

        embeddings = self.encoder(x)

        age = self.age(embeddings)
        density = self.density(embeddings)
        biopsy = self.biopsy(embeddings).squeeze(1)
        biopsy = torch.sigmoid(biopsy)

        preds = self.classifier(embeddings)
        #print(preds)
        preds = torch.sigmoid(preds)
        preds = preds.squeeze(1)    # why?
        #print(preds)

        rawAttn = self.rawAttn(embeddings)      # regression of attention score

        #print('preds', preds.min(), preds.max(), preds.shape)
        return preds, rawAttn, age, density, biopsy

    def training_step(self, batch, batch_index):
        imgs, labels, rawAttn, age, birads, density, biopsy = batch#[0]
        preds, rawAttnPred, agePred, densityPred, biopsyPred = self.forward(imgs)
        binLoss = self.criterion(preds, labels)
        biopsyLoss = self.biopsyLoss(biopsyPred, biopsy)
        rawLoss = self.regLoss(rawAttnPred, rawAttn)

        ageLoss = self.ageLoss(agePred, age.long())/2
        densityLoss = self.densityLoss(densityPred, density.long())
        acc = self.accuracy(preds, labels.int())

        loss = binLoss #+ rawLoss + ageLoss + densityLoss + biopsyLoss

        self.wandb_log(
            train_loss=loss, train_acc=acc,
            #train_cancerL=binLoss,
            #rawLoss=rawLoss, train_ageLoss=ageLoss,
            #train_densityL=densityLoss, train_biopsyL=biopsyLoss
        )
        return dict(loss=loss, acc=acc)

    def training_epoch_end(self, outputs) -> None:
        acc = torch.stack([item['acc'] for item in outputs]).mean()
        loss = torch.stack([item['loss'] for item in outputs]).mean()
        self.wandb_log(train_epoch_acc=acc, train_epoch_loss=loss)

    def validation_step(self, batch, batch_index):
        imgs, labels, rawAttn, age, birads, density, biopsy = batch#[0]
        preds, rawAttnPred, agePred, densityPred, biopsyPred = self.forward(imgs)
        binLoss = self.criterion(preds, labels)
        biopsyLoss = self.biopsyLoss(biopsyPred, biopsy)
        rawLoss = self.regLoss(rawAttnPred, rawAttn)

        ageLoss = self.ageLoss(agePred, age.long())/2
        densityLoss = self.densityLoss(densityPred, density.long())
        acc = self.accuracy(preds, labels.int())

        loss = binLoss #+ rawLoss + ageLoss + densityLoss + biopsyLoss

        self.wandb_log(
            val_loss=loss, val_acc=acc,
            #val_cancerL=binLoss,
            #val_rawLoss=rawLoss, val_ageLoss=ageLoss,
            #val_densityL=densityLoss, val_biopsyL=biopsyLoss
        )
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
