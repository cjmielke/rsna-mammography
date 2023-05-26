import sys

import timm
import torch
import pytorch_lightning as pl
from torch import nn, Tensor
from torch.nn import BCELoss
from torch.nn.modules.loss import _Loss
from torchmetrics import Accuracy


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


def getModel(encoder):
    model = timm.create_model(encoder, pretrained=True, num_classes=0)
    o = model(torch.randn(2, 3, 224, 224))
    print(f'Original shape: {o.shape}')
    assert len(o.shape)==2
    embSize = o.shape[1]
    #o = model.forward_features(torch.randn(2, 3, 224, 224))
    #print(f'Unpooled shape: {o.shape}')
    return model, embSize


class F1loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self) -> None:
        super(F1loss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        eps = 1e-10

        tp = torch.sum((target * input), dim=0)
        tn = torch.sum((1 - target) * (1 - input), dim=0)
        fp = torch.sum((1 - target) * input, dim=0)
        fn = torch.sum(target * (1 - input), dim=0)

        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)

        f1 = 2 * p * r / (p + r + eps)
        f1 = torch.nan_to_num(f1, nan=0.0)
        return 1 - torch.mean(f1)




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



class Model(pl.LightningModule):
    def __init__(self, ARGS):
        super().__init__()
        self.args = ARGS

        if self.args.colorizer:
            self.colorizer = ColorizerModel(n=ARGS.colorizer)

        if self.args.encoder == 'stupid':
            self.encoder = nn.Sequential(*[
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ])
            embSize = 3
        else:
            self.encoder, embSize = getModel(self.args.encoder)
            print(f'embSize is {embSize}')



        classifierLayers = [
            #nn.BatchNorm1d(embSize),
            #nn.ReLU(),
            #nn.Linear(embSize, embSize//2),
            #nn.BatchNorm1d(embSize),
            nn.ReLU(),
            nn.Linear(embSize, 1, bias=True)
        ]
        self.classifier = nn.Sequential(*classifierLayers)
        self.criterion = BCELoss()
        self.f1criterion = F1loss()
        self.accuracy = Accuracy()

    def wandb_log(self, **kwargs):
        for k,v in kwargs.items(): self.log(k, v)

    '''
    def forward(self, batch):
        loss = 0
        preds = []
        lbls = []
        for img, lbl in batch:
            emb = self.encoder(img)
            pred = self.classifier(emb)
            loss = loss + self.criterion(pred, lbl)
            preds.append(pred)
            lbls.append(lbl)

        embeddings = self.encoder(x)
        preds = self.classifier(embeddings)
        #print(preds)
        preds = torch.sigmoid(preds)
        preds = preds.squeeze(1)    # why?
        #print(preds)

        rawAttn = self.rawAttn(embeddings)      # regression of attention score

        #print('preds', preds.min(), preds.max(), preds.shape)
        return preds, rawAttn
    '''

    def forward(self, imgs):
        if self.args.colorizer:
            imgs = self.colorizer(imgs)

        embs = self.encoder(imgs)
        print(f'embs {embs.shape}')
        preds = self.classifier(embs)
        preds = torch.squeeze(preds, dim=1)
        return preds

    def training_step(self, batch, batch_index):
        imgs, labels = batch#[0]
        preds = self.forward(imgs)
        preds = torch.sigmoid(preds)

        loss = self.criterion(preds, labels)
        acc = self.accuracy(preds, labels.int())

        self.wandb_log(train_loss=loss, train_acc=acc)
        return dict(loss=loss, acc=acc, preds=preds, cancer=labels)

    def training_epoch_end(self, outputs) -> None:
        acc, loss, score = self.agg(outputs)
        self.wandb_log(train_epoch_acc=acc, train_epoch_loss=loss, train_score=score)


    def validation_step(self, batch, batch_index):
        imgs, labels = batch#[0]
        #print(imgs.shape, labels)
        preds = self.forward(imgs)
        preds = torch.sigmoid(preds)
        #print(preds, labels)

        loss = self.criterion(preds, labels)
        acc = self.accuracy(preds, labels.int())

        self.wandb_log(val_loss=loss, val_acc=acc)
        return dict(loss=loss, acc=acc, preds=preds, cancer=labels)

    def agg(self, outputs):
        loss = torch.stack([item['loss'] for item in outputs]).mean()
        acc = torch.stack([item['acc'] for item in outputs]).mean()
        preds = torch.cat([item['preds'] for item in outputs])
        cancer = torch.cat([item['cancer'] for item in outputs])
        score = pfbeta(cancer, preds)
        return acc, loss, score

    def validation_epoch_end(self, outputs) -> None:
        '''
        loss = torch.stack([item['loss'] for item in outputs]).mean()
        acc = torch.stack([item['acc'] for item in outputs]).mean()

        preds = torch.cat([item['preds'] for item in outputs])
        cancer = torch.cat([item['cancer'] for item in outputs])

        score = pfbeta(cancer, preds)
        '''

        acc, loss, score = self.agg(outputs)
        self.wandb_log(val_epoch_acc=acc, val_epoch_loss=loss, score=score)


    def configure_optimizers(self):
        if self.args.opt == 'sgd':
            optim = torch.optim.SGD(self.parameters(), lr=self.args.lr)
        elif self.args.opt == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        else:
            raise ValueError
        return optim






