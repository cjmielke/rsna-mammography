import sys

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from misc import pfbeta
from models import Attn_Net_Gated, Attn_Net, MyAttn, initialize_weights, F1loss


class AttentionMIL_Lightning(pl.LightningModule):
    def __init__(self, lr=0.01, n_classes=1, nInput=1024, dropout=0.25, nReduced=None, nHiddenAttn=None, wandbRun=None,
                 opt='sgd', lossFun=None, weightDecay=0.0, l1_lambda=0.001,
                 focalAlpha=0.25, focalGamma=2.0, focalReduction=None, classifier='orig', argparse=None
                 ):
        super().__init__()

        self.args = argparse
        self.focalReduction = focalReduction
        self.focalAlpha = focalAlpha
        self.focalGamma = focalGamma
        self.l1_lambda = l1_lambda
        self.weightDecay = weightDecay
        self.lossFun = lossFun
        self.lr = lr
        self.opt = opt
        self.wandbRun = wandbRun
        self.criterion = nn.BCELoss()
        self.f1criterion = F1loss()
        self.accuracy = Accuracy()



        nReduced = nReduced or nInput//2
        nHiddenAttn = nHiddenAttn or nReduced//2

        print(f'Making model! {nInput} {nReduced}, {nHiddenAttn}')

        fc = [nn.Linear(nInput, nReduced), nn.ReLU(), nn.Dropout(dropout)]
        self.dimRed = nn.Sequential(*fc)

        self.attention_net1 = Attn_Net(L=nReduced, D=nHiddenAttn, dropout=dropout, n_classes=1)
        self.attention_net2 = Attn_Net(L=nReduced, D=nHiddenAttn, dropout=dropout, n_classes=1)
        self.attention_net3 = Attn_Net(L=nReduced, D=nHiddenAttn, dropout=dropout, n_classes=1)

        classifierIn = nInput if self.args.directClassification else nReduced

        '''        
        classifierLayers = []
        if self.args.batchNorm:
            classifierLayers.append(nn.BatchNorm1d(classifierIn))

        #if classifier == 'orig':
        classifierLayers.append(nn.Linear(classifierIn, 1, bias=False))
        #elif classifier == 'reludrop':
        #classifierLayers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(classifierIn, 1, bias=True)]

        self.classifier = nn.Sequential(*classifierLayers)
        '''

        self.classifier1 = nn.Linear(classifierIn, 1)
        self.classifier2 = nn.Linear(classifierIn, 1)
        self.classifier3 = nn.Linear(classifierIn, 1)

        self.finalClassification = nn.Linear(3,1, bias=False)

        #initialize_weights(self)           # FIXME

    def wandb_log(self, **kwargs):
        if not self.wandbRun: return
        for k, v in kwargs.items():
            self.wandbRun.log({k: v})

    def getAttentionScores(self, embs):
        with torch.no_grad():
            attentions = []
            rawAttentions = []
            for emb in embs:                    # list of training images
                A, h = self.attention_net(emb)
                #print(emb.shape, A.shape)
                A = torch.transpose(A, 1, 0)
                A_raw = A
                A = F.softmax(A, dim=1)             # A is weights, summing to 1, that ranks tile importance
                attentions.append(A)
                rawAttentions.append(A_raw)

            attentions = torch.concat(attentions)#.squeeze(dim=1)
            rawAttentions = torch.concat(rawAttentions)#.squeeze(dim=1)
            return attentions, rawAttentions

    def forward(self, bagBatch, attention_only=False):
        if self.args.model == 'MP':
            return self.forwardBagBatchMaxPool(bagBatch)

        imgEmbeddings = []
        tileClassifications = []
        tileScoreSums = []
        for bagOfEmbeddings in bagBatch:                    # list of training images
            #emb = nn.functional.normalize(emb, p=2.0, dim = 0)
            h = self.dimRed(bagOfEmbeddings)

            A1, _ = self.attention_net1(h)
            A2, _ = self.attention_net2(h)
            A3, _ = self.attention_net3(h)

            #print(emb.shape, A.shape)
            A1 = torch.transpose(A1, 1, 0)
            A2 = torch.transpose(A2, 1, 0)
            A3 = torch.transpose(A3, 1, 0)

            #A_raw = A
            #if attention_only: return A_raw

            A1 = F.softmax(A1, dim=1)             # A is weights, summing to 1, that ranks tile importance
            A2 = F.softmax(A2, dim=1)             # A is weights, summing to 1, that ranks tile importance
            A3 = F.softmax(A3, dim=1)             # A is weights, summing to 1, that ranks tile importance

            #print(A.shape, h.shape)       #             A          bag
            if self.args.directClassification:      # weights are applied to original embedding
                M1 = torch.mm(A1, bagOfEmbeddings)
                M2 = torch.mm(A2, bagOfEmbeddings)
                M3 = torch.mm(A3, bagOfEmbeddings)
            else:                                   # weights are applied to the dimensionality reduced embedding
                M1 = torch.mm(A1, h)                  # multiplies [1, 72] X [72, 512]   ->   (1,512) weighted avg vector
                M2 = torch.mm(A2, h)  # multiplies [1, 72] X [72, 512]   ->   (1,512) weighted avg vector
                M3 = torch.mm(A3, h)  # multiplies [1, 72] X [72, 512]   ->   (1,512) weighted avg vector
            # This is the weighted average vector that represents the whole bag (mammogram)

            imgEmbeddings.append((M1,M2,M3))

        imgEmbeddings1 = torch.cat([t[0] for t in imgEmbeddings])            # [batchSize, 512]
        imgEmbeddings2 = torch.cat([t[1] for t in imgEmbeddings])            # [batchSize, 512]
        imgEmbeddings3 = torch.cat([t[2] for t in imgEmbeddings])            # [batchSize, 512]

        P1 = self.classifier1(imgEmbeddings1).squeeze(1)                  # shape is [bs,1]
        P2 = self.classifier2(imgEmbeddings2).squeeze(1)                  # shape is [bs,1]
        P3 = self.classifier3(imgEmbeddings3).squeeze(1)                  # shape is [bs,1]

        #P = torch.sigmoid(P).squeeze(1)#[0]       # so is sigmoid(P)

        scores = torch.stack([P1, P2, P3], dim=1)
        #print(f'final scores : {scores.shape}')
        #print(f'final scores : {scores}')

        preds = self.finalClassification(scores)
        preds = torch.sigmoid(preds).squeeze(1)
        #print(preds)


        #sys.exit()

        return preds


    def forwardBagBatchMaxPool(self, bagBatch):
        pooled = []
        for embs in bagBatch:                    # list of training images
            # embs.shape is (55,1024)

            #maxPool = torch.max_pool1d(embs, kernel_size=1)
            #h = self.dimRedNet(embs)
            A, h = self.attention_net(embs)
            #maxPool, idx = torch.max(embs, dim=0)
            maxPool, idx = torch.max(h, dim=0)
            pooled.append(maxPool)

        imgEmbeddings = torch.stack(pooled)

        P = self.classifier(imgEmbeddings)                  # shape is [bs,1]
        predictions = torch.sigmoid(P).squeeze(1)#[0]       # so is sigmoid(P)

        return predictions


    def loss(self, pred, cancer):
        if self.lossFun=='BCE':
            loss = self.criterion(pred, cancer)
        elif self.lossFun=='F':
            loss = self.f1criterion(pred, cancer)
        elif self.lossFun=='BCE_F':
            loss = self.criterion(pred,cancer) + self.f1criterion(pred, cancer)
        elif self.lossFun.lower() == 'focal':
            loss = sigmoid_focal_loss(pred, cancer, alpha=self.focalAlpha, gamma=self.focalGamma, reduction=self.focalReduction)

        else:
            raise NotImplementedError

        if self.l1_lambda:
            l1_norm = sum(torch.linalg.norm(p, 1) for p in self.parameters())
            loss = loss + self.l1_lambda * l1_norm

        return loss

    def training_step(self, batch, batch_index):
        if self.args.metadata:
            emb, cancer, age, laterality, rows = batch
            pred = self.forward(emb)
        else:
            emb, cancer = batch
            pred = self.forward(emb)

        loss = self.loss(pred, cancer)
        acc = self.accuracy(pred, cancer.int())

        self.log('train_loss', loss, batch_size=len(batch))
        self.log('train_accuracy', acc, batch_size=len(batch))
        return dict(loss=loss, pred=pred, cancer=cancer)

    def training_epoch_end(self, outputs) -> None:
        preds = torch.cat([item['pred'] for item in outputs])
        cancer = torch.cat([item['cancer'] for item in outputs])
        acc = self.accuracy(preds, cancer.int())
        score = pfbeta(cancer, preds)
        #self.log_pfbeta_thresholds(cancer, preds, 'train')
        self.wandb_log(train_acc=acc, train_score=score)


    def validation_step(self, batch, batch_index):
        if self.args.metadata:
            emb, cancer, age, laterality, rows = batch
            pred = self.forward(emb)
        else:
            emb, cancer = batch
            pred = self.forward(emb)

        loss = self.loss(pred, cancer)
        acc = self.accuracy(pred, cancer.int())

        self.log('val_loss', loss, batch_size=len(batch), on_epoch=True)
        self.log('val_accuracy', acc, batch_size=len(batch), on_epoch=True)
        return dict(val_loss=loss, pred=pred, cancer=cancer)


    def log_pfbeta_thresholds(self, cancer, predsT, prefix):
        preds = predsT.cpu().detach().numpy()
        for thresh in [0.2, 0.5]:
            preds[preds<thresh] = 0.0
            s = pfbeta(cancer, preds)
            k = f'{prefix}pf_{thresh}'
            self.wandb_log(**{k:s})

        # one additional extreme test :
        preds = predsT.cpu().detach().numpy()
        preds[preds < 0.5] = 0.0
        for thresh in [0.9, 0.5]:
            preds[preds > thresh] = 1.0
            s = pfbeta(cancer, preds)
            k = f'{prefix}pf_{thresh}_P'
            self.wandb_log(**{k: s})

    def validation_epoch_end(self, outputs) -> None:
        preds = torch.cat([item['pred'] for item in outputs])
        cancer = torch.cat([item['cancer'] for item in outputs])
        acc = self.accuracy(preds, cancer.int())
        score = pfbeta(cancer, preds)
        print(cancer)
        print(preds)
        self.log_pfbeta_thresholds(cancer,preds, 'val')
        self.wandb_log(val_acc=acc, score=score)

    def configure_optimizers(self):
        #optim = torch.optim.SGD(self.parameters(), lr=0.06)
        if self.opt == 'sgd':
            optim = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weightDecay)
        elif self.opt == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weightDecay)
        else:
            raise NotImplementedError
        return optim

