import sys

import pandas as pd
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



        nHiddenAttn = nHiddenAttn or nReduced//2

        print(f'Making model! {nInput} {nReduced}, {nHiddenAttn}')

        if self.args.deep:
            nReduced = nReduced or nInput // 4
            fc = [
                nn.Linear(nInput, nInput//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(nInput//2, nReduced),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        else:
            nReduced = nReduced or nInput // 2
            fc = [
                nn.Linear(nInput, nReduced),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.dimRed = nn.Sequential(*fc)

        if self.args.attn == 'gated':
            self.attention_net = Attn_Net_Gated(L=nReduced, D=nHiddenAttn, dropout=dropout, n_classes=1)
        elif self.args.attn == 'nongated':
            self.attention_net = Attn_Net(L=nReduced, D=nHiddenAttn, dropout=dropout, n_classes=1)
        elif self.args.attn == 'myattn':
            self.attention_net = MyAttn(L=nReduced, D=nHiddenAttn, dropout=dropout, n_classes=1)
        else:
            raise NotImplementedError

        self.tileClassifier = nn.Sequential(*[
            #nn.Linear(nInput, nInput//2),
            #nn.ReLU(),
            #nn.Dropout(dropout),
            #nn.Linear(nInput//2, nInput//4),
            #nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(nInput, 1, bias=True),
            #nn.Linear(nInput//4, 1, bias=True),
            nn.Sigmoid()
        ])


        classifierIn = nInput if self.args.directClassification else nReduced

        classifierLayers = []
        if self.args.batchNorm:
            classifierLayers.append(nn.BatchNorm1d(classifierIn))

        #if classifier == 'orig':
        #classifierLayers.append(nn.Linear(classifierIn, 1, bias=False))
        #elif classifier == 'reludrop':
        classifierLayers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(classifierIn, 1, bias=True)]

        self.classifier = nn.Sequential(*classifierLayers)

        if self.args.metadata:
            self.finalClassification = nn.Linear(5, 1, bias=False)
        else:
            self.finalClassification = nn.Linear(3, 1, bias=False)

        #initialize_weights(self)           # FIXME

    def wandb_log(self, **kwargs):
        if not self.wandbRun: return
        for k, v in kwargs.items():
            self.wandbRun.log({k: v})

    def getAttentionScores(self, embs):
        with torch.no_grad():
            attentions = []
            rawAttentions = []
            for bagOfEmbeddings in embs:                    # list of training images
                h = self.dimRed(bagOfEmbeddings)
                A, h = self.attention_net(h)
                #print(emb.shape, A.shape)
                A = torch.transpose(A, 1, 0)
                A_raw = A
                A = F.softmax(A, dim=1)             # A is weights, summing to 1, that ranks tile importance
                attentions.append(A)
                rawAttentions.append(A_raw)

            attentions = torch.concat(attentions)#.squeeze(dim=1)
            rawAttentions = torch.concat(rawAttentions)#.squeeze(dim=1)
            return attentions, rawAttentions

    def forward(self, bagBatch, age=None, laterality=None, attention_only=False):

        imgEmbeddings = []
        tileClassifications = []
        tileScoreSums = []
        for bagOfEmbeddings in bagBatch:                    # list of training images
            #emb = nn.functional.normalize(emb, p=2.0, dim = 0)
            h = self.dimRed(bagOfEmbeddings)
            A, h = self.attention_net(h)
            #print(emb.shape, A.shape)
            A = torch.transpose(A, 1, 0)

            A_raw = A
            if attention_only: return A_raw

            A = F.softmax(A, dim=1)             # A is weights, summing to 1, that ranks tile importance
            #print(A.shape, h.shape)       #             A          bag
            tc = self.tileClassifier(bagOfEmbeddings)       # returns tile-level cancer score
            tileScoreSums.append(torch.sum(tc))
            MTC = torch.mm(A, tc)                           # applies attention weights to the cancer scores -> returns scalar for whole slide
            tileClassifications.append(MTC)
            if self.args.directClassification:      # weights are applied to original embedding
                M = torch.mm(A, bagOfEmbeddings)
            else:                                   # weights are applied to the dimensionality reduced embedding
                M = torch.mm(A, h)                  # multiplies [1, 72] X [72, 512]   ->   (1,512) weighted avg vector
            # This is the weighted average vector that represents the whole bag (mammogram)
            imgEmbeddings.append(M)

        imgEmbeddings = torch.cat(imgEmbeddings)            # [batchSize, 512]
        tileClassifications = torch.cat(tileClassifications).squeeze(1)
        tileScoreSums = torch.stack(tileScoreSums)#.squeeze(1)
        #tileClassifications = torch.sigmoid(tileClassifications).squeeze(1)       # tile-level only
        #print(f'tile classifications : {tileClassifications.shape}')
        #print(f'tile classifications : {tileClassifications}')
        #print(imgEmbeddings.shape)

        P = self.classifier(imgEmbeddings).squeeze(1)                  # shape is [bs,1]
        #P = torch.sigmoid(P).squeeze(1)#[0]       # so is sigmoid(P)

        scores = [P, tileClassifications, tileScoreSums]
        if self.args.metadata:
            scores += [age, laterality]

        scores = torch.stack(scores, dim=1)
        #print(f'final scores : {scores.shape}')
        #print(f'final scores : {scores}')

        preds = self.finalClassification(scores)
        preds = torch.sigmoid(preds).squeeze(1)
        #print(preds)


        #sys.exit()

        return preds
        #return torch.sigmoid(tileScoreSums)

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
            pred = self.forward(emb, age=age, laterality=laterality)
        else:
            emb, cancer = batch
            pred = self.forward(emb)
        #print('pred', pred.min(), pred.max())
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
        rows = None
        if self.args.metadata:
            emb, cancer, age, laterality, rows = batch
            pred = self.forward(emb, age=age, laterality=laterality)
        else:
            emb, cancer = batch
            pred = self.forward(emb)

        loss = self.loss(pred, cancer)
        acc = self.accuracy(pred, cancer.int())

        self.log('val_loss', loss, batch_size=len(batch), on_epoch=True)
        self.log('val_accuracy', acc, batch_size=len(batch), on_epoch=True)
        return dict(val_loss=loss, pred=pred, cancer=cancer, rows=rows)



    def validation_epoch_end(self, outputs) -> None:
        preds = torch.cat([item['pred'] for item in outputs])
        cancer = torch.cat([item['cancer'] for item in outputs])
        acc = self.accuracy(preds, cancer.int())
        score = pfbeta(cancer, preds)
        print(cancer)
        print(preds)

        df = pd.concat([item['rows'] for item in outputs])
        df['preds'] = preds.detach().cpu().numpy()
        dfscore = pfbeta(df['cancer'], df['preds'])

        maxPredsPerLat = df[['patient_id','laterality','preds']].groupby(['patient_id', 'laterality']).max().reset_index()
        #del df['preds']
        for rn, row in maxPredsPerLat.iterrows():
            #df[(df.patient_id==row.patient_id) & (df.laterality==row.laterality)]['preds'] = row['preds']
            df.loc[(df.patient_id==row.patient_id) & (df.laterality==row.laterality),'preds'] = row['preds']

        dfscoreF = pfbeta(df['cancer'], df['preds'])
        df['preds'] = 1.0*(df.preds > self.args.predThresh)
        dfscoreFT = pfbeta(df['cancer'], df['preds'])

        self.log_pfbeta_thresholds(cancer,preds, 'val')

        maxPredsPerPatient = df[['patient_id', 'preds']].groupby('patient_id').max().reset_index()
        for rn, row in maxPredsPerPatient.iterrows():
            df.loc[(df.patient_id==row.patient_id),'preds'] = row['preds']

        dfscorePF = pfbeta(df['cancer'], df['preds'])
        df['preds'] = 1.0*(df.preds > self.args.predThresh)
        dfscorePFT = pfbeta(df['cancer'], df['preds'])


        self.wandb_log(val_acc=acc, score=score, dfscore=dfscore,dfscoreF=dfscoreF,dfscoreFT=dfscoreFT,
                       dfscorePFT=dfscorePFT, dfscorePF=dfscorePF)







    def configure_optimizers(self):
        #optim = torch.optim.SGD(self.parameters(), lr=0.06)
        if self.opt == 'sgd':
            optim = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weightDecay)
        elif self.opt == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weightDecay)
        else:
            raise NotImplementedError
        return optim

