import sys

import timm
import torch
from torch import nn, Tensor, autograd
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.modules.loss import _Loss
from torchmetrics import Accuracy
from torchvision.ops import sigmoid_focal_loss

from misc import pfbeta


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)




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



"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=0.0, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()
        ]

        if dropout:
            self.module.append(nn.Dropout(dropout))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes




"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""
class Attn_Net_Gated(nn.Module):

    def __init__(self, L=1024, D=256, dropout=0.0, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
Attention MIL (Multiple Instance Learning) model
args:
    nInput: input feature dimension from tiles
    nReduced: dimensionality of reduced FC layer, which then feeds into both the classifier and attention head
    nHiddenAttn: Hidden units in attention network
    n_classes: number of classes (experimental usage for multiclass MIL)
"""



autograd.set_detect_anomaly(True)


def getTimmModel(encoder):
    model = timm.create_model(encoder, pretrained=True, num_classes=0)
    o = model(torch.randn(2, 3, 224, 224))
    print(f'Original shape: {o.shape}')
    #o = model.forward_features(torch.randn(2, 3, 224, 224))
    #print(f'Unpooled shape: {o.shape}')
    return model, o.shape[1]





class SimpleModel(nn.Module):
    def __init__(self, k=3, outDim=256):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.outDim = outDim

        f = 8
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

        # third stage at 56 tensor size
        f3 = 3*f
        layers += [
            nn.Conv2d(f2, f3, (k, k), padding='same'),
            nn.ReLU(),
            nn.Conv2d(f3, f3, (k, k), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        ]

        # forth stage at 28 tensor size
        f4 = 4*f
        layers += [
            nn.Conv2d(f3, f4, (k, k), padding='same'),
            nn.ReLU(),
            nn.Conv2d(f4, f4, (k, k), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        ]

        # 14 x 14 x 16
        layers += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(14*14*f4, self.outDim)
        ]

        self.features = nn.Sequential(*layers)


    def forward(self, rgbImg):
        f = self.features(rgbImg)
        return f




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



class EncoderModel(nn.Module):

    def __init__(self, ARGS):
        super().__init__()
        self.args = ARGS

        if self.args.colorize:
            self.colorizer = ColorizerModel(n=self.args.colorize)

        if self.args.backbone == 'simple':
            self.encoder = SimpleModel()
            self.embSize = self.encoder.outDim
        else:
            self.encoder, self.embSize = getTimmModel(self.args.backbone)

    def forward(self, imgs):

        if self.args.colorize:
            imgs = self.colorizer(imgs)

        embs = self.encoder(imgs)
        return embs


class EndToEnd_AMIL(pl.LightningModule):
    def __init__(self, ARGS, model='A', backbone='efficientnet_b3', lr=0.01, n_classes=1, dropout=0.25, nReduced=None,
                 nHiddenAttn=None, wandbRun=None, opt='sgd', gated=False, lossFun=None, weightDecay=0.0, l1_lambda=0.0,
                 focalAlpha=0.25, focalGamma=2.0, focalReduction=None
                 ):
        super().__init__()

        '''
        self.model = model
        self.backbone, self.embSize = getTimmModel(backbone)
        '''
        self.args = ARGS

        self.encoder = EncoderModel(ARGS)

        self.args = ARGS
        self.focalReduction = focalReduction
        self.focalAlpha = focalAlpha
        self.focalGamma = focalGamma
        self.l1_lambda = l1_lambda
        self.weightDecay = weightDecay
        self.lossFun = lossFun
        self.lr = lr
        self.opt = opt

        nInput = self.encoder.embSize
        nReduced = nReduced or nInput//2
        nHiddenAttn = nHiddenAttn or nReduced//2

        print(f'Making model! {nInput} {nReduced}, {nHiddenAttn}')
        self.wandbRun = wandbRun

        fc = [
            nn.Linear(nInput, nReduced),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        self.dimRedNet = nn.Sequential(*fc)


        if gated:
            attention_net = Attn_Net_Gated(L=nReduced, D=nHiddenAttn, dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=nReduced, D=nHiddenAttn, dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)


        fc = [
            #nn.Linear(nReduced, nReduced//2),
            nn.BatchNorm1d(nInput),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nInput, 1, bias=True)
        ]
        self.classifier = nn.Sequential(*fc)

        #self.classifier = nn.Linear(nReduced, n_classes)
        #self.classifier = nn.Linear(nInput, n_classes)


        #self.classifierB = nn.Linear(nInput, 1)

        #initialize_weights(self)
        self.criterion = nn.BCELoss()
        self.f1criterion = F1loss()
        self.accuracy = Accuracy()

    def wandb_log(self, **kwargs):
        for k, v in kwargs.items():
            self.log(k, v, sync_dist=True)

    def getAttentionScores(self, embs):
        with torch.no_grad():
            attentions = []
            for emb in embs:                    # list of training images
                A, h = self.attention_net(emb)
                #print(emb.shape, A.shape)
                A = torch.transpose(A, 1, 0)
                A_raw = A
                A = F.softmax(A, dim=1)             # A is weights, summing to 1, that ranks tile importance
                attentions.append(A)

            attentions = torch.concat(attentions)#.squeeze(dim=1)
            return attentions

    def forward(self, imgBatch):
        #print(type(imgBatch), len(imgBatch))
        #print(imgBatch[0].shape)
        embeddings = []
        # at bs==1, imgBatch = [single tensor(55,3,224,224)]
        for imageTileStack in imgBatch:
            #print(imageTileStack.shape)                # (55, 3, 224, 224)
            embs = self.encoder(imageTileStack)
            #print(emb.shape)                           # (55,1024)
            embeddings.append(embs)                      # each source image could have fewer img patches

        #embeddings = torch.stack(embeddings)
        #print(embeddings.shape)                        # (1,55,1024)

        if self.args.model == 'A':
            return self.forwardBagBatch(embeddings)
        elif self.args.model == 'MP':
            return self.forwardBagBatchMaxPool(embeddings)
        elif self.args.model == 'MAX':
            return self.forwardBagBatchMaxScore(embeddings)
        else:
            raise NotImplementedError

    def forwardBagBatch(self, bagBatch, attention_only=False):
        #predictions = []
        imgVectors = []
        for embs in bagBatch:                    # list of training images
            # embs.shape is (55,1024)
            #emb = nn.functional.normalize(emb, p=2.0, dim = 0)

            A, h = self.attention_net(embs)
            A = torch.transpose(A, 1, 0)

            A_raw = A
            if attention_only: return A_raw

            A = F.softmax(A, dim=1)             # A is weights, summing to 1, that ranks tile importance
            #print(A.shape, h.shape)       #             A          bag
            M = torch.mm(A, h)             # multiplies [1, 72] X [72, 512]   ->   (1,512) weighted avg vector
            imgVectors.append(M)

        imgVectors = torch.cat(imgVectors)
        #print(imgVectors.shape)             # (16,128)  (batchSize, BagVector)

        P = self.classifier(imgVectors)                  # shape is [1,1]
        predictions = torch.sigmoid(P)#[0]       # so is sigmoid(P)

        return predictions.squeeze(1)

    def forwardBagBatchMaxPool(self, bagBatch, attention_only=False):
        predictions = []
        for embs in bagBatch:                    # list of training images
            # embs.shape is (55,1024)

            #maxPool = torch.max_pool1d(embs, kernel_size=1)
            h = self.dimRedNet(embs)
            #maxPool, idx = torch.max(embs, dim=0)
            maxPool, idx = torch.max(h, dim=0)
            #print(maxPool.shape)           [1024]
            #c

            #P = self.classifierB(maxPool)                  # shape is [1,1]
            P = self.classifier(maxPool)                  # shape is [1,1]
            prediction = torch.sigmoid(P)#[0]       # so is sigmoid(P)
            predictions.append(prediction)

        #print(predictions)
        predictions = torch.concat(predictions)#.squeeze(dim=1)
        #print(predictions)
        return predictions

    def forwardBagBatchMaxScore(self, bagBatch, attention_only=False):
        predictions = []
        for embs in bagBatch:                    # list of training images
            # embs.shape is (55,1024)
            #maxPool = torch.max_pool1d(embs, kernel_size=1)
            #h = self.dimRedNet(embs)                        # Ntiles x H
            #P = self.classifier(h)                          # shape is [Ntiles,1]
            P = self.classifier(embs)                          # shape is [Ntiles,1]
            P = torch.max(P)
            prediction = torch.sigmoid(P)#[0]       # so is sigmoid(P)
            predictions.append(prediction)

        #print(predictions)
        predictions = torch.stack(predictions)#.squeeze(dim=1)
        #print(predictions)
        #sys.exit()
        return predictions



    def loss(self, pred, cancer):
        cancer = cancer.float()
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
        tileSets, cancer = batch

        #print('emb/label : ', emb.shape, cancer.shape)
        pred = self.forward(tileSets)
        #print(cancer, cancer.dtype, pred)
        loss = self.loss(pred, cancer)

        acc = self.accuracy(pred, cancer.int())

        self.log('train_loss', loss, batch_size=len(batch))
        self.log('train_accuracy', acc, batch_size=len(batch))
        #return dict(loss=loss, pred=pred, cancer=cancer)
        return dict(loss=loss)

    '''
    def training_epoch_end(self, outputs) -> None:
        #preds = torch.cat([item[0] for item in outputs])
        #cancer = torch.cat([item[1] for item in outputs])
        preds = torch.cat([item['pred'] for item in outputs])
        cancer = torch.cat([item['cancer'] for item in outputs])
        acc = self.accuracy(preds, cancer.int())
        score = pfbeta(cancer, preds)
        self.log_pfbeta_thresholds(cancer, preds, 'train')
        print(cancer)
        print(preds)
        #self.log('val_accuracy', acc, batch_size=len(batch), on_epoch=True)
        self.wandb_log(train_acc=acc, train_score=score)
    '''

    def validation_step(self, batch, batch_index):
        tileSets, cancer = batch
        pred = self.forward(tileSets)
        loss = self.loss(pred, cancer)
        acc = self.accuracy(pred, cancer.int())

        self.log('val_loss', loss, batch_size=len(batch), on_epoch=True)
        #self.log('val_accuracy', acc, batch_size=len(batch), on_epoch=True)
        return dict(loss=loss, pred=pred, cancer=cancer)


    def log_pfbeta_thresholds(self, cancer, predsT, prefix):
        preds = predsT.cpu().detach().numpy()
        for thresh in [0.2, 0.5]:
            preds[preds<thresh] = 0.0
            s = pfbeta(cancer, preds)
            k = f'{prefix}pf_{thresh}'
            #k = "pfbeta_"+str(thresh)
            #self.wandb_log(**{k:s})
            self.log(k, s, sync_dist=True)

        # one additional extreme test :
        preds = predsT.cpu().detach().numpy()
        preds[preds < 0.5] = 0.0
        for thresh in [0.9, 0.5]:
            preds[preds > thresh] = 1.0
            s = pfbeta(cancer, preds)
            k = f'{prefix}pf_{thresh}_P'
            # k = "pfbeta_"+str(thresh)
            #self.wandb_log(**{k: s})
            self.log(k, s, sync_dist=True)

    def validation_epoch_end(self, outputs) -> None:
        preds = torch.cat([item['pred'] for item in outputs])
        cancer = torch.cat([item['cancer'] for item in outputs])
        acc = self.accuracy(preds, cancer.int())
        score = pfbeta(cancer, preds)
        self.log_pfbeta_thresholds(cancer,preds, 'val')
        print(cancer)
        print(preds)
        #self.log('val_accuracy', acc, batch_size=len(batch), on_epoch=True)
        self.wandb_log(val_acc=acc, score=score)

    def configure_optimizers(self):
        #optim = torch.optim.SGD(self.parameters(), lr=0.06)
        if self.opt == 'sgd':
            #self.backbone.requires_grad_(False)        # FIXME - why did I do this?
            optim = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weightDecay)
            #params = list(self.attention_net.parameters()) + list(self.classifier.parameters())
            #optim = torch.optim.SGD(params, lr=self.lr, weight_decay=self.weightDecay)
        elif self.opt == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weightDecay)
        else:
            raise NotImplementedError
        return optim


