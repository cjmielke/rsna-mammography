import torch
from torch import nn, Tensor
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


def f1_loss(input, target):
    eps = 1e-10
    #tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    #tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    #fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    #fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    tp = torch.sum((target * input), dim=0)
    #tn = torch.sum((1 - target) * (1 - input), dim=0)
    fp = torch.sum((1 - target) * input, dim=0)
    fn = torch.sum(target * (1 - input), dim=0)

    #p = tp / (tp + fp + K.epsilon())
    #r = tp / (tp + fn + K.epsilon())

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = 2 * p * r / (p + r + eps)

    #f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    f1 = torch.nan_to_num(f1, nan=0.0)
    return 1 - torch.mean(f1)



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


def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss


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
            nn.Tanh()]

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
        return A


"""
Attention MIL (Multiple Instance Learning) model
args:
    nInput: input feature dimension from tiles
    nReduced: dimensionality of reduced FC layer, which then feeds into both the classifier and attention head
    nHiddenAttn: Hidden units in attention network
    n_classes: number of classes (experimental usage for multiclass MIL)
"""




class AttentionMIL_Hacked(pl.LightningModule):
    def __init__(self, lr=0.01, n_classes=1, nInput=1024, dropout=0.25, nReduced=None, nHiddenAttn=None, wandbRun=None,
                 opt='sgd', gated=False, lossFun=None, weightDecay=0.0, l1_lambda=0.001,
                 focalAlpha=0.25, focalGamma=2.0, focalReduction=None
                 ):
        super().__init__()

        #self.args = args
        self.focalReduction = focalReduction
        self.focalAlpha = focalAlpha
        self.focalGamma = focalGamma
        self.l1_lambda = l1_lambda
        self.weightDecay = weightDecay
        self.lossFun = lossFun
        self.lr = lr
        self.opt = opt
        nReduced = nReduced or nInput//2
        nHiddenAttn = nHiddenAttn or nReduced//2

        print(f'Making model! {nInput} {nReduced}, {nHiddenAttn}')

        self.wandbRun = wandbRun
        fc = [nn.Linear(nInput, nReduced), nn.ReLU(), nn.Dropout(dropout)]
        self.dimred_net = nn.Sequential(*fc)

        if gated:
            self.attention_net = Attn_Net_Gated(L=nInput, D=nHiddenAttn, dropout=dropout, n_classes=1)
        else:
            self.attention_net = Attn_Net(L=nReduced, D=nHiddenAttn, dropout=dropout, n_classes=1)

        #self.classifier = nn.Linear(nReduced, n_classes)           # ORIGINAL
        #classifierLayers = [nn.Linear(nReduced, 1), nn.ReLU(), nn.Dropout(dropout)]
        classifierLayers = [nn.ReLU(), nn.Dropout(dropout), nn.Linear(nInput, 1)]
        self.classifier = nn.Sequential(*classifierLayers)
        print('hacked that SOB')

        initialize_weights(self)
        self.criterion = nn.BCELoss()
        self.f1criterion = F1loss()
        self.accuracy = Accuracy()

    def wandb_log(self, **kwargs):
        if not self.wandbRun: return
        for k, v in kwargs.items():
            self.wandbRun.log({k: v})

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


    def forward(self, bagBatch, attention_only=False):
        predictions = []
        for embs in bagBatch:                    # list of training images
            #emb = nn.functional.normalize(emb, p=2.0, dim = 0)
            h = self.dimred_net(embs)

            A_raw = self.attention_net(embs)
            #print(emb.shape, A.shape)
            A_raw = torch.transpose(A_raw, 1, 0)

            if attention_only: return A_raw

            A = F.softmax(A_raw, dim=1)             # A is weights, summing to 1, that ranks tile importance
            #print(A.shape, h.shape)       #             A          bag
            M = torch.mm(A, embs)             # multiplies [1, 72] X [72, 512]   ->   (1,512) weighted avg vector
            P = self.classifier(M)                  # shape is [1,1]
            prediction = torch.sigmoid(P)[0]       # so is sigmoid(P)
            #print(P.shape, P, prediction.shape)
            #prediction = prob
            predictions.append(prediction)

        #print(predictions)
        predictions = torch.concat(predictions)#.squeeze(dim=1)
        #print(predictions)
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
        #print(batch)   [[],[]]
        emb, cancer = batch
        #print('emb/label : ', emb.shape, cancer.shape)
        pred = self.forward(emb)
        #print(pred.shape, cancer.shape)
        #print(pred, cancer)
        #loss = self.criterion(pred, cancer) + self.f1criterion(pred, cancer)
        #loss = self.criterion(pred, cancer)
        #loss = self.f1criterion(pred, cancer)       # FIXME
        loss = self.loss(pred, cancer)

        #opt = self.optimizers()
        #opt.zero_grad()
        #self.manual_backward(loss)
        #opt.step()

        #if batch_index%10==0:
        #wandb_log(loss=loss)
        acc = self.accuracy(pred, cancer.int())

        self.log('train_loss', loss, batch_size=len(batch))
        self.log('train_accuracy', acc, batch_size=len(batch))
        #self.wandb_log(train_loss=loss, train_acc=acc)
        #return pred, cancer
        #return loss
        return dict(loss=loss, pred=pred, cancer=cancer)

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


    def validation_step(self, batch, batch_index):
        emb, cancer = batch
        pred = self.forward(emb)
        #loss = self.criterion(pred, cancer)
        loss = self.loss(pred, cancer)

        acc = self.accuracy(pred, cancer.int())
        #self.log('val_accuracy', acc, on_epoch=False)

        self.log('val_loss', loss, batch_size=len(batch), on_epoch=True)
        self.log('val_accuracy', acc, batch_size=len(batch), on_epoch=True)
        #self.wandb_log(val_loss=loss)
        #return loss
        #return pred, cancer
        return dict(loss=loss, pred=pred, cancer=cancer)


    def log_pfbeta_thresholds(self, cancer, predsT, prefix):
        preds = predsT.cpu().detach().numpy()
        for thresh in [0.2, 0.5]:
            preds[preds<thresh] = 0.0
            s = pfbeta(cancer, preds)
            k = f'{prefix}pf_{thresh}'
            #k = "pfbeta_"+str(thresh)
            self.wandb_log(**{k:s})

        # one additional extreme test :
        preds = predsT.cpu().detach().numpy()
        preds[preds < 0.5] = 0.0
        for thresh in [0.9, 0.5]:
            preds[preds > thresh] = 1.0
            s = pfbeta(cancer, preds)
            k = f'{prefix}pf_{thresh}_P'
            # k = "pfbeta_"+str(thresh)
            self.wandb_log(**{k: s})

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
            optim = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weightDecay)
        elif self.opt == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weightDecay)
        else:
            raise NotImplementedError
        return optim

