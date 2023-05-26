import torch
from torch import nn, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.modules.loss import _Loss
from torchmetrics import Accuracy

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
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

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



class AttentionMIL(nn.Module):
    def __init__(self, n_classes=1, nInput=1024, nReduced=512, nHiddenAttn=256):
        super(AttentionMIL, self).__init__()

        fc = [nn.Linear(nInput, nReduced), nn.ReLU(), nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=nReduced, D=nHiddenAttn, dropout=0.25, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.classifier = nn.Linear(nReduced, n_classes)
        initialize_weights(self)

    def forward(self, emb, attention_only=False):
        #h = kwargs['x_path']
        h = emb.squeeze(dim=0)

        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)

        A_raw = A

        if attention_only:
            return A_raw

        A = F.softmax(A, dim=1)
        #print(A.shape, h.shape)
        M = torch.mm(A, h)
        prediction = self.classifier(M)
        prediction = torch.sigmoid(prediction)
        return prediction

    def get_slide_features(self, **kwargs):
        h = kwargs['x_path']

        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)

        if 'attention_only' in kwargs.keys():
            if kwargs['attention_only']:
                return A

        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        return M

    def relocate(self, device):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifier = self.classifier.to(device)






class AttentionMIL_Lightning(pl.LightningModule):
    def __init__(self, n_classes=1, nInput=1024, nReduced=None, nHiddenAttn=None, wandbRun=None):
        super().__init__()

        #self.args = args
        nReduced = nReduced or nInput//2
        nHiddenAttn = nHiddenAttn or nReduced//2

        self.wandbRun = wandbRun
        fc = [nn.Linear(nInput, nReduced), nn.ReLU(), nn.Dropout(0.25)]
        #attention_net = Attn_Net_Gated(L=nReduced, D=nHiddenAttn, dropout=0.25, n_classes=1)
        attention_net = Attn_Net(L=nReduced, D=nHiddenAttn, dropout=0.25, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.classifier = nn.Linear(nReduced, n_classes)
        initialize_weights(self)
        self.criterion = nn.BCELoss()
        self.f1criterion = F1loss()
        self.accuracy = Accuracy()

    def wandb_log(self, **kwargs):
        if not self.wandbRun: return
        for k, v in kwargs.items():
            self.wandbRun.log({k: v})

    def forward(self, embs, attention_only=False):
        predictions = []
        for emb in embs:                    # list of training images
            #h = kwargs['x_path']

            #emb = nn.functional.normalize(emb, p=2.0, dim = 0)

            A, h = self.attention_net(emb)
            #print(emb.shape, A.shape)
            A = torch.transpose(A, 1, 0)

            A_raw = A

            if attention_only:
                return A_raw

            A = F.softmax(A, dim=1)             # A is weights, summing to 1, that ranks tile importance
            #print(A.shape, h.shape)
            M = torch.mm(A, h)             # multiplies [1, 72] X [72, 512]   ->   (1,512) weighted avg vector
            P = self.classifier(M)
            prediction = torch.sigmoid(P)[0]
            #print(P.shape, P, prediction.shape)
            #prediction = prob
            predictions.append(prediction)

        #print(predictions)
        predictions = torch.concat(predictions)#.squeeze(dim=1)
        #print(predictions)
        return predictions


    def training_step(self, batch, batch_index):
        #print(batch)   [[],[]]
        emb, cancer = batch
        #print('emb/label : ', emb.shape, cancer.shape)
        pred = self.forward(emb)
        #print(pred.shape, cancer.shape)
        #print(pred, cancer)
        loss = self.criterion(pred, cancer) + self.f1criterion(pred, cancer)

        #opt = self.optimizers()
        #opt.zero_grad()
        #self.manual_backward(loss)
        #opt.step()

        #if batch_index%10==0:
        #wandb_log(loss=loss)
        acc = self.accuracy(pred, cancer.int())

        self.log('train_loss', loss, batch_size=len(batch))
        self.log('train_accuracy', acc, batch_size=len(batch))
        self.wandb_log(train_loss=loss, train_acc=acc)
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
        print(cancer)
        print(preds)
        #self.log('val_accuracy', acc, batch_size=len(batch), on_epoch=True)
        self.wandb_log(train_epoch_acc=acc, train_score=score)


    def validation_step(self, batch, batch_index):
        emb, cancer = batch
        pred = self.forward(emb)
        loss = self.criterion(pred, cancer)
        acc = self.accuracy(pred, cancer.int())
        #self.log('val_accuracy', acc, on_epoch=False)

        self.log('val_loss', loss, batch_size=len(batch), on_epoch=True)
        self.log('val_accuracy', acc, batch_size=len(batch), on_epoch=True)
        self.wandb_log(val_loss=loss)
        #return loss
        return pred, cancer

    def validation_epoch_end(self, outputs) -> None:
        preds = torch.cat([item[0] for item in outputs])
        cancer = torch.cat([item[1] for item in outputs])
        acc = self.accuracy(preds, cancer.int())
        score = pfbeta(cancer, preds)
        print(cancer)
        print(preds)
        #self.log('val_accuracy', acc, batch_size=len(batch), on_epoch=True)
        self.wandb_log(val_acc=acc, score=score)

    def configure_optimizers(self):
        #optim = torch.optim.SGD(self.parameters(), lr=0.06)
        optim = torch.optim.SGD(self.parameters(), lr=0.1)
        return optim

