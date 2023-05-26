import sys

import pytorch_lightning as pl
import timm
import torch
import torchmetrics
import wandb
from torch import nn as nn
from config import Config


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        """
        Focal Loss function class taken from:
        https://github.com/clcarwin/focal_loss_pytorch
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def probabilistic_f1(labels, preds, beta=1):
    """
    Function taken from Awsaf's notebook:
    https://www.kaggle.com/code/awsaf49/rsna-bcd-efficientnet-tf-tpu-1vm-train
    """
    eps = 1e-5
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp + eps)
    c_recall = ctp / (y_true_count + eps)
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + eps)
        return result
    else:
        return 0.0


def wandb_log(**kwargs):
    for k, v in kwargs.items():
        wandb.log({k: v})



class RSNAModel(pl.LightningModule):
    def __init__(self, pretrained=True):
        super(RSNAModel, self).__init__()
        # Model Architecture
        self.model = timm.create_model(Config['MODEL_NAME'], pretrained=pretrained, num_classes=0)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name.startswith('block'):
                    param.requires_grad=False
                print(name, param.requires_grad)
        o = self.model(torch.randn(2, 3, 1024, 1024))
        print(f'Output tensor shape : {o.shape}')
        #print(self.model.feature_info)

        #self.n_features = self.model.head.in_features
        #self.n_features = self.model.feature_info[-1]['num_chs']
        self.n_features = o.shape[-1]
        print(f'Embedding size : {self.n_features}')
        #self.model.reset_classifier(0)
        self.fc = nn.Linear(self.n_features, Config['NUM_LABELS'])

        # Loss functions
        self.train_loss = nn.BCEWithLogitsLoss()
        self.valid_loss = nn.BCEWithLogitsLoss()

        # Metric
        self.f1 = torchmetrics.F1Score(task='binary')

    def forward(self, x):
        features = self.model(x)
        #print(features.shape)
        output = self.fc(features)
        return output

    def training_step(self, batch, batch_idx):
        imgs = batch[0]
        target = batch[1]

        out = self(imgs).view(-1)
        train_loss = self.train_loss(out, target)

        logs = {'loss': train_loss}
        #wandb_log(train_step_loss=train_loss.item())
        return {'loss': train_loss, 'log': logs}

    def training_epoch_end(self, outputs):
        #avg_prob_f1 = torch.stack([x['prob_f1'] for x in outputs]).mean()
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        wandb_log(train_loss=avg_loss)

    def validation_step(self, batch, batch_idx):
        imgs = batch[0]
        target = batch[1]

        out = self(imgs).view(-1)
        valid_loss = self.valid_loss(out, target)

        self.f1(out, target)
        f1_current = self.f1(out, target)
        prob_f1 = probabilistic_f1(target.cpu().detach(), out.cpu().detach())
        self.log('f1_valid_epoch', self.f1, on_epoch=True, on_step=True)

        #wandb_log(val_step_loss=valid_loss.item(), f1_step_valid=f1_current, prob_f1=prob_f1)

        return {'val_loss': valid_loss, 'f1_score': f1_current, 'prob_f1': prob_f1}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_prob_f1 = torch.stack([x['prob_f1'] for x in outputs]).mean()

        logs = {'val_loss': avg_loss}
        wandb_log(val_loss=avg_loss, avg_prob_f1=avg_prob_f1)

        print(f"val_loss: {avg_loss}")
        return {'avg_val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=Config['LR'])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=Config['T_max'],
            eta_min=Config['min_lr']
        )

        return [opt], [sch]

