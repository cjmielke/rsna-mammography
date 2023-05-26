import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from tqdm import trange

from nyuModel import ModifiedDenseNet121
import torch.nn.functional as F

from data import getData, FullDataset
import pytorch_lightning as pl


#model = ModifiedDenseNet121(num_classes=4)
#o = model(torch.randn(2, 3, 224, 224))
#print(f'output : {o.shape}')
# sys.exit()

dev = torch.device('cuda')


def pfbeta(labels, preds, beta=1):
    try: labels = labels.cpu()
    except: pass

    #preds = preds.cpu()

    preds = preds.clip(0, 1)
    y_true_count = labels.sum()

    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.0





#class FullModel(nn.Module):
class FullModel(pl.LightningModule):

    def __init__(self, A):
        self.args = A
        super().__init__()
        self.encoder = ModifiedDenseNet121(num_classes=4)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.reducer = nn.Sequential(*[
            nn.Conv2d(1024, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ])

        self.reducer = nn.Sequential(*[
            nn.ReLU(),
            nn.MaxPool2d(2)
        ])


        self.classifier = nn.Linear(1024, 1)

        self.criterion = BCELoss()

    def wandb_log(self, **kwargs):
        for k,v in kwargs.items():
            self.log(k, v)


    def forward(self, x):
        x = torch.cat([x,x,x], dim=1)
        with torch.no_grad():
            feats = self.encoder(x)
        #print(feats.shape)              # [1, 1024, 128, 93]
        x = self.reducer(feats)
        #print(x.shape)                  # [1, 128, 14, 9]
        out = F.adaptive_avg_pool2d(x, (1, 1)).view(feats.size(0), -1)
        #print(out.shape)                # [1, 128]
        out = torch.flatten(out, 1)
        pred = self.classifier(out).squeeze(1)
        pred = torch.sigmoid(pred)

        return pred


    def training_step(self, batch, batch_index):
        imgs, labels, rows = batch
        preds = self.forward(imgs)
        cancerLoss = self.criterion(preds, labels.float())

        self.wandb_log(loss=cancerLoss)
        return dict(loss=cancerLoss)

    def validation_step(self, batch, batch_index):
        imgs, cancer, rows = batch
        preds = self.forward(imgs)
        cancerLoss = self.criterion(preds, cancer.float())

        self.wandb_log(val_loss=cancerLoss)
        return dict(val_loss=cancerLoss, pred=preds, cancer=cancer, rows=rows)

    def validation_epoch_end(self, outputs):
        preds = torch.cat([item['pred'] for item in outputs])
        cancer = torch.cat([item['cancer'] for item in outputs])
        score = pfbeta(cancer, preds)


        df = pd.concat([item['rows'] for item in outputs])
        df['preds'] = preds.detach().cpu().numpy()
        dfscore = pfbeta(df['cancer'], df['preds'])

        maxPredsPerLat = df[['patient_id','laterality','preds']].groupby(['patient_id', 'laterality']).max().reset_index()
        #del df['preds']
        for rn, row in maxPredsPerLat.iterrows():
            #df[(df.patient_id==row.patient_id) & (df.laterality==row.laterality)]['preds'] = row['preds']
            df.loc[(df.patient_id==row.patient_id) & (df.laterality==row.laterality),'preds'] = row['preds']

        dfscoreF = pfbeta(df['cancer'], df['preds'])
        df['preds'] = 1.0*(df.preds > 0.5)
        dfscoreFT = pfbeta(df['cancer'], df['preds'])

        self.wandb_log(dfscoreFT=dfscoreFT, dfscoreF=dfscoreF, dfscore=dfscore, score=score)


    def configure_optimizers(self):
        params = list(self.reducer.parameters()) + list(self.classifier.parameters())
        if self.args.opt == 'sgd':
            optim = torch.optim.SGD(params, lr=self.args.lr)
        elif self.args.opt == 'adam':
            optim = torch.optim.Adam(params, lr=self.args.lr)
        else:
            raise ValueError
        return optim




def collate(batch):
    imgs = [item[0] for item in batch]
    lbls = np.asarray([item[1] for item in batch])
    rows = pd.DataFrame([item[2] for item in batch])

    imgs = torch.stack(imgs)
    lbls = torch.from_numpy(lbls).float()

    return imgs, lbls, rows


if __name__=='__main__':

    trainDF, trainSampler, valDF = getData(sampler=4)
    trainData = FullDataset(trainDF)
    valData = FullDataset(valDF)
    trainLoader = DataLoader(trainData, batch_size=1, sampler=trainSampler, collate_fn=collate, num_workers=4, persistent_workers=True)
    valLoader = DataLoader(valData, batch_size=1, shuffle=True, collate_fn=collate, num_workers=4, persistent_workers=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("-gpu", default='14')
    parser.add_argument("--opt", default='sgd')
    parser.add_argument("-notes", default=None)
    args = parser.parse_args()


    wandb_logger = WandbLogger(project='nyu_end2end', notes=args.notes)
    del args.notes
    wandb_logger.log_hyperparams(args)


    model = FullModel(args).to(dev)

    #for x in trange(16):
    #    model(torch.zeros(1, 1, 4096, 2048).to(dev))


    dev = args.gpu
    if ',' not in dev: dev = [int(dev)]

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/fast/rsna-breast/checkpoints/nyu_end2end/{wandb_logger.experiment.name}/",
        save_top_k=2, monitor="val_loss", mode='min', train_time_interval=timedelta(minutes=10)
    )

    trainer = pl.Trainer(max_epochs=1000, callbacks=[checkpoint_callback],
                         logger=wandb_logger,
                         accelerator='gpu', devices=dev,
                         #val_check_interval=0.1,
                         #val_check_interval=1,
                         limit_train_batches=32,
                         limit_val_batches=32,
                         num_sanity_val_steps=1,
                         accumulate_grad_batches=16,
                         log_every_n_steps=4
                         )

    #if args.checkpoint:
    #    model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu'))['state_dict'])


    trainer.fit(model=model, train_dataloaders=trainLoader, val_dataloaders=valLoader)
    #trainer.fit(model=model, train_dataloaders=[trainDataloader])

