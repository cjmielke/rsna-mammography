import time
from glob import iglob

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import MSELoss


def getEmbeddingSizeForEncoder(encoder):
    pat = f'/fast/rsna-breast/features/224/{encoder}/*/*.pt'
    fn = next(iglob(pat))
    tensor = torch.load(fn)

    embeddingSize = tensor.shape[1]
    return embeddingSize


class Model(LightningModule):

    def __init__(self, embSize=1024, nHidden=128, wandb=None):
        super().__init__()

        self.wandb = wandb
        self.encoder = nn.Linear(embSize, nHidden)
        self.decoder = nn.Linear(nHidden, embSize)
        self.criterion = MSELoss()

    def getFeatures(self, x):
        return self.encoder(x)

    def forward(self, x):
        #print(type(x), len(x))
        h = self.encoder(x)
        xH = self.decoder(h)
        #print(x.shape, h.shape, xH.shape)
        return xH


    def step(self, batch, batch_index):
        #latents, latentsB = batch#[0]
        latents = batch[0]
        #print(latents.shape)
        #print(imgs.shape, labels.shape)
        l = self.forward(latents)
        nLatents = latents.shape[0]
        loss = self.criterion(l, latents) #/ nLatents
        #print(loss)

        #self.wandb_log(train_loss=loss, train_acc=acc)
        #time.sleep(1)
        return dict(loss=loss)

    def training_step(self, batch, batch_index):
        out = self.step(batch, batch_index)
        self.log('train_loss', out['loss'])
        return out

    def validation_step(self, batch, batch_index):
        out = self.step(batch, batch_index)
        self.log('val_loss', out['loss'])
        return out

    def configure_optimizers(self):
        #optim = torch.optim.SGD(self.parameters(), lr=0.001)
        optim = torch.optim.Adam(self.parameters(), lr=0.01)
        return optim

