import os
from datetime import timedelta

import timm
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.nn import Conv2d
from misc import parser, getTimmModel
from dataloader import TileLightlyDataset


import torch
from torch import nn
import torchvision
import copy
import pytorch_lightning as pl

from lightly.data import LightlyDataset
from lightly.data import MoCoCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class MoCo(pl.LightningModule):
    def __init__(self):
        super().__init__()
        if args.encoder == 'resnet18':
            resnet = torchvision.models.resnet18()
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            embSize = 512
        else:
            self.backbone, embSize = getTimmModel(args.encoder)

        #self.projection_head = MoCoProjectionHead(512, 512, 128)
        self.projection_head = MoCoProjectionHead(embSize, embSize, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(memory_bank_size=4096)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def training_step(self, batch, batch_idx):
        update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)
        (x_query, x_key), _, _ = batch
        query = self.forward(x_query)
        key = self.forward_momentum(x_key)
        loss = self.criterion(query, key)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim



if __name__ == '__main__':

    args = parser.parse_args()
    IMG_SIZE = args.imgSize
    gpus = args.gpu if torch.cuda.is_available() else 0

    model = MoCo()
    dataset = TileLightlyDataset()

    sslAlgo = os.path.split(__file__)[1].rstrip('.py')
    wandb_logger = WandbLogger(project='mamm-ssl')
    wandb_logger.log_hyperparams(args)
    wandb_logger.log_hyperparams(dict(ssl=sslAlgo))


    collate_fn = MoCoCollateFunction(input_size=IMG_SIZE)           # was 32, for cifar

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=dataset.sampler,
        batch_size=args.bs,             # 256!
        collate_fn=collate_fn,
        #shuffle=True,
        shuffle=False,                  # must disable for the sampler to work
        drop_last=True,
        num_workers=2,
    )

    checkpoint_callback = ModelCheckpoint(
        #dirpath=f"/fast/rsna-breast/checkpoints/{sslAlgo}/{args.encoder}/",
        dirpath=f"/fast/rsna-breast/checkpoints/{sslAlgo}/{args.encoder}_{wandb_logger.experiment.name}/",
        save_top_k=2, monitor="train_loss", train_time_interval=timedelta(minutes=30)
    )

    dev = args.gpu
    if ',' not in dev: dev = [int(dev)]

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=100,
        callbacks=[checkpoint_callback],
        accelerator='gpu',
        devices=dev,
        #gpus=[4,5,6,7],
        #gpus=[args.gpu],
        num_sanity_val_steps=0
    )

    trainer.fit(model=model, train_dataloaders=dataloader)


