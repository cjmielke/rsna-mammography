import os
from datetime import timedelta

import timm
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.nn import Conv2d
from misc import parser, getTimmModel
from dataloader import TileLightlyDataset

args = parser.parse_args()
IMG_SIZE = args.imgSize

import torch
gpus = args.gpu if torch.cuda.is_available() else 0
from torch import nn
import torchvision
import pytorch_lightning as pl


from lightly.data import SimCLRCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead


class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        if args.encoder == 'resnet18':
            resnet = torchvision.models.resnet18()
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            embSize = 512
        else:
            self.backbone, embSize = getTimmModel(args.encoder)

        #self.backbone, embSize = getTimmModel(args.encoder)
        #self.projection_head = SimCLRProjectionHead(512, 2048, 2048)
        self.projection_head = SimCLRProjectionHead(embSize, embSize//2, embSize//2)
        # for efficientnet_b3, embSize is 1536 !
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), targets, fns = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim



if __name__ == '__main__':


    model = SimCLR()

    dataset = TileLightlyDataset()

    sslAlgo = os.path.split(__file__)[1].rstrip('.py')
    wandb_logger = WandbLogger(project='mamm-ssl')
    wandb_logger.log_hyperparams(args)
    wandb_logger.log_hyperparams(dict(ssl=sslAlgo))


    collate_fn = SimCLRCollateFunction(input_size=IMG_SIZE, gaussian_blur=0.)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.bs, collate_fn=collate_fn,
        drop_last=True, num_workers=2,
        sampler=dataset.sampler, shuffle=False
    )

    checkpoint_callback = ModelCheckpoint(
        #dirpath=f"/fast/rsna-breast/checkpoints/{sslAlgo}/{args.encoder}/",
        dirpath=f"/fast/rsna-breast/checkpoints/{sslAlgo}/{args.encoder}_{wandb_logger.experiment.name}/",
        save_top_k=2, monitor="train_loss", train_time_interval=timedelta(minutes=30)
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=100,
        callbacks=[checkpoint_callback],
        accelerator='gpu',
        #gpus=[4,5,6,7],
        gpus=[args.gpu],
        num_sanity_val_steps=0
    )

    trainer.fit(model=model, train_dataloaders=dataloader)


