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

from lightly.data import SwaVCollateFunction
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')






class SwaV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        if args.encoder == 'resnet18':
            resnet = torchvision.models.resnet18()
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            embSize = 512
        else:
            self.backbone, embSize = getTimmModel(args.encoder)

        #self.projection_head = SwaVProjectionHead(512, 512, 128)
        self.projection_head = SwaVProjectionHead(embSize, embSize, embSize//4)
        #self.prototypes = SwaVPrototypes(embSize//4, n_prototypes=embSize//2)
        self.prototypes = SwaVPrototypes(embSize // 4, n_prototypes=128)
        #self.projection_head = SwaVProjectionHead(embSize, 512, 128)
        #self.prototypes = SwaVPrototypes(128, n_prototypes=32)
        self.criterion = SwaVLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def training_step(self, batch, batch_idx):
        self.prototypes.normalize()
        crops, _, _ = batch
        multi_crop_features = [self.forward(x.to(self.device)) for x in crops]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        #optim = torch.optim.Adam(self.parameters(), lr=0.01)
        optim = torch.optim.SGD(self.parameters(), lr=0.01)

        return optim




if __name__ == '__main__':

    args = parser.parse_args()
    #IMG_SIZE = args.imgSize
    gpus = args.gpu if torch.cuda.is_available() else 0

    model = SwaV(args)
    #model.load_state_dict(torch.load('/fast/rsna-breast/checkpoints/swav/tf_mobilenetv3_small_minimal_100_lucky-rooster-85/epoch=0-step=3152.ckpt')['state_dict'])
    dataset = TileLightlyDataset()

    sslAlgo = os.path.split(__file__)[1].rstrip('.py')
    wandb_logger = WandbLogger(project='mamm-ssl')
    wandb_logger.log_hyperparams(args)
    wandb_logger.log_hyperparams(dict(ssl=sslAlgo))


    collate_fn = SwaVCollateFunction(crop_sizes=[224], crop_counts=[4],
                                     crop_min_scales=[0.8],
                                     crop_max_scales=[1.2],
                                     rr_degrees=90.0
                                     )

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
        save_top_k=2, monitor="train_loss", train_time_interval=timedelta(minutes=10)
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


