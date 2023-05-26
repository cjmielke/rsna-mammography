import os
from datetime import timedelta
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.nn import Conv2d
from misc import parser, wandb_log, getTimmModel
#from dataloader import getDataset
from dataloader import TileLightlyDataset

args = parser.parse_args()
IMG_SIZE = args.imgSize

import torch
gpus = args.gpu if torch.cuda.is_available() else 0
from torch import nn
import torchvision
import pytorch_lightning as pl

from lightly.data import ImageCollateFunction
from lightly.loss import VICRegLoss

## The projection head is the same as the Barlow Twins one
from lightly.models.modules import BarlowTwinsProjectionHead



class VICReg(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        if args.encoder == 'resnet18':
            resnet = torchvision.models.resnet18()
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            embSize = 512
        else:
            self.backbone, embSize = getTimmModel(args.encoder)

        '''
        resnet = torchvision.models.resnet18()
        print(list(resnet.children()))
        layers = list(resnet.children())[:-2]
        layers.append(Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.backbone = nn.Sequential(*layers)
        inp = torch.zeros((2, 3, IMG_SIZE, IMG_SIZE))
        out = self.backbone(inp)
        print(f'Output shape : {out.shape}')
        # was torch.Size([2, 512, 1, 1])
        # now torch.Size([2, 3, 7, 7])
        #sys.exit()
        nBatch, C, H, W = out.shape
        inDim = C*H*W
        '''

        #self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)
        print(f'Projection head mapping {embSize} elements to {embSize*4}')
        #self.projection_head = BarlowTwinsProjectionHead(embSize, embSize*4, embSize*4)
        self.projection_head = BarlowTwinsProjectionHead(embSize, embSize, embSize)
        self.criterion = VICRegLoss()

    def forward(self, x):
        #print(x.min(), x.mean(), x.max())
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        #if batch_index%10==0:
        wandb_log(loss=loss)
        #self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        #optim = torch.optim.SGD(self.parameters(), lr=0.06)
        optim = torch.optim.SGD(self.parameters(), lr=0.01)
        return optim


model = VICReg(args)


dataset_train = TileLightlyDataset(RGB=True)


sslAlgo = os.path.split(__file__)[1].rstrip('.py')
wandb_logger = WandbLogger(project='mamm-ssl')
wandb_logger.log_hyperparams(args)
wandb_logger.log_hyperparams(dict(ssl=sslAlgo))


collate_fn = ImageCollateFunction(
    input_size=IMG_SIZE,
    rr_prob=1.0,
    rr_degrees=90.0,
    min_scale=0.90,
    vf_prob=0.5,
    hf_prob=0.5
)

dataloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.bs, collate_fn=collate_fn,
    shuffle=True, drop_last=True, num_workers=16,
)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"/fast/rsna-breast/checkpoints/{sslAlgo}/resnet18/",
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
    num_sanity_val_steps=0
)

trainer.fit(model=model, train_dataloaders=dataloader)


