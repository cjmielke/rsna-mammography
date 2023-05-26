import sys
from datetime import timedelta

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import getDataloaders
from argparse import ArgumentParser
import pytorch_lightning as pl
from models import Model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-gpu", default='0')
    parser.add_argument("--dataset", default='heatmaps')
    parser.add_argument("-wandb", default="mamm-wholeclassifier")
    parser.add_argument("--encoder", default='stupid')
    parser.add_argument("--opt", default='sgd')
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--colorizer", default=0, type=int)
    parser.add_argument("-notes", default=None)
    parser.add_argument("--sampler", default=4, type=float)
    args = parser.parse_args()

    #if args.encoder != 'tf_mobilenetv3_small_minimal_100':
    #    sys.exit()

    trainLoader, valLoader = getDataloaders(args)


    model = Model(args)

    '''
    for batch in trainLoader:
        imgs, lbls = batch
        print(imgs.shape, lbls.shape)
        preds = model(imgs)
        print(preds)

    '''

    wandb_logger = WandbLogger(project=args.wandb, notes=args.notes)
    del args.notes
    del args.wandb
    wandb_logger.log_hyperparams(args)

    dev = args.gpu
    if ',' not in dev: dev = [int(dev)]



    #model.load_state_dict(torch.load('/fast/rsna-breast/checkpoints/wholeClassifier/vivid-sweep-66/epoch=40-step=41664.ckpt')['state_dict'])
    #print('loaded!')

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/fast/rsna-breast/checkpoints/wholeClassifier/{wandb_logger.experiment.name}/",
        save_top_k=2, monitor="train_loss", train_time_interval=timedelta(minutes=10)
    )
    trainer = pl.Trainer(max_epochs=300, callbacks=[checkpoint_callback],
                         logger=wandb_logger,
                         accelerator='gpu', devices=dev,
                         #val_check_interval=0.1,
                         #val_check_interval=1,
                         limit_train_batches=128*8,
                         limit_val_batches=128*8,
                         num_sanity_val_steps=0,
                         #accumulate_grad_batches=8
                         )
    trainer.fit(model=model, train_dataloaders=trainLoader, val_dataloaders=valLoader)
    #trainer.fit(model=model, train_dataloaders=[trainDataloader])









