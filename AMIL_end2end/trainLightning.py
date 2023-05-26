import argparse
import os
from datetime import timedelta
from glob import glob

import pandas as pd
import tables
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import IterableDataset, WeightedRandomSampler

from datasets import getDatasets, GenericCollate
import pytorch_lightning as pl
#from torchsample.samplers import StratifiedSampler

# needed for the caching hack implemented in the dataset ....
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from models import EndToEnd_AMIL

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", default='0', type=str)
    parser.add_argument("--bs", default=4, type=int)
    parser.add_argument("-maxEpochs", default=200, type=int)
    parser.add_argument("--backbone", default='xcit_nano_12_p16_224_dist')
    #parser.add_argument("--backbone", default='mobilenetv3_small_075')
    #parser.add_argument("-quarter", action="store_true")
    #parser.add_argument("-tenth", action="store_true")
    #parser.add_argument("-sampler", default='half', choices=['half', 'quarter', 'tenth'])
    parser.add_argument("--sampler", default=16.0, type=float)
    parser.add_argument("--lr", default=0.001, type=float)            # FIXME
    parser.add_argument("--reducedDim", default=64, type=int)
    parser.add_argument("--hiddenAttn", default=0, type=int)
    parser.add_argument("--opt", default='sgd')
    parser.add_argument("-wandb", default='mamm-end2end-amil')
    #parser.add_argument("--lossFun", default='focal')
    #parser.add_argument("--lossFun", default='BCE_F')
    parser.add_argument("--lossFun", default='BCE')
    #parser.add_argument("--gated", action='store_true')        # ugh
    parser.add_argument("--gated", default=False)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--weightDecay", type=float, default=0.0)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--focalAlpha", type=float, default=0.95)
    parser.add_argument("--focalGamma", type=float, default=2.0)
    parser.add_argument("--focalReduction", type=str, default='sum')
    parser.add_argument("--tileLimit", type=int, default=8)
    parser.add_argument("--colorize", type=int, default=2)
    parser.add_argument("--model", default='MAX')
    #parser.add_argument("-l1norm", action='store_true')
    args = parser.parse_args()

    if args.lossFun.lower() != 'focal':
        args.focalAlpha, args.focalGamma, args.focalReduction = None, None, None

    #args.gated = True           # FIXME
    #if args.gated.lower() == 'true': args.gated = True
    #elif args.gated.lower() == 'false': args.gated = False

    trainDataset, valDataset = getDatasets(args)

    print('train/val split : ', trainDataset.imagesDF.shape, valDataset.imagesDF.shape)

    tiles, lbl = trainDataset[0]
    print(tiles)
    print(lbl)


    labels = torch.from_numpy(trainDataset.imagesDF.cancer.to_numpy())

    if args.sampler == 0:
        raise ValueError('Dont')
        sampler = None
    else:
        class_count = list(trainDataset.imagesDF.cancer.value_counts())
        class_count[1] = args.sampler * class_count[1]      # overweight the positive cases. If sampler=1.0, then its trained 50/50
        print(class_count)
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        print(class_weights)
        class_weights_all = class_weights[labels]
        print(class_weights_all)
        sampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all), replacement=True)

    trainCollate = GenericCollate(training=True)
    valCollate = GenericCollate(training=False)

    trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=args.bs, persistent_workers=True,
                                    shuffle=False, drop_last=False, num_workers=2, collate_fn=trainCollate, sampler=sampler)

    valDataloader = torch.utils.data.DataLoader(valDataset, batch_size=1, persistent_workers=True,
                                    shuffle=True, drop_last=False, num_workers=2, collate_fn=valCollate)

    '''
    if args.wandb:
        run = wandb.init(project=args.wandb)
        conf = run.config
        conf.backbone = args.backbone
        del args.wandb
        wandb.config.update(args)
        #print(run.name)
        name = run.name
    else:
        run, name = None, None
    '''

    wandb_logger = WandbLogger(project=args.wandb)
    del args.wandb
    wandb_logger.log_hyperparams(args)
    name = wandb_logger.experiment.name
    run=None


    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/fast/rsna-breast/checkpoints/AMIL_end2end/{args.backbone}/{name}/",
        save_top_k=2, monitor="score", mode='max', train_time_interval=timedelta(minutes=10)
    )

    dev = args.gpu if torch.cuda.is_available() else 0
    if ',' not in dev: dev=[int(dev)]
    else: dev = [int(g) for g in dev.split(',')]

    trainer = pl.Trainer(max_epochs=args.maxEpochs, callbacks=[checkpoint_callback],
                         logger=wandb_logger, log_every_n_steps=10,
                         accelerator='gpu',
                         #devices=[dev],
                         devices=dev,
                         #val_check_interval=0.1,
                         #val_check_interval=10,
                         #limit_train_batches=1024,
                         limit_train_batches=64,
                         #limit_val_batches=256,         # 11k validation images, so this is 1/43 of the validation set
                         #limit_val_batches=512,
                         num_sanity_val_steps=1,
                         accumulate_grad_batches=16,
                         #accumulate_grad_batches=128,
                         #check_val_every_n_epoch=512,
    )

    model = EndToEnd_AMIL(args, model=args.model, backbone=args.backbone, wandbRun=run, opt=args.opt, lr=args.lr,
            nReduced=args.reducedDim, nHiddenAttn=args.hiddenAttn, lossFun=args.lossFun,
            gated=args.gated, dropout=args.dropout, weightDecay=args.weightDecay, l1_lambda=args.l1,
            focalAlpha=args.focalAlpha, focalGamma=args.focalGamma, focalReduction=args.focalReduction
    )

    #chkpts = '/fast/rsna-breast/checkpoints/tileClassifier'
    #chkpt = f'{chkpts}/deit3_small_patch16_224_sweepy-sweep-45/epoch=83-step=21270.ckpt'

    chkpt = '/fast/rsna-breast/checkpoints/tileClassifier/xcit_nano_12_p16_224_dist_polished-leaf-74/epoch=17-step=10549.ckpt'

    stateDict = torch.load(chkpt)['state_dict']
    #print(stateDict.keys())
    #model.encoder.load_state_dict(stateDict, strict=False)

    trainer.fit(model=model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)
    #trainer.fit(model=model, train_dataloaders=trainDataloader)



