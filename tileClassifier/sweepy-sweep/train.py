import argparse
from datetime import timedelta

import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import WeightedRandomSampler
from data import getTablesV2, getTablesAttn, TileDataset, CollateFunction, getNewAttnsWithRaws
from models import Model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", default='0')
    parser.add_argument("--bs", default=8, type=int)
    parser.add_argument("--encoder", default='deit3_small_patch16_224')
    parser.add_argument("--attnThresh", default=0.8, type=float, help='minimum attention score from MIL model to consider a tile cancerous')
    parser.add_argument("--sigma", default=2.0, help="number of sigma raw attention score to threshold")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--opt", default='sgd')

    #parser.add_argument("--sampler", default=8.0, type=float)
    #parser.add_argument("--lr", default=0.01, type=float)
    #parser.add_argument("--reducedDim", default=64, type=int)
    #parser.add_argument("--hiddenAttn", default=0, type=int)
    #parser.add_argument("--opt", default='sgd')
    parser.add_argument("-wandb", default='mamm-tile-classifier')
    parser.add_argument("-target", default='attentionWithRaw')
    #parser.add_argument("-l1norm", action='store_true')
    parser.add_argument("--colorize", default=8, type=int)
    parser.add_argument("--pool", default=1, type=int)
    parser.add_argument("--kernel", default=3, type=int)
    parser.add_argument("-notes", default=None)
    args = parser.parse_args()

    if args.target == 'cancerv2':
        trainDF, valDF = getTablesV2(args)
    elif args.target == 'attention':
        trainDF, valDF = getTablesAttn(args)
    elif args.target == 'attentionWithRaw':
        trainDF, valDF = getNewAttnsWithRaws(rawSigma=args.sigma)
    else: raise ValueError

    # sampler for the training set
    class_weights = trainDF.weight.to_numpy()
    sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(class_weights), replacement=True)
    #sampler = None


    # sampler for the validation set
    Vclass_weights = valDF.weight.to_numpy()
    valSampler = WeightedRandomSampler(weights=Vclass_weights, num_samples=len(Vclass_weights), replacement=True)



    trainDataset = TileDataset(trainDF, args)
    valDataset = TileDataset(valDF, args)

    #collate_fn = CollateFunction().to


    trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=args.bs, persistent_workers=True,
                                                  shuffle=False, drop_last=False, num_workers=2, collate_fn=CollateFunction(args), sampler=sampler)

    valDataloader = torch.utils.data.DataLoader(valDataset, batch_size=32, persistent_workers=True,
                                                shuffle=False, drop_last=False, num_workers=2, collate_fn=CollateFunction(args), sampler=valSampler)

    '''
    run = wandb.init(project=args.wandb)
    conf = run.config
    conf.encoder = args.encoder
    del args.wandb
    wandb.config.update(args)
    name = run.name
    #print(run.name)
    '''

    #GPUS = args.gpu                    # meh, do this later .... wandb dashboard should just report GPU count, not list
    #args.gpu = len(args.gpu.spl)

    wandb_logger = WandbLogger(project=args.wandb, notes=args.notes)
    del args.notes
    wandb_logger.log_hyperparams(args)

    run, name = None, 'foo'

    model = Model(args)

    dev = args.gpu
    if ',' not in dev: dev = [int(dev)]

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/fast/rsna-breast/checkpoints/tileClassifier/{args.encoder}_{wandb_logger.experiment.name}/",
        save_top_k=2, monitor="train_loss", train_time_interval=timedelta(minutes=10)
    )
    trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback],
                         logger=wandb_logger,
                         accelerator='gpu', devices=dev,
                         #val_check_interval=0.1,
                         #val_check_interval=1,
                         limit_train_batches=32*8,
                         limit_val_batches=4*8,
                         num_sanity_val_steps=0
                         )
    trainer.fit(model=model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)
    #trainer.fit(model=model, train_dataloaders=[trainDataloader])







