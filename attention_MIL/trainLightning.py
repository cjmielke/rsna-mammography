import argparse
import os
import random
from datetime import timedelta
from glob import glob
from random import choice

import pandas as pd
import tables
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import IterableDataset, WeightedRandomSampler

from datasets import EmbeddingDataset, collate, EmbeddingDatasetH5, getTensorFileDataset, getH5Dataset
from misc import getPtImgIDs
from sklearn.model_selection import GroupShuffleSplit
import pytorch_lightning as pl
#from torchsample.samplers import StratifiedSampler


# needed for the caching hack implemented in the dataset ....
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", default='0')
    parser.add_argument("--bs", default=64, type=int)
    parser.add_argument("-maxEpochs", default=200, type=int)
    parser.add_argument("--encoder", default='deit3_small_patch16_224')
    #parser.add_argument("--encoder", default='xcit_polished_leaf')
    #parser.add_argument("--encoder", default='deit_sweepy_sweep')
    # parser.add_argument("-quarter", action="store_true")
    # parser.add_argument("-tenth", action="store_true")
    # parser.add_argument("-sampler", default='half', choices=['half', 'quarter', 'tenth'])
    parser.add_argument("--sampler", default=8, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--reducedDim", default=128, type=int)
    parser.add_argument("--hiddenAttn", default=0, type=int)
    parser.add_argument("--opt", default='sgd')
    parser.add_argument("-wandb", default='mil-super')
    parser.add_argument("--lossFun", default='BCE_F')
    parser.add_argument("-quant", action='store_true')
    # parser.add_argument("--gated", action='store_true')        # ugh
    parser.add_argument("--attn", default='nongated', type=str)
    parser.add_argument("--noise", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--weightDecay", type=float, default=0.0)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--focalAlpha", type=float, default=0.75)
    parser.add_argument("--focalGamma", type=float, default=2.0)
    parser.add_argument("--focalReduction", type=str, default='mean')
    parser.add_argument("-hacked", action='store_true')
    parser.add_argument("--normTensors", action='store_true')
    parser.add_argument("--classifier", default='orig')

    parser.add_argument("-checkpoint", default=None)
    #parser.add_argument("-checkpoint", default='chk/deit_sweepy_sweep/cheerful-orchid-87/epoch=28-step=38729.ckpt')
    #parser.add_argument("-checkpoint", default='/fast/rsna-breast/checkpoints/classifier/deit_brilliant_laughter/thriving-rabbit-400/epoch=49-step=67162.ckpt')
    #parser.add_argument("-checkpoint", default='/fast/rsna-breast/checkpoints/classifier/deit_sweepysweep_new/sweet-laughter-413/epoch=43-step=29710.ckpt')
    #parser.add_argument("-checkpoint", default='/fast/rsna-breast/checkpoints/classifier/vit_tiny_smooth_frost/stellar-night-207/epoch=81-step=55372.ckpt')



    parser.add_argument("--batchNorm", default=1, type=int)
    #parser.add_argument("--tensorDrop", default=0.15, type=float)
    parser.add_argument("--tensorDrop", default=0.15, type=float)
    parser.add_argument("--directClassification", default=0, type=int)
    parser.add_argument("-notes", default=None, type=str)
    parser.add_argument("--model", default="super", type=str)
    parser.add_argument("-metadata", default=1, type=int)
    parser.add_argument("--deep", default=None, type=int)
    parser.add_argument("--predThresh", default=0.5, type=float)
    #parser.add_argument("--predThresh", default=random.choice([.1,.2,.3,.4,.5,.6,.7,.8,.9]), type=float)

    #parser.add_argument("--tileSize", default=224, type=int)
    parser.add_argument("--tileSize", default='224', type=str)

    parser.add_argument("--exclude123", default=1, type=int, help='exclude the first set of 123 easily-classified cancer images')
    parser.add_argument("--exclude_stage2", default=1, type=int)

    return parser

if __name__ == '__main__':

    parser = getParser()

    #parser.add_argument("-l1norm", action='store_true')
    args = parser.parse_args()

    if args.model == 'super':
        args.metadata = 1
        #args.deep = random.choice([0,1])

    #if args.metadata:
    #    assert args.model=='super'

    if args.lossFun.lower() != 'focal':
        args.focalAlpha, args.focalGamma, args.focalReduction = None, None, None

    ##args.gated = True           # FIXME
    #if args.gated.lower() == 'true': args.gated = True
    #elif args.gated.lower() == 'false': args.gated = False

    # might not exist, but if it does we'll have a faster dataloader
    complib = 'blosc:lz4'

    #if args.quant and os.path.exists(h5file):
    if args.quant:
        h5file = f'/fast/rsna-breast/featuresH5/{args.encoder}_quant.h5'
        H5 = tables.open_file(h5file, mode='r')
        labelDF, trainDF, valDF = getH5Dataset(h5file)
        totalTiles = labelDF.nTiles.sum()
        print(f'sum of tilecounts in csv : {totalTiles} |   Length of CArray in H5 : {len(H5.root.tensors)}')
        assert totalTiles==len(H5.root.tensors)
        trainDataset = EmbeddingDatasetH5(trainDF, H5)
        valDataset = EmbeddingDatasetH5(valDF, H5)

    else:
        trainDF, valDF = getTensorFileDataset(args)
        trainDataset = EmbeddingDataset(trainDF, args, noise=args.noise, tensorDrop=args.tensorDrop)
        valDataset = EmbeddingDataset(valDF, args)

    print('train/val split : ', trainDF.shape, valDF.shape)

    B = trainDataset[0]
    embeddingSize = B[0].shape[1]
    print(f'Embedding size is {embeddingSize}')

    if args.model == 'super':
        from superModel import AttentionMIL_Lightning
    elif args.model == 'multi':
        from multiModel import AttentionMIL_Lightning
    else:
        from models import AttentionMIL_Lightning

    #sampler = StratifiedBatchSampler(trainDF.cancer, batchSize)
    #weights = make_weights_for_balanced_classes(trainDataset, 2)
    #weights = torch.DoubleTensor(weights)
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    labels = torch.from_numpy(trainDataset.labelDF.cancer.to_numpy())
    #sampler = StratifiedSampler(labels, batch_size=batchSize)
    #class_count = [i for i in get_class_distribution(trainDataset).values()]

    '''
    if args.sampler == 'quarter':            # try making the class balance something other than 50% and compare performance
        class_count[1] = 2*class_count[1]
    elif args.sampler == 'tenth':
        class_count[1] = 10*class_count[1]
    '''

    if args.sampler == 0:
        sampler = None
    else:
        class_count = list(trainDF.cancer.value_counts())
        class_count[1] = args.sampler * class_count[1]      # overweight the positive cases. If sampler=1.0, then its trained 50/50
        print(class_count)
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        print(class_weights)
        class_weights_all = class_weights[labels]
        print(class_weights_all)
        sampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all), replacement=True)


    trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=args.bs, persistent_workers=True,
                                                  shuffle=False, drop_last=True, num_workers=4, collate_fn=collate, sampler=sampler)

    valDataloader = torch.utils.data.DataLoader(valDataset, batch_size=64, persistent_workers=True,
                                                shuffle=False, drop_last=False, num_workers=4, collate_fn=collate)

    if args.hacked:
        args.wandb = 'mil-hacked'


    if args.wandb:
        run = wandb.init(project=args.wandb, notes=args.notes)
        conf = run.config
        conf.encoder = args.encoder
        del args.wandb
        wandb.config.update(args)
        #print(run.name)
        name = run.name
    else: run, name = None, None
    #run = None

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/fast/rsna-breast/checkpoints/classifier/{args.encoder}/{name}/",
        save_top_k=2, monitor="val_loss", train_time_interval=timedelta(minutes=10)
    )
    earlyStop = EarlyStopping(monitor="val_loss", mode="min", patience=20)

    #dev = args.gpu if torch.cuda.is_available() else 0
    dev = args.gpu
    if ',' not in dev: dev = [int(dev)]

    trainer = pl.Trainer(max_epochs=args.maxEpochs,
                         callbacks=[checkpoint_callback, earlyStop],
                         accelerator='gpu',
                         devices=dev,
                         #devices=args.gpu,
                         #val_check_interval=0.1,
                         num_sanity_val_steps=0,
                         #accumulate_grad_batches=16
    )

    if args.hacked:
        from modelsHacked import AttentionMIL_Hacked
        model = AttentionMIL_Hacked(nInput=embeddingSize, wandbRun=run, opt=args.opt, lr=args.lr,
                nReduced=args.reducedDim, nHiddenAttn=args.hiddenAttn, lossFun=args.lossFun,
                gated=args.gated, dropout=args.dropout, weightDecay=args.weightDecay, l1_lambda=args.l1,
                focalAlpha=args.focalAlpha, focalGamma=args.focalGamma, focalReduction=args.focalReduction
        )

    else:
        model = AttentionMIL_Lightning(nInput=embeddingSize, wandbRun=run, opt=args.opt, lr=args.lr,
                nReduced=args.reducedDim, nHiddenAttn=args.hiddenAttn, lossFun=args.lossFun,
                dropout=args.dropout, weightDecay=args.weightDecay, l1_lambda=args.l1,
                focalAlpha=args.focalAlpha, focalGamma=args.focalGamma, focalReduction=args.focalReduction,
                classifier=args.classifier, argparse=args
        )

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu'))['state_dict'])

    trainer.fit(model=model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)
    #trainer.fit(model=model, train_dataloaders=trainDataloader)




