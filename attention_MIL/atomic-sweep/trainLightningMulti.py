import argparse
import os
from datetime import timedelta
from glob import glob

import pandas as pd
import tables
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import IterableDataset, WeightedRandomSampler

from datasets import EmbeddingDataset, collate, EmbeddingDatasetH5, getTensorFileDataset, getH5Dataset
from misc import getPtImgIDs
from sklearn.model_selection import GroupShuffleSplit
import pytorch_lightning as pl
#from torchsample.samplers import StratifiedSampler


# needed for the caching hack implemented in the dataset ....
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", default=0, type=int)
    parser.add_argument("--bs", default=64, type=int)
    parser.add_argument("--encoder", default='efficientnet_b3')
    #parser.add_argument("-quarter", action="store_true")
    #parser.add_argument("-tenth", action="store_true")
    #parser.add_argument("-sampler", default='half', choices=['half', 'quarter', 'tenth'])
    parser.add_argument("--sampler", default=8.0, type=float)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--reducedDim", default=16, type=int)
    parser.add_argument("--hiddenAttn", default=0, type=int)
    parser.add_argument("--opt", default='sgd')
    parser.add_argument("-wandb", default='mamm-mil')
    parser.add_argument("-quant", action='store_true')
    #parser.add_argument("-l1norm", action='store_true')
    args = parser.parse_args()

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
        trainDF, valDF = getTensorFileDataset(args.encoder)
        trainDataset = EmbeddingDataset(trainDF, args.encoder)
        valDataset = EmbeddingDataset(valDF, args.encoder)

    print('train/val split : ', trainDF.shape, valDF.shape)

    emb, lbl = trainDataset[0]
    embeddingSize = emb.shape[1]
    print(f'Embedding size is {embeddingSize}')
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
                                                  shuffle=False, drop_last=False, num_workers=4, collate_fn=collate, sampler=sampler)

    valDataloader = torch.utils.data.DataLoader(valDataset, batch_size=64, persistent_workers=True,
                                                shuffle=False, drop_last=False, num_workers=4, collate_fn=collate)


    run = wandb.init(project=args.wandb)
    conf = run.config
    conf.encoder = args.encoder
    del args.wandb
    wandb.config.update(args)
    #print(run.name)
    #run = None

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/fast/rsna-breast/checkpoints/classifier/{args.encoder}/{run.name}/",
        save_top_k=2, monitor="train_loss", train_time_interval=timedelta(minutes=10)
    )

    dev = args.gpu if torch.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback],
                         accelerator='gpu',
                         #devices=[dev],
                         devices=[5,6,7,8],
                         #devices=args.gpu,
                         #val_check_interval=0.1,
                         num_sanity_val_steps=0,
                         #accumulate_grad_batches=16
    )



    model = AttentionMIL_Lightning(nInput=embeddingSize, wandbRun=run, opt=args.opt, lr=args.lr,
            nReduced=args.reducedDim, nHiddenAttn=args.hiddenAttn
    )


    trainer.fit(model=model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)
    #trainer.fit(model=model, train_dataloaders=trainDataloader)




