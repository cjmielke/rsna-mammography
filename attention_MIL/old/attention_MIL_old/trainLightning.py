import argparse
from datetime import timedelta
from glob import glob

import pandas as pd
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import IterableDataset, WeightedRandomSampler

from datasets import EmbeddingDataset, collate
from misc import getPtImgIDs
from sklearn.model_selection import GroupShuffleSplit
import pytorch_lightning as pl
#from torchsample.samplers import StratifiedSampler


# needed for the caching hack implemented in the dataset ....
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", default=1, type=int)
    parser.add_argument("-bs", default=64, type=int)
    parser.add_argument("-encoder", default='efficientnet_b3')
    #parser.add_argument("-quarter", action="store_true")
    #parser.add_argument("-tenth", action="store_true")
    parser.add_argument("-sampler", default='half', choices=['half', 'quarter', 'tenth'])
    parser.add_argument("-reducedDim", default=0, type=int)
    parser.add_argument("-hiddenAttn", default=0, type=int)
    #parser.add_argument("-l1norm", action='store_true')
    args = parser.parse_args()



    # gather tensor files and join into dataframe - then split into train and validation
    labelDF = pd.read_csv('/fast/rsna-breast/train.csv')
    tensorFiles = glob(f'/fast/rsna-breast/features/{args.encoder}/*/*.pt')
    rows = []
    for tf in tensorFiles:
        ptID, imgID = getPtImgIDs(tf)
        rows.append(dict(patient_id=ptID, image_id=imgID, tensorFile=tf))
    fileDF = pd.DataFrame(rows)
    labelDF = labelDF.merge(fileDF, on=['patient_id', 'image_id'])
    labelDF = labelDF.sample(frac=1)  # shuffle!

    missing = labelDF[labelDF.tensorFile.isna()]

    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7)
    split = splitter.split(labelDF, groups=labelDF['patient_id'])
    train_inds, val_inds = next(split)

    trainDF = labelDF.iloc[train_inds]
    valDF = labelDF.iloc[val_inds]

    # verify no overlap between patients
    assert len(set(trainDF.patient_id).intersection(set(valDF.patient_id))) == 0




    trainDataset = EmbeddingDataset(trainDF)
    valDataset = EmbeddingDataset(valDF)

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
    class_count = list(trainDF.cancer.value_counts())

    if args.sampler == 'quarter':            # try making the class balance something other than 50% and compare performance
        class_count[1] = 2*class_count[1]
    elif args.sampler == 'tenth':
        class_count[1] = 10*class_count[1]

    print(class_count)
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    print(class_weights)
    class_weights_all = class_weights[labels]
    print(class_weights_all)

    weighted_sampler = WeightedRandomSampler( weights=class_weights_all, num_samples=len(class_weights_all), replacement=True )
    if args.sampler == 'none':
        weighted_sampler = None

    trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=args.bs,
                                                  shuffle=False, drop_last=False, num_workers=1, collate_fn=collate, sampler=weighted_sampler)

    valDataloader = torch.utils.data.DataLoader(valDataset, batch_size=16,
                                                shuffle=False, drop_last=False, num_workers=8, collate_fn=collate)


    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/fast/rsna-breast/checkpoints/classifier/{args.encoder}/",
        save_top_k=2, monitor="train_loss", train_time_interval=timedelta(minutes=10)
    )

    dev = args.gpu if torch.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback],
                         #gpus=gpus,
                         #gpus=3,
                         accelerator='gpu', devices=[dev],
                         #val_check_interval=0.1,
                         num_sanity_val_steps=0,
                         #accumulate_grad_batches=16
    )


    run = wandb.init(project='mamm-mil')
    conf = run.config
    conf.encoder = args.encoder
    wandb.config.update(args)
    #run = None

    model = AttentionMIL_Lightning(nInput=embeddingSize, wandbRun=run,
                                   nReduced=args.reducedDim, nHiddenAttn=args.hiddenAttn
    )


    trainer.fit(model=model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)
    #trainer.fit(model=model, train_dataloaders=trainDataloader)




