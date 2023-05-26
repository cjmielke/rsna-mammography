

import argparse
import os
from datetime import timedelta
from glob import glob

import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from dataloader import EmbeddingDataset, getDatasets, collate
from models import Model, getEmbeddingSizeForEncoder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", default='0')
    parser.add_argument("--bs", default=1, type=int)
    #parser.add_argument("--encoder", default='deit3_small_patch16_224')
    #parser.add_argument("--encoder", default='deit_sweepysweep_new')
    parser.add_argument("--encoder", default='nyu')
    parser.add_argument("-wandb", default='mamm-latent-autoencoder')
    parser.add_argument("-noise", default=0.00, type=float)
    parser.add_argument("-hidden", default=32, type=int)
    #parser.add_argument("-out", default='autoencoded')
    #parser.add_argument("-l1norm", action='store_true')
    args = parser.parse_args()

    trainDF, valDF = getDatasets()

    #print(trainDF.shape, valDF.shape)
    #c

    embSize = getEmbeddingSizeForEncoder(args.encoder)
    model = Model(embSize=embSize, wandb=None, nHidden=args.hidden)


    # sampler for the training set
    class_weights = trainDF.weight.to_numpy()
    sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(class_weights), replacement=True)
    #sampler = None


    # sampler for the validation set
    #Vclass_weights = valDF.weight.to_numpy()
    #valSampler = WeightedRandomSampler(weights=Vclass_weights, num_samples=len(Vclass_weights), replacement=True)
    valSampler = None

    trainDataset = EmbeddingDataset(trainDF, args.encoder, noise=args.noise)
    valDataset = EmbeddingDataset(valDF, args.encoder)

    #collate_fn = CollateFunction().to

    trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=args.bs, persistent_workers=True,
                                                  shuffle=False, drop_last=False, num_workers=2, collate_fn=collate, sampler=sampler)

    valDataloader = torch.utils.data.DataLoader(valDataset, batch_size=args.bs, persistent_workers=True,
                                                shuffle=False, drop_last=False, num_workers=2, collate_fn=collate, sampler=valSampler)


    #GPUS = args.gpu                    # meh, do this later .... wandb dashboard should just report GPU count, not list
    #args.gpu = len(args.gpu.spl)

    wandb_logger = WandbLogger(project=args.wandb)
    wandb_logger.log_hyperparams(args)
    name = wandb_logger.experiment.name

    #wandb_logger = None
    #run, name = None, 'foo'



    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/fast/rsna-breast/checkpoints/latentAutoencoder/{args.encoder}_{name}/",
        save_top_k=2, monitor="train_loss", train_time_interval=timedelta(minutes=10)
    )
    trainer = pl.Trainer(max_epochs=3, callbacks=[checkpoint_callback],
                         #logger=wandb_logger,
                         accelerator='gpu', gpus=args.gpu,
                         #val_check_interval=0.1,
                         #val_check_interval=1,
                         #limit_train_batches=32*8,
                         #limit_val_batches=4*8,
                         #num_sanity_val_steps=0,
                         accumulate_grad_batches=64
                         )

    try:
        #trainer.fit(model=model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)
        trainer.fit(model=model, train_dataloaders=trainDataloader)
    except KeyboardInterrupt:
        pass


    # now produce outputs
    print('Saving tensors')
    for tensorF in tqdm(glob(f'/fast/rsna-breast/features/224/{args.encoder}/*/*.pt')):
        tensor = torch.load(tensorF)
        hidden = model.getFeatures(tensor)
        #print(f'hidden {hidden.shape}')

        p, fn = os.path.split(tensorF)
        outDir = p.replace(args.encoder, f'{args.encoder}_autoenc{args.hidden}')
        #print(outDir)
        #continue
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        outFile = os.path.join(outDir, fn)
        #print(outFile)
        #continue
        torch.save(hidden, outFile)








