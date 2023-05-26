# reference implementation from : https://www.kaggle.com/code/heyytanay/train-pytorch-lightning-gpu-tpu-w-b-kfolds


import pandas as pd
import wandb
from torch.utils.data import DataLoader
import pytorch_lightning as pl

#from sklearn.model_selection import StratifiedGroupKFold
from split import StratifiedGroupKFold

import warnings

from config import Config
from dataset import RSNAData
from model import RSNAModel

warnings.simplefilter('ignore')

if __name__ == "__main__":
    run = wandb.init(project='mamm', config=Config, group='vision', job_type='train')
    # run.config.

    # Load the data and pass it onto the training function
    df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/train.csv")
    df['img_name'] = df['patient_id'].astype(str) + "/" + df['image_id'].astype(str) + ".png"
    df = df.sample(frac=1).reset_index(drop=True)
    df.head()

    kfold = StratifiedGroupKFold(n_splits=Config['SPLITS'])
    for fold_, (train_idx, valid_idx) in enumerate(kfold.split(df, df['cancer'].values, df['patient_id'].values)):
        print(f"{'=' * 40} Fold: {fold_} / 5 {'=' * 40}")

        train_df = df.loc[train_idx].reset_index(drop=True)
        valid_df = df.loc[valid_idx].reset_index(drop=True)

        train_dataset = RSNAData(df=train_df, img_folder=Config['PARENT_PATH'], augments=None)
        valid_dataset = RSNAData(df=valid_df, img_folder=Config['PARENT_PATH'], augments=None)
        train_loader = DataLoader(train_dataset, batch_size=Config['TRAIN_BS'], shuffle=True, num_workers=Config['NUM_WORKERS'])
        valid_loader = DataLoader(valid_dataset, batch_size=Config['VALID_BS'], shuffle=False, num_workers=Config['NUM_WORKERS'])

        model = RSNAModel()
        #trainer = pl.Trainer( max_epochs=Config['NB_EPOCHS'], gpus=1, val_check_interval=0.1)
        trainer = pl.Trainer( max_epochs=Config['NB_EPOCHS'], gpus=4)
        trainer.fit(model, train_loader, valid_loader)


    run.finish()












