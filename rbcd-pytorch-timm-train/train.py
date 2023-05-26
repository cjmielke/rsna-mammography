import multiprocessing as mp
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from timm.data.transforms_factory import create_transform
from timm.loss import BinaryCrossEntropy
from timm.optim import create_optimizer_v2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score





KAGGLE_DIR = Path("/") / "kaggle"

INPUT_DIR = KAGGLE_DIR / "input"

DATA_ROOT_DIR = INPUT_DIR / "rsna-breast-cancer-detection"

TRAIN_IMAGES_DIR = INPUT_DIR / "rsna-mammo-dicomsdl-1024" / "train_images_processed_cv2_dicomsdl_1024"
TRAIN_CSV_PATH = DATA_ROOT_DIR / "train.csv"

ACCELERATOR = "gpu"
BATCH_SIZE = 8
DEVICES = 1
DROP_RATE = 0.3
DROP_PATH_RATE = 0.2
ETA_MIN = 1e-6
FAST_DEV_RUN = False
IMAGE_SIZE = 1024
LEARNING_RATE = 3e-4
MAX_EPOCHS = 5
MODEL_NAME = "tf_efficientnetv2_s"
NUM_SPLITS = 4
NUM_WORKERS = mp.cpu_count()
OVERFIT_BATCHES = 0
OPTIMIZER = "AdamW"
PRECISION = 16
SEED = 42
UPSAMPLE = 10
VAL_FOLD = 0.0
WEIGHT_DECAY = 1e-6


##### Prepare Data

def prepare_data(csv_path, images_dir, create_splits: bool = False):
    df = pd.read_csv(csv_path)

    df["image"] = (
            str(images_dir)
            + "/"
            + df["patient_id"].astype(str)
            + "/"
            + df["image_id"].astype(str)
            + ".png"
    )

    if create_splits:
        skf = StratifiedGroupKFold(n_splits=NUM_SPLITS)
        for fold, (_, val_) in enumerate(
                skf.split(X=df, y=df.cancer, groups=df.patient_id)
        ):
            df.loc[val_, "fold"] = fold

    # Save
    file_path = csv_path.name
    df.to_csv(file_path, index=False)

    print(f"Created {file_path} with {len(df)} rows")

    return df


train_df = prepare_data(TRAIN_CSV_PATH, TRAIN_IMAGES_DIR, create_splits=True)
#Created
#train.csv
#with 54706 rows


###### Dataset

class RBCDDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row.image).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        try:
            return image, row.cancer
        except:
            return image


##### LightningDataModule

class TimmDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            data_csv_path: str,
            num_workers: int,
            upsample: int,
            val_fold: float,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.df = pd.read_csv(data_csv_path)

        self.spatial_size = (IMAGE_SIZE, IMAGE_SIZE)

        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()

    def _init_train_transform(self):
        return create_transform(
            input_size=self.spatial_size,
            is_training=True,
            scale=(0.75, 1.33),
            ratio=(0.08, 1.0),
            hflip=0.5,
            vflip=0.5,
            color_jitter=0.4,
            interpolation="random",
        )

    def _init_val_transform(self):
        return create_transform(
            input_size=self.spatial_size,
            is_training=False,
            interpolation="bilinear",
        )

    def setup(self, stage=None):
        if self.hparams.data_csv_path == "train.csv":
            val_fold = self.hparams.val_fold
            train_df = self.df[self.df.fold != val_fold].reset_index(drop=True)
            val_df = self.df[self.df.fold == val_fold].reset_index(drop=True)

            # Upsample cancer data (from https://www.kaggle.com/code/awsaf49/rsna-bcd-efficientnet-tf-tpu-1vm-train?scriptVersionId=113444994&cellId=67) # noqa: E501
            pos_df = train_df[train_df.cancer == 1].sample(frac=self.hparams.upsample, replace=True)
            neg_df = train_df[train_df.cancer == 0]
            train_df = pd.concat([pos_df, neg_df], axis=0, ignore_index=True)

            self.train_dataset = self._dataset(train_df, self.train_transform)
            self.val_dataset = self._dataset(val_df, self.val_transform)
            self.predict_dataset = self._dataset(val_df, self.val_transform)
        else:
            self.predict_dataset = self._dataset(self.df, self.val_transform)

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def predict_dataloader(self):
        return self._dataloader(self.predict_dataset)

    def _dataset(self, df, transform):
        return RBCDDataset(df=df, transform=transform)

    def _dataloader(self, dataset, train=False):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
        )




########################

class TimmModule(pl.LightningModule):
    def __init__(
            self,
            drop_rate: float,
            drop_path_rate: float,
            eta_min: float,
            learning_rate: float,
            max_epochs: int,
            model_name: str,
            optimizer: str,
            weight_decay: float,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = self._init_model()

        self.metrics = self._init_metrics()

        self._loss_fn = self._init_loss_fn()

    def _init_model(self):
        return timm.create_model(
            self.hparams.model_name,
            pretrained=True,
            num_classes=1,
            drop_rate=self.hparams.drop_rate,
            drop_path_rate=self.hparams.drop_path_rate,
        )

    def _init_metrics(self):
        metrics = {
            "f1": BinaryF1Score(),
        }
        metric_collection = MetricCollection(metrics)

        return torch.nn.ModuleDict(
            {
                "train_metrics": metric_collection.clone(prefix="train_"),
                "val_metrics": metric_collection.clone(prefix="val_"),
            }
        )

    def _init_loss_fn(self):
        return F.binary_cross_entropy_with_logits

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.parameters(),
            opt=self.hparams.optimizer,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.eta_min,
        )

        return [optimizer], [scheduler]

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        images = batch[0]
        logits = self(images).view(-1)
        preds = logits.sigmoid()
        return preds

    def predict_step(self, batch, batch_idx):
        try:
            images, labels = batch[0], batch[1].float()
            logits = self(images).view(-1)
            preds = logits.sigmoid()
            return preds, labels
        except:
            images = batch
            logits = self(images).view(-1)
            preds = logits.sigmoid()
            return preds

    def _shared_step(self, batch, stage):
        images, labels = batch[0], batch[1].float()
        logits = self(images).view(-1)

        loss = self._loss_fn(logits, labels)

        self.metrics[f"{stage}_metrics"](logits, labels)

        self._log(stage, loss, batch_size=len(images))

        return loss

    def _log(self, stage, loss, batch_size):
        self.log(f"{stage}_loss", loss, batch_size=batch_size)
        self.log_dict(self.metrics[f"{stage}_metrics"], batch_size=batch_size)









############ Train



pl.seed_everything(SEED, workers=True)

data_module = TimmDataModule(
    batch_size=BATCH_SIZE,
    data_csv_path="train.csv",
    num_workers=NUM_WORKERS,
    upsample=UPSAMPLE,
    val_fold=VAL_FOLD,
)

module = TimmModule(
    drop_rate=DROP_RATE,
    drop_path_rate=DROP_PATH_RATE,
    eta_min=ETA_MIN,
    learning_rate=LEARNING_RATE,
    max_epochs=MAX_EPOCHS,
    model_name=MODEL_NAME,
    optimizer=OPTIMIZER,
    weight_decay=WEIGHT_DECAY,
)

trainer = pl.Trainer(
    accelerator=ACCELERATOR,
    benchmark=True,
    devices=1,
    fast_dev_run=FAST_DEV_RUN,
    logger=pl.loggers.CSVLogger(save_dir='logs/'),
    log_every_n_steps=5,
    max_epochs=MAX_EPOCHS,
    overfit_batches=OVERFIT_BATCHES,
    precision=PRECISION,
)

trainer.fit(module, datamodule=data_module)







