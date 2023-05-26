import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os

BATCH_SIZE = 64

# workaround for https://github.com/pytorch/vision/issues/1938 - error 403 when downloading mnist dataset
import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


class LitMNIST(LightningModule):

  def __init__(self):
    super().__init__()

    # mnist images are (1, 28, 28) (channels, width, height)
    self.layer_1 = torch.nn.Linear(28 * 28, 128)
    self.layer_2 = torch.nn.Linear(128, 256)
    self.layer_3 = torch.nn.Linear(256, 10)

  def forward(self, x):
    batch_size, channels, width, height = x.size()

    # (b, 1, 28, 28) -> (b, 1*28*28)
    x = x.view(batch_size, -1)
    x = self.layer_1(x)
    x = F.relu(x)
    x = self.layer_2(x)
    x = F.relu(x)
    x = self.layer_3(x)

    x = F.log_softmax(x, dim=1)
    return x

  def process_batch(self, batch):
      return batch

  def training_step(self, batch, batch_idx):
      x, y = self.process_batch(batch)
      logits = self(x)
      loss = F.nll_loss(logits, y)
      return loss

  def cross_entropy_loss(self, logits, labels):
      return F.nll_loss(logits, labels)

  def configure_optimizers(self):
      return Adam(self.parameters(), lr=1e-3)

  def prepare_data(self):
      # download data only
      self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())

  def setup(self, stage=None):
      # transforms for images
      transform=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
      self.mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transform)

  def train_dataloader(self):
       return DataLoader(self.mnist_train, batch_size=64, num_workers=8, pin_memory=True)









model = LitMNIST()
trainer = Trainer(max_epochs=5, gpus=1)
# ddp work only in no-interactive mode, to test it unncoment and run as a script
# trainer = Trainer(gpus=8, strategy="ddp", max_epochs=5)
## MNIST data set is not always available to download due to network issues
## to run this part of example either uncomment below line
# trainer.fit(model)












import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

# Path to MNIST dataset
data_path = os.path.join(os.environ['DALI_EXTRA_PATH'], 'db/MNIST/training/')

@pipeline_def
def GetMnistPipeline(device, shard_id=0, num_shards=1):
    jpegs, labels = fn.readers.caffe2(path=data_path, shard_id=shard_id, num_shards=num_shards, random_shuffle=True, name="Reader")
    images = fn.decoders.image(jpegs,
                               device='mixed' if device == 'gpu' else 'cpu',
                               output_type=types.GRAY)
    images = fn.crop_mirror_normalize(images,
                                      dtype=types.FLOAT,
                                      std=[0.3081 * 255],
                                      mean=[0.1307 * 255],
                                      output_layout="CHW")
    if device == "gpu":
        labels = labels.gpu()
    # PyTorch expects labels as INT64
    labels = fn.cast(labels, dtype=types.INT64)
    return images, labels






class DALILitMNIST(LitMNIST):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
      # no preparation is needed in DALI
      pass

    def setup(self, stage=None):
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size
        mnist_pipeline = GetMnistPipeline(batch_size=BATCH_SIZE, device='gpu', device_id=device_id, shard_id=shard_id,
                                       num_shards=num_shards, num_threads=8)
        self.train_loader = DALIClassificationIterator(mnist_pipeline, reader_name="Reader",
                                                       last_batch_policy=LastBatchPolicy.PARTIAL)
    def train_dataloader(self):
        return self.train_loader

    def process_batch(self, batch):
        x = batch[0]["data"]
        y = batch[0]["label"].squeeze(-1)
        return (x, y)









# Even if previous Trainer finished his work it still keeps the GPU booked, force it to release the device.
if 'PL_TRAINER_GPUS' in os.environ:
    os.environ.pop('PL_TRAINER_GPUS')
model = DALILitMNIST()
trainer = Trainer(max_epochs=5, gpus=1, num_sanity_val_steps=0)
# ddp work only in no-interactive mode, to test it unncoment and run as a script
# trainer = Trainer(gpus=8, strategy="ddp", max_epochs=5)
trainer.fit(model)









class BetterDALILitMNIST(LitMNIST):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
      # no preparation is needed in DALI
      pass

    def setup(self, stage=None):
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size
        mnist_pipeline = GetMnistPipeline(batch_size=BATCH_SIZE, device='gpu', device_id=device_id, shard_id=shard_id, num_shards=num_shards, num_threads=8)

        class LightningWrapper(DALIClassificationIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):
                out = super().__next__()
                # DDP is used so only one pipeline per process
                # also we need to transform dict returned by DALIClassificationIterator to iterable
                # and squeeze the lables
                out = out[0]
                return [out[k] if k != "label" else torch.squeeze(out[k]) for k in self.output_map]

        self.train_loader = LightningWrapper(mnist_pipeline, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    def train_dataloader(self):
        return self.train_loader










# Even if previous Trainer finished his work it still keeps the GPU booked, force it to release the device.
if 'PL_TRAINER_GPUS' in os.environ:
    os.environ.pop('PL_TRAINER_GPUS')
model = BetterDALILitMNIST()
trainer = Trainer(max_epochs=5, gpus=1, num_sanity_val_steps=0)
# ddp work only in no-interactive mode, to test it unncoment and run as a script
# trainer = Trainer(gpus=8, strategy="ddp", max_epochs=5)
trainer.fit(model)
















