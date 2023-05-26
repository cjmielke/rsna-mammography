'''
compute feature vectors for all tiles as quickly as possible
Faster version attempt :

Multiple queues in pipeline? Either way, all tiles for each image need to be grouped together ....

( ptID/imgID pairs ) -> Q1

Q1 -> 16x[glob files, decode all images with PIL, transform PIL -> tensor] -> Q2
Q2 -> 8x[1 worker thread for each GPU collects batches of tiles and runs through GPU]

'''
import torch



import argparse
import os
from typing import List

import timm
from PIL import Image
import pandas as pd
from torch import nn

from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms as T
from multiprocessing import Process, Queue, Lock
from torch.utils.data import DataLoader

# there are 54k unique patientID_imageID pairs
# Can distribute these to workers, which then perform a glob to get all tiles
# for now, lets try writing a single-GPU version

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}



import multiprocessing
import time



class TileDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __getitem__(self, item):
        fn = self.files[item]
        img = Image.open(fn).convert('RGB')
        return img, fn

    def __len__(self):
        return len(self.files)


class GenericCollate(nn.Module):

    def __init__(self):
        super(GenericCollate, self).__init__()

        transform = [
            T.ToTensor(),
            T.Normalize(
                mean=imagenet_normalize['mean'],
                std=imagenet_normalize['std']
            )
        ]

        transform = T.Compose(transform)

        self.transform = transform

    def forward(self, batch: List[tuple]):
        batch_size = len(batch)
        transforms = [self.transform(batch[i][0]).unsqueeze_(0) for i in range(batch_size)]
        # list of filenames
        fnames = [item[1] for item in batch]

        # tuple of transforms
        transformed = torch.cat(transforms, 0)

        return transformed, fnames



class Consumer(multiprocessing.Process):

    def __init__(self, encoder, gpu, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.encoder = encoder
        self.gpu = gpu
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.device = torch.device(f"cuda:{self.gpu}")

        #self.transform = self.makeTransform()


    def makeTransform(self):
        transform = [
            T.ToTensor(),
            #T.Normalize(
            #    mean=imagenet_normalize['mean'],
            #    std=imagenet_normalize['std']
            #)
        ]
        transform = T.Compose(transform)
        return transform



    def loadModel(self, encoder):
        self.model = timm.create_model(encoder, pretrained=True, num_classes=0)
        o = self.model(torch.randn(2, 3, 224, 224))
        print(f'Original shape: {o.shape}')
        o = self.model.forward_features(torch.randn(2, 3, 224, 224))
        print(f'Unpooled shape: {o.shape}')
        self.model.to(self.device)
        self.model.eval()
        #print(self.model)
        self.dataset = None
        self.dataloader = None

    def doTaskBorked(self, task):
        transform = self.makeTransform()
        ptID, imgID, rowCol = task
        tileFiles = [f'/fast/rsna-breast/tiles/224/{ptID}/{imgID}_{R}_{C}.png' for R, C in rowCol]
        for fn in tileFiles:
            img = Image.open(fn).convert('RGB')
            print(fn, img)
            #tensor = self.transform(img)
            tensor = transform(img)
            print('transformed')
            tensor.to(self.device)
            print(tensor.shape)
            emb = self.model(tensor)
            print(f'embedding : {emb.shape}')

    def getDataloader(self, tileFiles):
        if self.dataloader is None:
            self.dataset = TileDataset(tileFiles)
            self.dataloader = DataLoader(self.dataset, batch_size=16, num_workers=1,
                    shuffle=False, drop_last=False, collate_fn=GenericCollate()
            )
        else:
            self.dataset.files = tileFiles
        return self.dataloader

    def doTask(self, task):
        ptID, imgID, rowCol = task
        tileFiles = [f'/fast/rsna-breast/tiles/224/{ptID}/{imgID}_{R}_{C}.png' for R, C in rowCol]
        #dataloader = DataLoader(dataset, batch_size=8, num_workers=1,
        #            shuffle=False, drop_last=False, collate_fn=GenericCollate()
        #)
        dataloader = self.getDataloader(tileFiles)

        embeddings = []
        for tensor, fnames in tqdm(dataloader):
            #print(tensor.shape)
            emb = self.model(tensor.to(self.device))
            #print(f'embedding : {emb.shape}')
            embeddings.append(emb.detach().cpu())

        embeddings = torch.cat(embeddings)
        #print(f'Final embeddings shape : {embeddings.shape}')
        numEmbeddings = embeddings.shape[0]
        assert numEmbeddings == len(tileFiles)

        outPath = f'/fast/rsna-breast/features/{self.encoder}/{ptID}/'
        outFile = os.path.join(outPath, f'{imgID}.pt')
        if not os.path.exists(outPath):
            try: os.makedirs(outPath)
            except: pass
        #print(outFile)
        torch.save(embeddings, outFile)

    def run(self):
        self.loadModel(self.encoder)
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('{}: Exiting'.format(proc_name))
                self.task_queue.task_done()
                break

            #print('{}: {}'.format(proc_name, next_task))
            #answer = next_task()
            self.doTask(next_task)
            self.task_queue.task_done()
            #self.result_queue.put(answer)


class Task:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self):
        time.sleep(0.1)  # pretend to take time to do the work
        return '{self.a} * {self.b} = {product}'.format(
            self=self, product=self.a * self.b)

    def __str__(self):
        return '{self.a} * {self.b}'.format(self=self)


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("-encoder", default='xception41')
    parser.add_argument("-numGPU", default=1, type=int)
    args = parser.parse_args()

    numGPUs = args.numGPU
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue(maxsize=numGPUs*2)
    results = multiprocessing.Queue()

    # Start consumers
    #numGPUs = multiprocessing.cpu_count() * 2
    print('Creating {} GPU workers'.format(numGPUs))
    consumers = [
        Consumer(args.encoder, i, tasks, results)
        for i in range(numGPUs)
    ]
    for w in consumers:
        w.start()

    # Enqueue jobs
    #num_jobs = 10
    #for i in range(num_jobs):
    #    tasks.put(Task(i, i))


    # load up database of all tiles and filter for the ones we want
    #df = pd.read_feather('/fast/rsna-breast/tile_224_stats.feather')
    df = pd.read_feather('/fast/rsna-breast/tile_224_stats_sorted.feather')#.head(200)
    df = df[df['max'] > 50]
    #df = df.sort_values(['ptID', 'imgID'])


    rowCol = []
    ptID, imgID = None, None
    for rn, row in tqdm(df.iterrows(), total=len(df)):      # dataset was pre-sorted
        if row.imgID!=imgID or row.ptID!=ptID:              # this breaks up the rows into patient/image groupings
            #Q.append((ptID, imgID, rowCol))
            if len(rowCol):                                 # skips first iteration
                tasks.put((ptID, imgID, rowCol))
            ptID, imgID = int(row.ptID), int(row.imgID)
            rowCol = []
        rowCol.append((int(row.row), int(row.col)))
    #Q.append((ptID, imgID, rowCol))
    tasks.put((ptID, imgID, rowCol))


    # Add a poison pill for each consumer
    for i in range(numGPUs):
        tasks.put(None)

    # Wait for all of the tasks to finish
    print('waiting for join')
    tasks.join()




