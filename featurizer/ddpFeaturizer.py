import argparse
import os

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import loadModel
import pandas as pd
from PIL import Image
from torchvision import transforms as T

#from torch.nn.parallel import DistributedDataParallel
#from torch import distributed as dist
#dist.init_process_group(backend='nccl')

class TilesDataset(Dataset):
    def __init__(self):
        self.df = pd.read_feather('/fast/rsna-breast/tables/tile_224_stats_sorted.feather').reset_index()
        print(f'Loaded tiles dataset : {self.df.shape}')
        print(self.df.head())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        R = self.df.iloc[item].astype(int)
        tileFile = f'/fast/rsna-breast/newtiles/224/{R.ptID}/{R.imgID}_{R.row}_{R.col}.png'
        pil = Image.open(tileFile).convert('RGB')
        return item, pil, R


imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}
norm = imagenet_normalize
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=norm['mean'], std=norm['std'])
])

def collate(batch):
    rows = [b[2] for b in batch]
    idx = torch.from_numpy(np.asarray([b[0] for b in batch]))
    imgs = torch.stack([transform(b[1]) for b in batch])
    return idx, imgs, rows





class Model(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=True, num_classes=0)
        o = self.encoder(torch.randn(2, 3, 224, 224))
        print(f'Original shape: {o.shape}')

    def forward(self, idxs, imgs):
        embs = self.encoder(imgs)
        return idxs, embs

if __name__ == '__main__':

    #torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    #parser.add_argument("df")

    parser.add_argument("-encoder", default='deit3_small_patch16_224')

    #parser.add_argument("-weights", default=None)
    parser.add_argument("-weights", default='/fast/rsna-breast/checkpoints/tileClassifier/deit3_small_patch16_224_brilliant-laughter-2/epoch=12-step=3301.ckpt')

    #parser.add_argument("-out", default='deit3_small_patch16_224')
    parser.add_argument("-out", default='deit_brilliant_laughter')
    #parser.add_argument("-gpu", default=0, type=int)
    args = parser.parse_args()


    def saveEmbeddings(embeddingsList, ptID, imgID):
        if len(embeddingsList) == 0: return
        embeddings = torch.stack(embeddingsList)
        outPath = f'/fast/rsna-breast/features/{args.out}/{ptID}/'
        outFile = os.path.join(outPath, f'{imgID}.pt')
        if not os.path.exists(outPath):
            try:
                os.makedirs(outPath)
            except:
                pass
        # print(type(embeddings))
        torch.save(embeddings, outFile)


    encoder = Model(args.encoder)
    #model = DistributedDataParallel(encoder)

    model = torch.nn.DataParallel(encoder)
    print('Model:', type(model))
    print('Devices:', model.device_ids)

    model = model.cuda()

    dataset = TilesDataset()
    dataloader = DataLoader(dataset, collate_fn=collate, num_workers=35, batch_size=32)

    nExpectedBatches = len(dataset) // 64
    curIDX = -1
    activePtID, activeImgID = None, None
    embeddingsList = []
    for idxs, imgs, rows in tqdm(dataloader, total=nExpectedBatches):

        Outidxs, embs = model(idxs.cuda(), imgs.cuda())
        embs = embs.detach().cpu()
        #print(embs.shape)
        #print(Outidxs)
        for idx, embedding, row in zip(Outidxs, embs, rows):
            curIDX += 1
            assert curIDX == idx

            ptID, imgID, = row['ptID'], row['imgID']
            if imgID != activeImgID:
                saveEmbeddings(embeddingsList, activePtID, activeImgID)
                activePtID, activeImgID = ptID.item(), imgID.item()
                embeddingsList = []
            embeddingsList.append(embedding)
        saveEmbeddings(embeddingsList, activePtID, activeImgID)



