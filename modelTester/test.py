import sys

import numpy as np
import pandas as pd
import timm
import torch
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from data import getData



def getModel(encoder, num_classes=0):
    model = timm.create_model(encoder, pretrained=True, num_classes=num_classes)
    o = model(torch.randn(2, 3, 224, 224))
    print(f'Original shape: {o.shape}')
    assert len(o.shape)==2
    embSize = o.shape[1]
    #o = model.forward_features(torch.randn(2, 3, 224, 224))
    #print(f'Unpooled shape: {o.shape}')
    return model, embSize



imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}


transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=imagenet_normalize['mean'], std=imagenet_normalize['std'])
])

class TileDataset(Dataset):
    def __init__(self,df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        R = self.df.iloc[item]
        fn = f'/fast/rsna-breast/newtiles/224/{int(R.ptID)}/{int(R.imgID)}_{int(R.row)}_{int(R.col)}.png'
        tile = Image.open(fn).convert('RGB')
        #return tile, R.cancer
        return transform(tile)

'''
def collate(batch):
    labels = torch.from_numpy(np.asarray([b[1] for b in batch]))
    imgs = np.asarray([transform(b[0]) for b in batch])
    imgs = torch.from_numpy(imgs)
    return imgs, labels

'''

def testEncoder(encoder, trainTiles, valTiles, trainLoader, valLoader):
    dev = torch.device('cuda')
    model, embSize = getModel(encoder)
    model = nn.DataParallel(model)
    model.to(dev)
    model.eval()

    def getEmbeddings(loader):
        embeddings = []
        for imgBatch in tqdm(loader, total=len(loader)):
            embs = model(imgBatch.to(dev)).detach().cpu().numpy()
            #print(embs.shape)
            embeddings.append(embs)
        embeddings = np.concatenate(embeddings)
        #print(embeddings.shape)
        return embeddings

    trainEmbeddings = getEmbeddings(trainLoader)
    valEmbeddings = getEmbeddings(valLoader)

    print(f'tembs : {trainEmbeddings.shape}, valembs : {valEmbeddings.shape}')

    logReg = linear_model.LogisticRegression(max_iter=300)

    LR = logReg.fit(trainEmbeddings, trainTiles.cancer)

    def eval(embs, lbls):
        preds = LR.predict_proba(embs)
        AUC = metrics.roc_auc_score(lbls, preds[:, 1])
        pred = 1.0 * (preds[:, 1] > 0.5)
        F1 = metrics.f1_score(lbls, pred)
        return AUC, F1

    trainAUC, trainF1 = eval(trainEmbeddings, trainTiles.cancer)
    valAUC, valF1 = eval(valEmbeddings, valTiles.cancer)

    print(f'Train auc/f1 : {trainAUC} {trainF1}   Val auc/f1: {valAUC} {valF1}')

    return dict(tAUC=trainAUC, tF1=trainF1, vAUC=valAUC, vF1=valF1, encoder=encoder)






if __name__=='__main__':

    modelStats = pd.read_csv('/fast/home/cosmo/rsna-mammography/scripts/timm_model_stats.csv')
    pretrained = timm.list_models(pretrained=True)
    modelStats = modelStats[modelStats.encoder.isin(pretrained)]
    #modelStats = modelStats.sample(frac=1)

    modelStats = modelStats[modelStats.embSize < 192]

    #trainTiles, valTiles = getData()
    DF, _ = getData()
    #nTrain = len(DF)
    #trainTiles = DF.head()
    trainTiles, valTiles = train_test_split(DF, shuffle=True, random_state=42, test_size=0.5)
    print(trainTiles.shape, valTiles.shape)
    print(trainTiles.cancer.value_counts())
    print(valTiles.cancer.value_counts())
    #sys.exit()

    #trainTiles = trainTiles.head(1000)
    trainData, valData = TileDataset(trainTiles), TileDataset(valTiles)
    trainLoader = DataLoader(trainData, num_workers=24, batch_size=16*12)
    valLoader = DataLoader(valData, num_workers=24, batch_size=16*12)


    #resRows = []
    try:
        prevResults = pd.read_csv('resultsDP.tsv', sep='\t')
        print(modelStats.shape)
        modelStats = modelStats[~modelStats.encoder.isin(prevResults.encoder)]
        print(modelStats.shape)
        resRows = prevResults.to_dict(orient='records')
        df = pd.DataFrame(resRows).sort_values('vAUC', ascending=False).round(decimals=3)
        df[['vAUC', 'vF1', 'tAUC', 'tF1', 'encoder']].to_csv('resultsDP.tsv', sep='\t', index=False)
    except:
        resRows = []

    print(resRows)
    #for encoder in ['deit3_small_patch16_224', 'efficientnet_b3', 'vgg16']:
    for _, MR in modelStats.iterrows():
        encoder = MR.encoder
        if '_384' in encoder or '_192' in encoder:
            print('skipping - wrong size')
            continue

        try:
            res = testEncoder(encoder, trainTiles, valTiles, trainLoader, valLoader)
            resRows.append(res)
            df = pd.DataFrame(resRows).sort_values('vAUC', ascending=False).round(decimals=3)
            df[['vAUC', 'vF1', 'tAUC', 'tF1', 'encoder']].to_csv('resultsDP.tsv', sep='\t', index=False)
        except Exception as e:
            print(e)
            #raise


