import os
import shutil

import pandas as pd
import torch.hub
from tqdm import tqdm

from misc import getEmbeddingSizeForEncoder



from trainLightning import parser
args = parser.parse_args()
args.checkpoint = None          # set manually below

def compute():


    chkpt = '/fast/rsna-breast/checkpoints/efficientnet_b3_giddy_rain/epoch=24-step=16393.ckpt'
    encoder = 'efficientnet_b3'
    tensorPath = f'/fast/rsna-breast/features/{encoder}/'
    embeddingSize = getEmbeddingSizeForEncoder(encoder)


    tileDF = pd.read_feather('/fast/rsna-breast/tables/tile_224_stats_sorted.feather')
    print(tileDF)


    tileDF = tileDF[tileDF['max']>50]

    trainDF = pd.read_csv('/fast/rsna-breast/train.csv')
    cancerDF = trainDF[trainDF.cancer == 1]
    cancerPatients = list(cancerDF.patient_id)
    cancerImages = list(cancerDF.image_id)

    print(tileDF.shape)
    tilesFromCancerPatientsDF = tileDF[tileDF.ptID.isin(cancerPatients)]
    print(tilesFromCancerPatientsDF.shape)

    tilesFromCancerImagesDF = tileDF[tileDF.imgID.isin(cancerImages)]
    print(tilesFromCancerImagesDF.shape)

    #pat = os.path.join(tensorPath, '*/*.pt')
    model = AttentionMIL_Lightning(nInput=embeddingSize, nReduced=64)
    print('WTF')
    state_dict = torch.load(chkpt)['state_dict']
    model.load_state_dict(state_dict)


    newDF = []
    #for (ptID, imgID), df in tqdm(tilesFromCancerPatientsDF.groupby(['ptID', 'imgID'])):       # by patient
    for (ptID, imgID), df in tqdm(tilesFromCancerImagesDF.groupby(['ptID', 'imgID'])):          # by image
        #print(ptID, imgID)
        ptFile = os.path.join(tensorPath, f'{ptID}/{imgID}.pt')
        tensorBag = torch.load(ptFile)
        if len(tensorBag) != len(df):        # should be same length, and importantly, same order!
            print(f'tensor bag has {len(tensorBag)} tensors, expected {len(df)}')
            raise ValueError
        #print(tensorBag.shape)
        attentionScores = model.getAttentionScores([tensorBag])
        #print(attentionScores.min(), attentionScores.max(), attentionScores.shape)
        df['attention'] = attentionScores[0].numpy().tolist()
        for c in ['min', 'max', 'std', 'mean']: del df[c]
        newDF.append(df)

    newDF = pd.concat(newDF)
    newDF.to_csv('attn_scores.csv', index=False)


def computeFromAll():
    """
    Instead of focusing on the cancer images, maybe we can compute attention scores from ALL patches
    We can then train a classifier to distinguish high-attention patches from the cancer/noncancer images
    This might be a much harder task then predicing high-attn-cancer / vs everything else
    """
    #chkpt = '/fast/rsna-breast/checkpoints/efficientnet_b3_giddy_rain/epoch=24-step=16393.ckpt'
    #encoder = 'efficientnet_b3'
    #chkpt = '/fast/rsna-breast/checkpoints/classifier/deit3_small_patch16_224/graceful-sweep-60/epoch=24-step=16393.ckpt'

    #chkpt = '/fast/rsna-breast/checkpoints/classifier/deit3_small_patch16_224/kind-sweep-53/epoch=57-step=39219.ckpt'
    #encoder = 'deit3_small_patch16_224'

    #chkpt = '/fast/rsna-breast/checkpoints/classifier/deit_sweepy_sweep/classic-sweep-9/epoch=61-step=41803.ckpt'
    #encoder = 'deit_sweepy_sweep'

    #chkpt = '/fast/rsna-breast/checkpoints/classifier/deit_sweepy_sweep/atomic-sweep-134/epoch=22-step=15503.ckpt'
    #encoder = 'deit_sweepy_sweep'

    #chkpt = '/fast/rsna-breast/checkpoints/classifier/deit_sweepy_sweep/dandy-sweep-42/epoch=8-step=5929.ckpt'
    #encoder = 'deit_sweepy_sweep'

    #####  PNGS and TILES regenerated here .... new stuff follows with untainted data

    #chkpt = '/fast/rsna-breast/checkpoints/classifier/deit3_small_patch16_224/apricot-sweep-45/epoch=59-step=40675.ckpt'
    #encoder = 'deit3_small_patch16_224'

    #chkpt = '/fast/rsna-breast/checkpoints/classifier/deit_sweepysweep_new/sweet-laughter-413/epoch=43-step=29710.ckpt'
    #encoder = 'deit_sweepysweep_new'

    chkpt = '/fast/rsna-breast/checkpoints/classifier/xcit_polished_leaf/glad-shape-220/epoch=84-step=461165.ckpt'
    encoder = 'xcit_polished_leaf'

    tensorPath = f'/fast/rsna-breast/features/224/{encoder}/'
    embeddingSize = getEmbeddingSizeForEncoder(encoder)

    tileDF = pd.read_feather('/fast/rsna-breast/tables/tile_224_stats_sorted.feather')
    #tileDF = pd.read_feather('/fast/rsna-breast/tables/old_broken/tile_224_stats_sorted.feather')      # FIXME

    # for deit_sweepy_sweep, all tiles needed
    #tileDF = tileDF[tileDF['max'] > 50]

    # pat = os.path.join(tensorPath, '*/*.pt')

    if args.model == 'super':
        from superModel import AttentionMIL_Lightning
    else:
        from models import AttentionMIL_Lightning

    model = AttentionMIL_Lightning(nInput=embeddingSize, nReduced=args.reducedDim, argparse=args)

    state_dict = torch.load(chkpt)['state_dict']
    model.load_state_dict(state_dict)

    newDF = []
    # for (ptID, imgID), df in tqdm(tilesFromCancerPatientsDF.groupby(['ptID', 'imgID'])):       # by patient
    for (ptID, imgID), df in tqdm(tileDF.groupby(['ptID', 'imgID'])):  # by image
        # print(ptID, imgID)
        ptFile = os.path.join(tensorPath, f'{ptID}/{imgID}.pt')
        tensorBag = torch.load(ptFile, map_location=torch.device('cpu')).detach()
        if len(tensorBag) != len(df):        # should be same length, and importantly, same order!
            print(f'tensor bag has {len(tensorBag)} tensors, expected {len(df)}')
            print(ptFile)
            raise ValueError
        #continue
        attentionScores, rawAttention = model.getAttentionScores([tensorBag])
        df['attention'] = attentionScores[0].numpy().tolist()
        df['raw'] = rawAttention[0].numpy().tolist()
        for c in ['min', 'max', 'std', 'mean']: del df[c]
        newDF.append(df)

    newDF = pd.concat(newDF)
    #newDF.to_csv('attn_scores_all.csv', index=False)
    newDF.reset_index().to_feather('attn_scores_all.feather')


def saveTopScoring():
    #DF = pd.read_csv('attn_scores_all.csv')
    DF = pd.read_feather('attn_scores_all.feather')

    df = DF.sort_values('attention', ascending=False).head(500).reset_index(drop=True)
    print(df)
    for rn, row in df.iterrows():
        tileFile = f'/fast/rsna-breast/tiles/224/{int(row.ptID)}/{int(row.imgID)}_{int(row.row)}_{int(row.col)}.png'
        p, fn = os.path.split(tileFile)
        fn = f'{rn}_{row.attention}_{fn}'
        dest = os.path.join('/fast/rsna-breast/topTiles/', fn)
        shutil.copyfile(tileFile,dest)

    df = DF.sort_values('raw', ascending=False).head(500).reset_index(drop=True)
    print(df)
    for rn, row in df.iterrows():
        tileFile = f'/fast/rsna-breast/tiles/224/{int(row.ptID)}/{int(row.imgID)}_{int(row.row)}_{int(row.col)}.png'
        p, fn = os.path.split(tileFile)
        fn = f'{rn}_{row.raw}_{fn}'
        dest = os.path.join('/fast/rsna-breast/topTilesRaw/', fn)
        shutil.copyfile(tileFile,dest)


if __name__=='__main__':
    #compute()
    computeFromAll()
    saveTopScoring()

    #newDF = pd.read_csv('attn_scores_all.csv')
    #print(newDF.columns)
    #newDF.reset_index().to_feather('attn_scores_all.feather')



