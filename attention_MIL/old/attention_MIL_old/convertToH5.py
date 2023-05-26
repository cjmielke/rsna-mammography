# convert a directory of tensors into an h5 array for faster downstream training
import os
import time
from glob import iglob

import pandas as pd
import tables
import torch
from tqdm import tqdm

# load the instance dataset
trainDF = pd.read_csv('/fast/rsna-breast/train.csv')

# load the tiles dataset
start = time.time()
tileDF = pd.read_feather('/fast/rsna-breast/tables/tile_224_stats_sorted.feather')#.head(1000)
tileDF = tileDF[tileDF['max'] > 50]
print(tileDF.shape)
print(f'It took {time.time()-start} seconds to load the tiles DF')


def convertForEncoder(encoderDir):
    encoderDir = encoderDir.rstrip('/')
    path, encoder = os.path.split(encoderDir)
    outFile = os.path.join(path, f'{encoder}.h5')
    #if os.path.exists(outFile):
    #    raise ValueError(f'Outfile already exists! {outFile}')

    #encoder = 'efficientnet_b3'
    print(encoder, outFile)


    pat = f'/fast/rsna-breast/features/{encoder}/*/*.pt'
    #for fn in iglob(pat):
    fn = next(iglob(pat))
    tensor = torch.load(fn)

    embeddingSize = tensor.shape[1]

    print(f'Expected embedding size is {embeddingSize}')

    h5file = tables.open_file(outFile, mode="w", title=encoder)
    #numTotalTensors = len()
    atom = tables.Float32Atom()
    filters = tables.Filters(complevel=5, complib='zlib')
    ca = h5file.create_carray(h5file.root, 'tensors', atom, (len(tileDF), embeddingSize), filters=filters)


    mammograms = tileDF.groupby(['ptID', 'imgID'])

    lastStartIndex=-1
    for group, DF in tqdm(mammograms):
        ptID, imgID = group
        #print(ptID, imgID)
        #print(DF)
        #c
        nTiles = DF.shape[0]
        startIDX = DF.index[0]
        #startIDX = DF.iloc[0].index
        #print(startIDX, lastStartIndex)
        assert startIDX > lastStartIndex        # should be monotonically increasing ...
        lastStartIndex = startIDX

        # only one row should match.....
        assert len(trainDF.loc[(trainDF.patient_id == ptID) & (trainDF.image_id == imgID)])==1
        trainDF.loc[(trainDF.patient_id == ptID) & (trainDF.image_id == imgID), 'nTiles'] = nTiles
        trainDF.loc[(trainDF.patient_id == ptID) & (trainDF.image_id == imgID), 'startIDX'] = startIDX

        # /fast/rsna-breast/features/efficientnet_b3/532/1031466118.pt
        tensorFile = f'/fast/rsna-breast/features/{encoder}/{ptID}/{imgID}.pt'
        tensor = torch.load(tensorFile)
        if tensor.shape[0]!=nTiles:
            print(f'WARNING! expected {nTiles} but found {tensor.shape[0]} tensors in {tensorFile}')

        ca[startIDX:startIDX+nTiles, :] = tensor.numpy()

    h5file.close()
    trainDF.to_csv(outFile.replace('.h5','_train.csv'))

#convertForEncoder('/fast/rsna-breast/features/efficientnet_b3')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("encoderDir")
args = parser.parse_args()

convertForEncoder(args.encoderDir)


