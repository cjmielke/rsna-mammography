# convert a directory of tensors into an h5 array for faster downstream training
import os
import time
from glob import iglob, glob

import pandas as pd
import tables
import torch
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize



# load the instance dataset
trainDF = pd.read_csv('/fast/rsna-breast/train.csv')

# load the tiles dataset
start = time.time()
tileDF = pd.read_feather('/fast/rsna-breast/tables/tile_224_stats_sorted.feather')
numTotalTiles = len(tileDF)
#numTotalTileFiles = len(glob('/fast/rsna-breast/tiles/*/*/*.png'))
#assert numTotalTiles==numTotalTileFiles


tileDF = tileDF[tileDF['max'] > 50]
numTotalTiles = len(tileDF)
del tileDF


#tileDF = tileDF[tileDF['max'] > 50]
#tileDF = tileDF.tail(2470+61)
#print(tileDF.shape)
print(f'It took {time.time()-start} seconds to load the tiles DF')


def quantizeTensors(tensors):
    arr = tensors.numpy()
    arrUnit = normalize(arr, axis=1)
    intArr = (arrUnit*255).astype(np.int8)
    assert len(intArr)==len(tensors)
    return intArr


def convertForEncoder(encoderDir):
    encoderDir = encoderDir.rstrip('/')
    path, encoder = os.path.split(encoderDir)
    complib = 'blosc:lz4'
    complibName = complib#.replace(':','_')
    outFile = os.path.join('/fast/rsna-breast/featuresH5/', f'{encoder}_quant.h5')

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
    atom = tables.Int8Atom()
    #filters = tables.Filters(complevel=1, complib='zlib')
    filters = tables.Filters(complib=complib, complevel=5)
    ca = h5file.create_carray(h5file.root, 'tensors', atom, (numTotalTiles, embeddingSize), filters=filters)

    #mammograms = tileDF.groupby(['ptID', 'imgID'])

    #lastStartIndex=-1
    startIDX = 0
    #for group, DF in tqdm(mammograms):
    for rn, row in tqdm(trainDF.iterrows(), total=len(trainDF)):
        ptID, imgID = row.patient_id, row.image_id

        # /fast/rsna-breast/features/efficientnet_b3/532/1031466118.pt
        tensorFile = f'/fast/rsna-breast/features/{encoder}/{ptID}/{imgID}.pt'
        tensor = torch.load(tensorFile)
        intTensors = quantizeTensors(tensor)

        nTiles = tensor.shape[0]

        # only one row should match.....
        assert len(trainDF.loc[(trainDF.patient_id == ptID) & (trainDF.image_id == imgID)])==1
        trainDF.loc[(trainDF.patient_id == ptID) & (trainDF.image_id == imgID), 'nTiles'] = nTiles
        trainDF.loc[(trainDF.patient_id == ptID) & (trainDF.image_id == imgID), 'startIDX'] = startIDX

        try:
            ca[startIDX:startIDX+nTiles, :] = intTensors
        except:
            print(startIDX, startIDX+nTiles, nTiles)
            h5file.close()
            trainDF.to_csv(outFile.replace('.h5', '_train.csv'))
            raise

        startIDX += nTiles

    h5file.close()
    trainDF.to_csv(outFile.replace('.h5','_train.csv'))

convertForEncoder('/fast/rsna-breast/features/efficientnet_b3')

'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("encoderDir")
args = parser.parse_args()

convertForEncoder(args.encoderDir)


'''