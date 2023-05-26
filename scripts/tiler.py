# split dicom into tiles
import os.path
from glob import glob

from joblib import Parallel, delayed
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image


def reshape_split(image: np.ndarray, kernel_size: tuple):
    #print(image.shape)
    height, width = image.shape
    tHeight, tWidth = kernel_size
    # need to pad
    tR, tC = height//tHeight, width//tWidth
    #print(tR, tC)
    pR, pC = height-tR*tHeight, width-tC*tWidth
    #print(tHeight-pR, tWidth-pC)
    image = np.pad(image, ((0,tHeight-pR), (0, tWidth-pC)))
    #print(image.shape)
    height, width = image.shape
    tiled_array = image.reshape(height//tHeight, tHeight, width//tWidth, tWidth)
    tiled_array = tiled_array.swapaxes(1,2)
    return tiled_array



def makeTiles(img, tileSize):
    #d = pydicom.read_file(dicom)
    #img = d.pixel_array
    tiled = reshape_split(img, (tileSize, tileSize))
    #print(tiled.shape)
    return tiled



def saveTiles(bigPng, tileSize):

    fp, fn = os.path.split(bigPng)
    imgID, _ = os.path.splitext(fn)
    _, ptID = os.path.split(fp)

    pil = Image.open(bigPng)

    # FIXME - make a half-sized image. Tiles are still 224, but encompass twice the region of interest
    width, height = pil.size
    newSize = (width//4, height//4)
    pil = pil.resize(newSize)


    img = np.asarray(pil)
    #img = (img - img.min()) / (img.max() - img.min())
    #print(f'Took {time.time()-start} seconds')
    #img = img * 255
    #img = img.astype(np.uint8)


    ########## Now we extract tiles

    destPath = os.path.join(f'/fast/rsna-breast/newtiles/{tileSize}_quarter/', ptID)

    if not os.path.exists(destPath):
        try: os.makedirs(destPath)      # race condition can cause crash
        except: pass


    tiles = makeTiles(img, tileSize)

    tileRows, tileCols, _, _ = tiles.shape
    for row in range(tileRows):
        for col in range(tileCols):
            fn = f'{imgID}_{row}_{col}.png'
            outFile = os.path.join(destPath, fn)
            if os.path.exists(outFile):
                #print('skipping tile')
                continue
            tile = tiles[row][col]
            #print(tile.shape, tile.min(), tile.mean(), tile.max())
            if tile.max()==0.0 or tile.mean()==0:
                continue
            nonzeros = np.count_nonzero(tile)
            nonzeroFrac = nonzeros/tile.size
            #print(nonzeroFrac, nonzeros, tile.size)
            #print(nonzeroFrac)
            if nonzeroFrac < 0.2: continue

            #fn = f'pngs/{row}_{col}.png'
            cv2.imwrite(outFile, tile)





#dicoms = glob('/fast/rsna-breast/train_images/*/*.dcm')
PNGS = glob('/fast/rsna-breast/pngfull-new/*/*.png')

#dicoms = glob('/fast/rsna-breast/train_images/10130/*.dcm')
#assembleFull(65389,240295884)

# test case
#dicoms = ['/fast/rsna-breast/train_images/6654/1480395667.dcm']

#shuffle(dicoms)
#for dicom in tqdm(dicoms):
#    saveTiles(dicom, 224)

def proc(f):
    saveTiles(f, 224)

Parallel(n_jobs=30)(delayed(proc)(p) for p in tqdm(PNGS))



