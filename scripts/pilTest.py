#from PIL import Image
import time

import cv2
from glob import glob

from joblib import Parallel, delayed
from tqdm import tqdm


def testAllFullSize():
    #pngs =
    NUM = 16
    pngs = glob('/fast/rsna-breast/pngfull/*/*.png')#[:NUM]

    for png in tqdm(pngs):
        i = cv2.imread(png)
        print(i.shape)

def testAllFullSizeJoblib():
    pngs = glob('/fast/rsna-breast/pngfull/*/*.png')#[:NUM]

    def proc(png):
        i = cv2.imread(png)
        print(i.shape)

    Parallel(n_jobs=10)(delayed(proc)(dicom) for dicom in tqdm(pngs))




def getTiles():
    start = time.time()
    #pngs = glob('/fast/rsna-breast/tiles/224/*/*.png')#[:NUM]
    pngs = glob('/fast/rsna-breast/tiles/224/28624/*.png')  # [:NUM]            # 1690 png tiles
    took = time.time()-start
    print(f'Took {took} seconds to run GLOB')
    return pngs

def testAllTiles():
    pngs = getTiles()
    for png in tqdm(pngs):
        i = cv2.imread(png)

def testAllTilesJoblib():
    pngs = getTiles()
    def proc(png):
        i = cv2.imread(png)

    Parallel(n_jobs=8)(delayed(proc)(dicom) for dicom in tqdm(pngs))


if __name__ == '__main__':
    start = time.time()

    #testAllFullSize()
    #testAllFullSizeJoblib()

    #testAllTiles()
    testAllTilesJoblib()

    took = time.time()-start
    print(f'Took {took} seconds')
    #print(f'Took {took} seconds, or {took/NUM} each')





