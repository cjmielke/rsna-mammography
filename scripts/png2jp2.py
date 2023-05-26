from PIL import Image
import os
import time

#import cv2
from glob import glob

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm



def getTiles():
    start = time.time()
    pngs = glob('/fast/rsna-breast/tiles/224/*/*.png')#[:NUM]
    took = time.time()-start
    print(f'Took {took} seconds to run GLOB')
    return pngs


pngs = getTiles()#[:1000]

def proc(png):
    #i = cv2.imread(png)
    p, fn = os.path.split(png)
    _, ptID = os.path.split(p)
    imgID_coords = fn.rstrip('.png')

    outPath = f'/fast/rsna-breast/tiles-jp2/224/{ptID}/'
    outFile = os.path.join(outPath, f'{imgID_coords}.jp2')

    if os.path.exists(outFile):
        #print(f'skipping {outFile}')
        return

    #imgID, row, col = imgID_coords.split('_')

    if not os.path.exists(outPath):
        try: os.makedirs(outPath)
        except: pass
    i = Image.open(png)
    i.save(outFile)


Parallel(n_jobs=16)(delayed(proc)(png) for png in tqdm(pngs))
#pd.DataFrame(rows).to_csv('tileStats.csv', index=False)

