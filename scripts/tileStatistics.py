#from PIL import Image
import os
import time

import cv2
from glob import glob

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

TILESET = '224_quarter'

def getTiles():
    start = time.time()
    pngs = glob(f'/fast/rsna-breast/tiles/{TILESET}/*/*.png')#[:NUM]
    took = time.time()-start
    print(f'Took {took} seconds to run GLOB')
    return pngs


pngs = getTiles()#[:1000]

def proc(png):
    i = cv2.imread(png)
    p, fn = os.path.split(png)
    _, ptID = os.path.split(p)
    imgID_coords = fn.rstrip('.png')
    imgID, row, col = imgID_coords.split('_')
    nonzeros = np.count_nonzero(i)
    nonzeroFrac = nonzeros / i.size

    return dict(ptID=int(ptID), imgID=int(imgID), row=int(row), col=int(col),
                min=i.min(), max=i.max(), std=i.std(), mean=i.mean(),
                nonzeroFrac=nonzeroFrac
                )

rows = Parallel(n_jobs=8)(delayed(proc)(dicom) for dicom in tqdm(pngs))
df = pd.DataFrame(rows).sort_values(['ptID', 'imgID', 'row', 'col']).reset_index()
#df.to_csv('tileStats.csv', index=False)

#df.to_feather('tileStats_20pct_sorted.feather')
df.to_feather(f'/fast/rsna-breast/tables/tile_{TILESET}_stats_sorted.feather')


