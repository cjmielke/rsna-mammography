# split dicom into tiles
import os.path
import sys
import time
from glob import glob
from random import shuffle

from joblib import Parallel, delayed
from tqdm import tqdm

import cv2
import numpy as np
import pydicom
import dicomsdl as dicoml
from PIL import Image


DCMSDL = False



def savePNG(dicomF):

    fp, fn = os.path.split(dicomF)
    imgID, _ = os.path.splitext(fn)
    _, ptID = os.path.split(fp)

    bigPngPath = os.path.join('/fast/rsna-breast/pngfull-new/', ptID)
    if not os.path.exists(bigPngPath):
        try: os.makedirs(bigPngPath)
        except: pass

    bigPng = os.path.join(bigPngPath, f'{imgID}.png')
    if os.path.exists(bigPng):
        print('skipping')
        return

    if DCMSDL:
        dicom = dicoml.open(dicomF)
        img = dicom.pixelData()
        img = (img - img.min()) / (img.max() - img.min())
        if dicom.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
            img = 1 - img
    else:
        dicom = pydicom.read_file(dicomF)
        img = dicom.pixel_array
        img = (img - img.min()) / (img.max() - img.min())
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img

    #print(f'Took {time.time()-start} seconds')

    img = img * 255
    img = img.astype(np.uint8)

    cv2.imwrite(bigPng, img)


f = '/fast/rsna-breast/train_images/10025/1365269360.dcm'
#saveTiles(f, 224)


dicoms = glob('/fast/rsna-breast/train_images/*/*.dcm')

#dicoms = glob('/fast/rsna-breast/train_images/10130/*.dcm')
#assembleFull(65389,240295884)

# test case
#dicoms = ['/fast/rsna-breast/train_images/6654/1480395667.dcm']

#shuffle(dicoms)
#for dicom in tqdm(dicoms):
#    saveTiles(dicom, 224)

def proc(f):
    savePNG(f)

Parallel(n_jobs=25)(delayed(proc)(dicom) for dicom in tqdm(dicoms))






