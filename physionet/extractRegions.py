import os.path

import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

df = pd.read_csv('/fast/physionet.org/files/vindr-mammo/1.0.0/finding_annotations.csv')
df['imgNum'] = range(0,len(df))

df['width'] = df['xmax']-df['xmin']
df['height'] = df['ymax']-df['ymin']

print(df.height.mean(), df.height.max())

df = df[~df.height.isna()]

def readDicom(dicomF):
    dicom = pydicom.read_file(dicomF)
    img = dicom.pixel_array
    img = (img - img.min()) / (img.max() - img.min())
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img
    return img

IMGSIZE=300

for idx, row in tqdm(df.iterrows(), total=len(df)):
    if df.height is None:
        continue
    #print(idx, row.imgNum)

    outF = f'/fast/rsna-breast/physionet_tiles/{idx}.png'
    if os.path.exists(outF): continue

    dcmF = f'/fast/physionet.org/files/vindr-mammo/1.0.0/images/{row.study_id}/{row.image_id}.dicom'
    img = readDicom(dcmF)

    centerY = int(row.ymin + row.height//2)
    centerX = int(row.xmin + row.width//2)

    startX = max(centerX - IMGSIZE//2, 0)
    startY = max(centerX - IMGSIZE//2, 0)
    endX = startX+IMGSIZE
    endY = startY+IMGSIZE

    print(startX,endX,startY,endY)

    img = img * 255
    img = img.astype(np.uint8)


    roi = img[startX:endX, startY:endY]
    print(outF)
    cv2.imwrite(outF, roi)
