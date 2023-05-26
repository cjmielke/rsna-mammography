import os
from glob import glob
from tqdm import tqdm
import shutil

pngs = glob('/fast/rsna-breast/512-png-ext/*.png')

for png in tqdm(pngs):
    p, f = os.path.split(png)
    fn, ext = os.path.splitext(f)
    ptId, imgId = fn.split('_')
    destDir = os.path.join('/fast/rsna-breast/512-png-ext-imagenet', ptId)
    if not os.path.exists(destDir):
        os.mkdir(destDir)
    destFile = os.path.join(destDir, f'{imgId}.png')
    os.symlink(png, destFile)