from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np


newPNGs = glob('/fast/rsna-breast/pngfull-new/*/*.png')

for newPNG in tqdm(newPNGs):
    oldPNG = newPNG.replace('pngfull-new','pngfull')

    new = np.asarray(Image.open(newPNG))
    old = np.asarray(Image.open(oldPNG))

    newPNG, oldPNG = newPNG.replace('/fast/rsna-breast/',''), oldPNG.replace('/fast/rsna-breast/','')

    if new.shape != old.shape:
        print(f'Shapes wrong : {newPNG} {new.shape} {oldPNG} {old.shape}')
        continue

    if np.abs( new-old ).sum() !=0:
        print(f'Dont match : {new}   {old}')


