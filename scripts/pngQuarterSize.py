import os
from random import shuffle

from PIL import Image
from glob import glob
from joblib import delayed, Parallel
from tqdm import tqdm


imgs = glob('/fast/rsna-breast/pngfull/*/*.png')
shuffle(imgs)

def process(f):
    of = f.replace('pngfull', 'pngQuarter')
    p, fn = os.path.split(of)
    if not os.path.exists(p):
        try: os.makedirs(p)
        except: pass
    if os.path.exists(of):
        return


    try: pil = Image.open(f)
    except:
        print(f)
        raise ValueError(f)
    width, height = pil.size
    newSize = (width//4, height//4)
    pil = pil.resize(newSize)

    pil.save(of)


Parallel(n_jobs=30)(delayed(process)(f) for f in tqdm(imgs))




