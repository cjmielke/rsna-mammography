import os
from glob import glob

import pandas as pd
import torch

df = pd.read_feather('/fast/rsna-breast/tables/tile_224_stats_sorted.feather')#.head(200)
df = df[df['max'] > 50]

for p in glob('/fast/rsna-breast/features/xception41/*/*.pt'):
    path, fn = os.path.split(p)
    imgID = fn.rstrip('.pt')
    _, ptID = os.path.split(path)
    t = torch.load(p)
    rows = df[(df.ptID==int(ptID)) & (df.imgID==int(imgID))]
    print(rows)
    print(t.shape, rows.shape)
    assert len(rows)==len(t)
    print('================')

