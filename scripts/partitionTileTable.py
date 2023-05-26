import numpy as np
import pandas as pd
from tqdm import tqdm

N_parts = 12

TILESET = '224_quarter'

#sourceFile = '/fast/rsna-breast/tables/tile_224_stats_sorted.feather'
sourceFile = f'/fast/rsna-breast/tables/tile_{TILESET}_stats_sorted.feather'
df = pd.read_feather(sourceFile)

patients = df.groupby('ptID')

nPer = patients.ngroups//N_parts
nPer = int(np.ceil(nPer))               # ROUND UP


def savePartition(lis):
    pDF = pd.concat(lis).reset_index()
    del pDF['index']
    #outFile = sourceFile.replace('.feather', f'_part{parition}.feather')
    #outFile = f'/fast/rsna-breast/tables/parts/tile_224_stats_sorted_part{parition}.feather'
    outFile = f'/fast/rsna-breast/tables/parts/tile_{TILESET}_stats_sorted_part{parition}.feather'
    pDF.to_feather(outFile)

parition = 1
lis = []
for ptID, patientDF in tqdm(patients, total=patients.ngroups):
    if len(lis) > nPer:
        savePartition(lis)
        lis = []
        parition += 1
    lis.append(patientDF)
savePartition(lis)

