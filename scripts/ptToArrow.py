import os
import torch
from glob import glob

import pyarrow as pa
import numpy as np
from tqdm import tqdm


def writeArrow(arr, fileName):
    arrays = [pa.array(col) for col in arr]
    names = [str(i) for i in range(len(arrays))]
    batch = pa.RecordBatch.from_arrays(arrays, names=names)
    with pa.OSFile(fileName, 'wb') as sink:
        with pa.RecordBatchStreamWriter(sink, batch.schema) as writer:
            writer.write_batch(batch)


#arr = np.random.randint(65535, size=(250, 4000000), dtype=np.uint16)


pat = '/fast/rsna-breast/features/efficientnet_b3/*/*.pt'

for pt in tqdm(glob(pat)):

    path, fn = os.path.split(pt)
    outFile = pt.replace('efficientnet_b3', 'arrow').replace('.pt','.arrow')

    if not os.path.exists(path):
        os.makedirs(path)

    tensor = torch.load(pt)
    writeArrow(tensor.numpy(), outFile)
