from time import time

import pandas as pd
import timm
from tqdm import tqdm
import torch

models = timm.list_models()

imgs = torch.zeros((16,3,224,224)).cuda()
device = torch.device('cuda:0')

rows = []
for modelName in tqdm(models):
    try:
        model = timm.create_model(modelName, pretrained=False, num_classes=0)
        model.to(device)
        model.eval()
        start = time()
        for x in range(1):
            out = model.forward(imgs).cpu()
        took = time()-start
        embSize = out.shape[1]
        #print(f'{modelName} took {took} seconds, shape {out.shape}')
        rows.append(dict(model=modelName, took=took, embSize=int(embSize)))
    except Exception as e:
        rows.append(dict(model=modelName, error=str(e)))
        #print(e)

    pd.DataFrame(rows).sort_values('took').to_csv('timm_model_stats.csv', index=False)



