'''
Idea : when training MIL models, they settle on a final set of tiles and fail to improve

What if, at the end of training, we can split the trainng set into
correctly-classified and incorrectly-classified instances?

During downstream training, with the attention scores, its safe to use high-attention tiles from the noncancer images.
However, it can be DANGEROUS to use high-attention tiles from the cancer set. If the wrong "top tiles" are chosen from
the cancer set, we may be inadvertedly be training the tile classifier to look at the wrong patterns.

Specifically, we better not use high-attention tiles from cancer slides that were not actually predicted as cancer, since
that is potential evidence that the wrong tiles have been selected by the attention mechanism.

So, if we trained a downstream MIL model, we want to just focus on the cancer slides that were missed.

Cancer tiles :
    Preducted noncancer : Missed (false negative) - we want to throw these into a downstream MIL training run
    Correctly guessed : we want to use these to build attention sets, but exclude them for a second stage of MIL training
            This might allow the MIL model to "discover" other tile types

Noncncer tiles :
   Predicted noncancer : we can include these as negatives
   Predicted cancer : these are decoy tiles! Definitely include them to force better discrimination!

'''
import argparse

import pandas as pd
import torch
from tqdm import tqdm

from trainLightning import getParser
from superModel import AttentionMIL_Lightning
from datasets import getTensorFileDataset, EmbeddingDataset, collate

# Load model, load checkpoint, iterate through whole training set, and find predictions

parser = getParser()

#parser = argparse.ArgumentParser()
#parser.add_argument("--encoder", default='deit3_small_patch16_224')
#parser.add_argument("--tileSize", default=224, type=int)
args = parser.parse_args()

args.encoder = 'xcit_polished_leaf'
chkpt = '/fast/rsna-breast/checkpoints/classifier/xcit_polished_leaf/glad-shape-220/epoch=84-step=461165.ckpt'

args.encoder = 'xcit_swept_capybara'
chkpt = '/fast/rsna-breast/checkpoints/classifier/xcit_swept_capybara/giddy-star-782/epoch=11-step=61251.ckpt'


args.model = 'super'
args.tensorDrop=0.0
args.noise = 0


trainDF, valDF = getTensorFileDataset(args)
#trainDF = trainDF[trainDF.cancer==1]         # FIXME



print(f'Size before easy exclusion : {trainDF.shape}')
easyCases = pd.read_csv('/fast/rsna-breast/tables/correct123.csv')
trainDF = trainDF[~trainDF.image_id.isin(easyCases.image_id)]
print(f'Size after easy exclusion : {trainDF.shape}')



trainDataset = EmbeddingDataset(trainDF, args)

#valDataset = EmbeddingDataset(valDF, args)
B = trainDataset[0]
embeddingSize = B[0].shape[1]

trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=args.bs, persistent_workers=True,
                                              shuffle=False, drop_last=False, num_workers=4, collate_fn=collate, sampler=None)

model = AttentionMIL_Lightning(nInput=embeddingSize, wandbRun=None, opt=args.opt, lr=args.lr,
        nReduced=args.reducedDim, nHiddenAttn=args.hiddenAttn,
        classifier='orig', argparse=args
)


model.load_state_dict(torch.load(chkpt)['state_dict'])
model.eval()

outRows = []
for batch in tqdm(trainDataloader, total=len(trainDataloader)):
    embeddings, labels, ages, laterality, rows = batch
    out = model(embeddings, age=ages, laterality=laterality).detach().cpu().numpy()
    #print(out.max())
    rows['prob'] = out
    rows = rows[['image_id', 'cancer', 'prob']]
    outRows.append(rows)

pd.concat(outRows).to_csv('predictions-2.csv')


