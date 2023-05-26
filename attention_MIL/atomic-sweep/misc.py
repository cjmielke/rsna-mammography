import os
from glob import iglob

import torch


def getEmbeddingSizeForEncoder(encoder):
    pat = f'/fast/rsna-breast/features/{encoder}/*/*.pt'
    fn = next(iglob(pat))
    tensor = torch.load(fn)

    embeddingSize = tensor.shape[1]
    return embeddingSize


def getPtImgIDs(imgPath):
    path, fn = os.path.split(imgPath)
    fn, _ = os.path.splitext(fn)
    if '_' in fn:
        imgID, row, col = fn.split('_')
    else:
        imgID = fn
    _, ptID = os.path.split(path)
    return int(ptID), int(imgID)



def pfbetaOld(labels, predictions, beta=1):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0



def pfbeta(labels, preds, beta=1):
    labels = labels.cpu()
    #preds = preds.cpu()

    preds = preds.clip(0, 1)
    y_true_count = labels.sum()

    ctp = preds[labels==1].sum()
    cfp = preds[labels==0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.0


