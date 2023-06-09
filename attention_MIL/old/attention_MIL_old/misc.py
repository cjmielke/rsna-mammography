import os


def getPtImgIDs(imgPath):
    path, fn = os.path.split(imgPath)
    fn, _ = os.path.splitext(fn)
    if '_' in fn:
        imgID, row, col = fn.split('_')
    else:
        imgID = fn
    _, ptID = os.path.split(path)
    return int(ptID), int(imgID)



def pfbeta(labels, predictions, beta=1):
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



