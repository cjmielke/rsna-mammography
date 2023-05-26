import pandas as pd


def makeSplit(labelDF):
    '''
    Splits dataset into predefined training/validation patients
    '''
    trainPatients = pd.read_csv('/fast/rsna-breast/trainingSplit.csv')
    valPatients = pd.read_csv('/fast/rsna-breast/validationSplit.csv')

    if 'patient_id' in labelDF.columns:
        trainDF = labelDF[labelDF.patient_id.isin(set(trainPatients.patient_id))]
        valDF = labelDF[labelDF.patient_id.isin(set(valPatients.patient_id))]
        # verify no overlap between patients
        assert len(set(trainDF.patient_id).intersection(set(valDF.patient_id))) == 0
    elif 'ptID' in labelDF.columns:
        trainDF = labelDF[labelDF.ptID.isin(set(trainPatients.patient_id))]
        valDF = labelDF[labelDF.ptID.isin(set(valPatients.patient_id))]
        # verify no overlap between patients
        assert len(set(trainDF.ptID).intersection(set(valDF.ptID))) == 0
    else:
        raise ValueError

    print(f'Train/Val shapes : {trainDF.shape} / {valDF.shape}')

    return trainDF, valDF


def getData():
    # get attention data

    atten = pd.read_feather('/fast/rsna-breast/tables/attn_scores_sweet_laughter_413.feather')
    #atten = atten.head(100000)   # FIXME!!!

    # link with cancer at image level
    labels = pd.read_csv('/fast/rsna-breast/train.csv')[['image_id', 'cancer']]
    atten = atten.merge(labels, left_on='imgID', right_on='image_id')

    # for each mammogram, get the top N tiles, by attention score
    # this of course makes the (probably wrong) assumption that the attention scores are correct
    topTiles = atten.sort_values('attention', ascending=False).groupby('image_id').head(1)

    # split into training and validation
    trainTiles, valTiles = makeSplit(topTiles)

    return trainTiles, valTiles



