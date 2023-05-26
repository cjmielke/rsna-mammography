"""
Important to define a permanent train/validation split for all future experiments
"""
from subprocess import check_call

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

labelDF = pd.read_csv('/fast/rsna-breast/train.csv')
labelDF = labelDF.sample(frac=1, random_state=42)  # shuffle!
splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7)
split = splitter.split(labelDF, groups=labelDF['patient_id'])
train_inds, val_inds = next(split)

trainDF = labelDF.iloc[train_inds]
valDF = labelDF.iloc[val_inds]

print(f'Train/Val split : {trainDF.shape} / {valDF.shape}')

print('Training dataset')
print(trainDF.cancer.value_counts())

print('Validation dataset')
print(valDF.cancer.value_counts())


trainDF[['patient_id','image_id']].to_csv('/fast/rsna-breast/trainingSplit.csv', index=False)
print('Training MD5')
check_call(['md5sum','/fast/rsna-breast/trainingSplit.csv'])
valDF[['patient_id','image_id']].to_csv('/fast/rsna-breast/validationSplit.csv', index=False)
print('Validation MD5')
check_call(['md5sum','/fast/rsna-breast/validationSplit.csv'])

"""
Training MD5                                                                                                              │
1bb2546e45af84100217011de37dee14  /fast/rsna-breast/trainingSplit.csv                                                     │
Validation MD5                                                                                                            │
b31a20c1e9fd99baf2c07640560fa9a4  /fast/rsna-breast/validationSplit.csv    
"""



