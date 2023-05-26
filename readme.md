python daliFeaturizer.py tbl/tile_224_stats_sorted_part2.feather -encoder deit3_small_patch16_224 -weights chkpt/tileClassifier/deit3_small_patch16_224_sweepy-sweep-45/epoch\=83-step\=21270.ckpt  -out deit_sweepy_sweep -g 12 -colorize 1



# Todo
* [ ] Split tile table into train/val pieces, and store them as feathers
* [ ] Rewrite SSL dataloader to load only the training tiles and apply stratified sampling
* [ ] Write a script to extract attention weights from each tile and export to a file
  * Might only need to do this for cancer patients only
* [ ] Write a basic trainer for binary classification of tiles, which samples based on attention weights
  * proposed sampling weights : 
    * For cancer patients : attention * (1/count)
    * For noncancer : 1.0 * (1/count)
    * Effectively, if all attn weights were 1, this would still be 50/50
* [ ] Get to the bottom of pixel intensity distributions
  * [ ] Run sample timm trainnig code and inspect values
  * [ ] Do the same for sample lightly code 
  * [ ] For one of the encoders, try re-extracting latents with pixels scaled 0-255. Compute std-deviations of the latent vectors


python3 daliFeaturizer.py /fast/rsna-breast/tables/tile_224_stats_sorted_part4.feather -gpu 6  -encoder deit3_medium_patch16_224


5,498,932 Tiles generated at 224 px!

noncancer/cancer counts of patients : [42780, 896]
sampling weights for patients is thus 1/count : [2.3375e-05, 1.1161e-03]
if sampling tiles and not patients, this looks different!

total tilecount : 5,498,932
tiles from cancer patients : 225,380
so a ratio of 0.04 ... call it 5% of the tile dataset are cancer
can keep those persistent in ram and double training speed if stratified sampling is used


Based on a scan of 34k of the ~50k dicoms

df.rows.value_counts()
  4096    24109
  3328     9042
  4740      732
  5928      338
  Name: rows, dtype: int64

df.cols.value_counts()
  3328    24109
  2560     9042
  3540      732
  4728      338
  Name: cols, dtype: int64



df.pixelSpacingC.value_counts()
  0.065238    32224
  0.050000     1070
  0.070000      923
  0.038889        4

Same table for cols/pixelSpacingCols
df[['rows','pixelSpacingR']].value_counts()
  rows  pixelSpacingR
  4096  0.065238         23610
  3328  0.065238          8614
  4740  0.050000           732
  4096  0.070000           499
  3328  0.070000           424
  5928  0.050000           338
  3328  0.038889             4


# all 54k dicoms

dfb.rows.value_counts()
4096    24109
3328     9042
5355     8267
2776     8225
2294     2703
3062     1276
4740      732
5928      338
2850       13
2473        3
1236        2

dfb.cols.value_counts()
3328    24109
2560     9042
4915     8267
2082     8225
1914     2703
2394     1289
3540      732
4728      338
2045        3
1022        2



