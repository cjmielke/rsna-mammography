#!/usr/bin/env python

# taken from https://practical.codes/dali/cv/image_processing/2020/11/10/nvidia_dali.html#The-Dataloader
import argparse

import pandas as pd
import torch
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from torch.utils.data import IterableDataset
import numpy as np
import os

from tqdm import tqdm
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

from models import loadModel


def getPtImgIDs(imgPath):
    path, fn = os.path.split(imgPath)
    fn, _ = os.path.splitext(fn)
    if '_' in fn:
        imgID, row, col = fn.split('_')
    else:
        imgID = fn
    _, ptID = os.path.split(path)
    return int(ptID), int(imgID)


#Just a Pytorch IterableDataset yielding (image,label,image_path) or whateverelse you want!
class DALIDataset(IterableDataset):
    def __init__(self, df, ARGS, **kwargs):
        super().__init__()
        self.df = df
        self.args = ARGS
        self.recs = self.df[['ptID', 'imgID', 'row', 'col']].to_records(index=False)
        #self.files = glob("/fast/rsna-breast/tiles-jp2/224/28624/*.jp2")
        #self.files = glob("/fast/rsna-breast/tiles-jp2/224/*/*.jp2")
        #print(f'{len(self.files)} files found')

    def __len__(self):
        return len(self.recs)

    def __iter__(self):
        for ptID, imgID, R, C in self.recs:
            #image_path = f'/fast/rsna-breast/tiles-jp2/224/{int(ptID)}/{int(imgID)}_{int(R)}_{int(C)}.jp2'
            image_path = f'/fast/rsna-breast/tiles/{self.args.tileSet}/{int(ptID)}/{int(imgID)}_{int(R)}_{int(C)}.png'
            f = open(image_path, "rb")
            image = np.frombuffer(f.read(), dtype=np.uint8) # don't decode the image just read the bytes!
            ptID = np.array([ptID])  # some label
            imgID = np.array([imgID])  # some label
            #print(ptID, imgID)
            #image = image.copy()           # this definitely broke DALI, could no longer recognize a valid jp2 file ...
            #image[0:100]=0
            yield image, ptID, imgID



#Using ExternalSource
class SimplePipeline(Pipeline):
    #Define the operations in the pipeline
    def __init__(self, external_datasource, args, max_batch_size=16, num_threads=2, device_id=0, is_train=True):
        super(SimplePipeline, self).__init__(max_batch_size, num_threads, device_id, seed=12)
        # Define Input nodes
        self.args = args
        resolution = args.tileSize
        crop = args.tileSize
        self.jpegs = ops.ExternalSource()
        self.in_ptIDs = ops.ExternalSource()
        self.in_imgIDs = ops.ExternalSource()
        ## Or pass source straight to ExternalSource this way you won't have do iter_setup.
        # self.jpegs,self.labels,self.paths=ops.ExternalSource(source=self.make_batch, num_outputs=3)

        # Define ops
        #self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        #outType = types.RGB if args['colorize']==0 else types.GRAY
        outType = types.RGB #if args['colorize']==0 else types.GRAY
        #outType = types.GRAY
        self.decode = ops.decoders.Image(device="mixed", output_type=outType)
        self.res = ops.Resize(device="gpu", resize_x=resolution, resize_y=resolution)
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        #mean = [0.5,0.5,0.5]
        #std = [0.5, 0.5, 0.5]
        #mean, std = None, None
        self.normalize = ops.CropMirrorNormalize(device="gpu", dtype=types.FLOAT, output_layout=types.NCHW,
                                            crop=(crop, crop), mean=mean, std=std
        )
        #self.path_pad = ops.Pad(fill_value=ord("?"),axes = (0,)) # We need to pad image_paths because we need the shapes to match.need dense tensor

        self.iterator = iter(external_datasource)


    # How the operations in the pipeline are used
    # Connect your input nodes to your ops
    def define_graph(self):

        self.images = self.jpegs()
        self.ptIDs = self.in_ptIDs()
        self.imgIDs = self.in_imgIDs()
        images = self.decode(self.images)
        #images = self.res(images)
        images = self.normalize(images)

        #paths = self.path_pad(self.paths)

        return (images, self.ptIDs, self.imgIDs)

    # The external source should be fed batches
    # I prefer to batch-ify things here because it keeps things compatible with an IterableDataset
    def make_batch(self):
        imgs, ptIDs, imgIDs = [], [], []
        for _ in range(self.max_batch_size):
            try:
                i, l, p = next(self.iterator)
            except StopIteration:
                break
            imgs.append(i)
            ptIDs.append(l)
            imgIDs.append(p)
        if len(imgs) == 0:
            raise StopIteration
        return imgs, ptIDs, imgIDs



    # Only needed when using ExternalSource
    # Connect the dataset outputs to external Sources
    def iter_setup(self):
        (images, ptIDs, imgIDs) = self.make_batch()
        self.feed_input(self.images, images)
        self.feed_input(self.ptIDs, ptIDs)
        self.feed_input(self.imgIDs, imgIDs)





from nvidia.dali.plugin.pytorch import DALIGenericIterator

def make_pipeline(dataset, args, device_index=0, num_threads=1, is_train=False):
    return_keys = ["images", "ptIDs", "imgIDs"]
    pipeline = SimplePipeline(dataset, args,  max_batch_size=args.bs, num_threads=num_threads,
                        device_id=device_index, is_train=is_train)
    pipeline_iterator = DALIGenericIterator([pipeline], return_keys,
                                            last_batch_policy=LastBatchPolicy.PARTIAL, dynamic_shape=True, last_batch_padded = True
                                            )
    return pipeline_iterator


def lookForExistingTensors(df, encoder):
    cutDF = []
    for (ptID, imgID), patientDF in df.groupby(['ptID','imgID']):
        tensorFile = f'/fast/rsna-breast/features/{encoder}/{ptID}/{imgID}.pt'
        if not os.path.exists(tensorFile):
            cutDF.append(patientDF)
    return pd.concat(cutDF)

if __name__ == '__main__':

    #torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("-df")
    #parser.add_argument("-encoder", default='deit3_small_patch16_224')
    #parser.add_argument("-encoder", default='deit3_small_patch16_384')
    #parser.add_argument("-encoder", default='efficientnet_b3')
    #parser.add_argument("-encoder", default='edgenext_xx_small')
    #parser.add_argument("-encoder", default='tf_mobilenetv3_small_minimal_100')
    #parser.add_argument("-encoder", default='')

    #parser.add_argument("-weights", default=None)
    #parser.add_argument("-weights", default='/fast/rsna-breast/checkpoints/tileClassifier/deit3_small_patch16_224_sweepy-sweep-45/epoch=83-step=21270.ckpt')
    #parser.add_argument("-weights", default='/fast/rsna-breast/checkpoints/regionClassifier/deit3_small_patch16_224_whole-sweep-23/epoch=95-step=24540.ckpt')
    #parser.add_argument("-weights", default='/fast/rsna-breast/checkpoints/regionClassifier/deit3_small_patch16_224_flowing-sweep-16/epoch=98-step=25287.ckpt')
    #parser.add_argument("-weights", default='/fast/rsna-breast/checkpoints/moco/efficientnet_b3_vibrant-rocket-72/epoch=0-step=38269.ckpt')
    #parser.add_argument("-weights", default='/fast/rsna-breast/checkpoints/tileClassifier/deit3_small_patch16_224_auspicious-bao-3/epoch=69-step=17665.ckpt')
    #parser.add_argument("-weights", default='/fast/rsna-breast/checkpoints/swav/efficientnet_b3_lunar-springroll-83/epoch=0-step=17821.ckpt')
    #parser.add_argument("-weights", default='/fast/rsna-breast/checkpoints/tileClassifier/deit3_small_patch16_224_virtuous-tiger-12/epoch=21-step=5476.ckpt')
    #parser.add_argument("-weights", default='/fast/rsna-breast/checkpoints/tileClassifier/deit3_small_patch16_224_vague-sweep-12/epoch=69-step=17920.ckpt')
    #parser.add_argument("-weights", default='/fast/rsna-breast/checkpoints/tileClassifier/tf_mobilenetv3_small_minimal_100_devoted-donkey-47/epoch=3-step=6217.ckpt')
    #parser.add_argument('-weights', default='/fast/rsna-breast/checkpoints/tileClassifier/tf_mobilenetv3_small_minimal_100_misunderstood-voice-49/epoch=17-step=34817.ckpt')
    #parser.add_argument('-weights', default='')

    #parser.add_argument("-out", default='foo')
    #parser.add_argument("-out", default='deit_flowing_sweep_16')
    #parser.add_argument("-out", default='deit_whole_sweep_23')
    #parser.add_argument("-out", default='edgenext_xx_small')
    #parser.add_argument("-out", default='effb3_swav_lunar_springroll')
    #parser.add_argument("-out", default='deit3_small_patch16_384')
    #parser.add_argument("-out", default='deit_virtuous_tiger')
    #parser.add_argument("-out", default='deit_vague_sweep_12')
    #parser.add_argument("-out", default='mobilenet_misunderstood_voice')

    #parser.add_argument("-encoder", default='vit_tiny_patch16_224')
    #parser.add_argument('-weights', default='/fast/rsna-breast/checkpoints/tileClassifier/vit_tiny_patch16_224_smooth-frost-73/epoch=18-step=11452.ckpt')
    #parser.add_argument("-out", default='vit_tiny_smooth_frost')


    #parser.add_argument("-encoder", default='xcit_nano_12_p8_224_dist')
    #parser.add_argument('-weights', default=None)
    #parser.add_argument("-out", default='xcit_nano_12_p8_224_dist')

    #parser.add_argument("-encoder", default='xcit_nano_12_p16_224_dist')
    #parser.add_argument('-weights', default='')
    #parser.add_argument("-out", default='xcit_nano_')



    #parser.add_argument("-encoder", default='xcit_nano_12_p16_224_dist')
    #parser.add_argument('-weights', default='/fast/rsna-breast/checkpoints/tileClassifier/xcit_nano_12_p16_224_dist_polished-leaf-74/epoch=17-step=10549.ckpt')
    #parser.add_argument("-out", default='xcit_polished_leaf')


    #parser.add_argument("-encoder", default='xcit_nano_12_p16_224_dist')
    #parser.add_argument('-weights', default='/fast/rsna-breast/checkpoints/tileClassifier/xcit_nano_12_p16_224_dist_swept-capybara-106/epoch=32-step=11400.ckpt')
    #parser.add_argument("-out", default='xcit_swept_capybara')


    #parser.add_argument("-encoder", default='xcit_nano_12_p16_224_dist')
    #parser.add_argument('-weights', default='/fast/rsna-breast/checkpoints/tileClassifier/xcit_nano_12_p16_224_dist_thoughtful-ring-108/epoch=71-step=4844.ckpt')
    #parser.add_argument("-out", default='xcit_thoughtful_ring')

    parser.add_argument("-encoder", default='nyu')
    parser.add_argument('-weights', default=None)       # weights are loaded in model
    parser.add_argument("-out", default='nyu')


    #parser.add_argument("-encoder", default='')
    #parser.add_argument('-weights', default=None)
    #parser.add_argument("-out", default='')

    #parser.add_argument("-encoder", default='')
    #parser.add_argument('-weights', default=None)
    #parser.add_argument("-out", default='')


    #parser.add_argument("-numGPU", default=1, type=int)
    parser.add_argument("-gpu", default=0, type=int)        # FIXME - need to get the DALI loader on same device
    parser.add_argument("-colorize", default=2, type=int)
    parser.add_argument("-bs", default=32, type=int)
    parser.add_argument("-tileSize", default=224, type=int)
    parser.add_argument("-tileSet", default='224', type=str)
    args = parser.parse_args()

    args.df = f'/fast/rsna-breast/tables/parts/tile_{args.tileSet}_stats_sorted_part{args.gpu}.feather'

    device = torch.device(f"cuda:{args.gpu}")
    encoder = loadModel(args, device=device)

    outDir = args.out or args.encoder

    # load up database of all tiles and filter for the ones we want
    #df = pd.read_feather('/fast/rsna-breast/tile_224_stats.feather')
    #df = pd.read_feather('/fast/rsna-breast/tile_224_stats_sorted.feather')#.head(200)
    df = pd.read_feather(args.df)
    #df = df.sort_values(['ptID', 'imgID'])

    #if 'max' in df.columns:
    #    df = df[df['max'] > 50]

    df = lookForExistingTensors(df, outDir)
    print(df)


    argsD = {
        'colorize': args.colorize,
        #"resolution": 224,
        #"crop": 224,
        #"batch_size": 32,
        # "max_batch_size": 128,
        # "image_folder": "/fast/rsna-breast/tiles-jp2/224/28624" # Change this
        #"image_folder": "/fast/rsna-breast/tiles-jp2/224/28624"  # Change this
    }

    total = 0
    dataset = DALIDataset(df, args)
    #train_dataloader = make_pipeline(dataset, argsD, device_index=args.gpu)
    train_dataloader = make_pipeline(dataset, args, device_index=args.gpu)
    nExpectedBatches = len(dataset) // args.bs

    def saveEmbeddings(embeddingsList, ptID, imgID):
        if len(embeddingsList)==0: return
        embeddings = torch.stack(embeddingsList)
        outPath = f'/fast/rsna-breast/features/{args.tileSet}/{outDir}/{ptID}/'
        outFile = os.path.join(outPath, f'{imgID}.pt')
        #print(outPath)
        if not os.path.exists(outPath):
            try: os.makedirs(outPath)
            except: pass
        #print(type(embeddings))
        torch.save(embeddings, outFile)

    activePtID, activeImgID = None, None
    embeddingsList = []
    for batch in tqdm(train_dataloader, total=nExpectedBatches):
        nInst = batch[0]["images"].shape[0]
        total += nInst
        #print(nInst, batch[0]["images"].shape, batch[0]["imgIDs"].shape)
        #print(batch[0]["images"].device, batch[0]["imgIDs"].device)

        images, ptIDs, imgIDs = batch[0]["images"], batch[0]["ptIDs"], batch[0]["imgIDs"]
        #print(ptIDs, imgIDs)
        #i=images
        #print(type(i))
        #print(i.dtype)
        #print(i.min(), i.max(), i.shape)
        #i = images.detach().cpu().numpy()
        #print(type(i), i.dtype, i.min(), i.mean(), i.max(), i.shape)
        #sys.exit()
        #continue

        # looks reasonable
        #print(images.min(), images.mean(), images.max())
        # tensor(-2.1179, device='cuda:15') tensor(0.6672, device='cuda:15') tensor(2.1520, device='cuda:15')

        emb = encoder(images).detach().cpu()
        #print(images.shape, emb.shape, imgIDs.shape)

        #for image, imgID in zip(images, imgIDs):
        #    print(image.shape, int(imgID))

        for em, ptID, imgID in zip(emb, ptIDs, imgIDs):
            #print(em.shape, imgID)
            if imgID != activeImgID:
                saveEmbeddings(embeddingsList, activePtID, activeImgID)
                activePtID, activeImgID = ptID.item(), imgID.item()
                embeddingsList = []
            embeddingsList.append(em)
        saveEmbeddings(embeddingsList, activePtID, activeImgID)


    print(f'Total instances returned : {total}')











