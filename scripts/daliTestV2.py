# taken from https://practical.codes/dali/cv/image_processing/2020/11/10/nvidia_dali.html#The-Dataloader
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from torch.utils.data import IterableDataset
import numpy as np
import os
from glob import glob

from tqdm import tqdm


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
    def __init__(self, base_path=None, **kwargs):
        super().__init__()
        self.files = os.scandir(base_path)
        self.files = glob("/fast/rsna-breast/tiles-jp2/224/28624/*.jp2")
        #self.files = glob("/fast/rsna-breast/tiles-jp2/224/*/*.jp2")
        print(f'{len(self.files)} files found')

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for fil in self.files:
            #if fil.name.endswith("jp2"): #or other supported types
            if fil.endswith("jp2"):
                image_path = fil#.path
                f = open(image_path, "rb")
                image = np.frombuffer(f.read(), dtype=np.uint8) # don't decode the image just read the bytes!
                ptID, imgID = getPtImgIDs(fil)
                ptID = np.array([ptID])  # some label
                imgID = np.array([imgID])  # some label
                image_path = [ord(x) for x in image_path] #this is a hacky trick used to pass image_paths(strings) through DALI.
                image_path = np.array(image_path, dtype=np.int32)
                #yield image, ptID, image_path
                yield image, ptID, imgID



from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

#Using ExternalSource
class SimplePipeline(Pipeline):
    #Define the operations in the pipeline
    def __init__(self, external_datasource, max_batch_size=16, num_threads=2, device_id=0, resolution=224, crop=224, is_train=True):
        super(SimplePipeline, self).__init__(max_batch_size, num_threads, device_id, seed=12)
        # Define Input nodes
        self.jpegs = ops.ExternalSource()
        self.in_ptIDs = ops.ExternalSource()
        self.in_imgIDs = ops.ExternalSource()
        ## Or pass source straight to ExternalSource this way you won't have do iter_setup.
        # self.jpegs,self.labels,self.paths=ops.ExternalSource(source=self.make_batch, num_outputs=3)

        # Define ops
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_x=resolution, resize_y=resolution)
        self.normalize = ops.CropMirrorNormalize(device="gpu", dtype=types.FLOAT, output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.path_pad = ops.Pad(fill_value=ord("?"),axes = (0,)) # We need to pad image_paths because we need the shapes to match.need dense tensor

        self.iterator = iter(external_datasource)

    # The external source should be fed batches
    # I prefer to batch-ify things here because it keeps things compatible with an IterableDataset
    def make_batch(self):
        imgs = []
        ptIDs = []
        imgIDs = []
        for _ in range(self.max_batch_size):
            try:
                i,l,p = next(self.iterator)
            except StopIteration:
                break
            imgs.append(i)
            ptIDs.append(l)
            imgIDs.append(p)
        if len(imgs)==0:
            raise StopIteration
        return (imgs,ptIDs, imgIDs)

    # How the operations in the pipeline are used
    # Connect your input nodes to your ops
    def define_graph(self):

        self.images = self.jpegs()
        self.labels = self.in_ptIDs()
        self.paths = self.in_imgIDs()
        images = self.decode(self.images)
        images = self.res(images)
        images = self.normalize(images)

        paths = self.path_pad(self.paths)

        return (images, self.labels, paths)

    # Only needed when using ExternalSource
    # Connect the dataset outputs to external Sources
    def iter_setup(self):
        (images,labels,paths) = self.make_batch()
        self.feed_input(self.images, images)
        self.feed_input(self.labels, labels)
        self.feed_input(self.paths, paths)





from nvidia.dali.plugin.pytorch import DALIGenericIterator

def make_pipeline(dataset, args, device_index=0, num_threads=1, is_train=False):
    return_keys = ["images", "labels", "image_path"]
    pipeline = SimplePipeline(dataset, max_batch_size=args["batch_size"], num_threads=num_threads,
                        device_id=device_index, resolution=args["resolution"], crop=args["crop"], is_train=is_train)
    pipeline_iterator = DALIGenericIterator([pipeline], return_keys,
                                            last_batch_policy=LastBatchPolicy.PARTIAL, dynamic_shape=True, last_batch_padded = True
                                            )
    return pipeline_iterator

args = {
    "resolution": 224,
    "crop":224,
    "batch_size": 128,
    #"max_batch_size": 128,
    #"image_folder": "/fast/rsna-breast/tiles-jp2/224/28624" # Change this
    "image_folder": "/fast/rsna-breast/tiles-jp2/224/28624"  # Change this
}


total = 0
dataset = DALIDataset(base_path=args["image_folder"])
train_dataloader = make_pipeline(dataset, args)
nExpectedBatches = len(dataset)//args['batch_size']
for batch in tqdm(train_dataloader, total=nExpectedBatches):
    nInst = batch[0]["images"].shape[0]
    print(nInst, batch[0]["images"].shape,batch[0]["labels"].shape,batch[0]["image_path"].shape)
    print(batch[0]["images"].device,batch[0]["labels"].device,batch[0]["image_path"].device)
    total += nInst
    # It is always batch[0]
    # The dictionary keys are named by return_keys arg.


print(f'Total instances returned : {total}')









