import sys
import time

import numpy as np
import pandas as pd
import pydicom
import glob, os
import pydicom
from pydicom.filebase import DicomBytesIO
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.types import DALIDataType



'''
#here's the magic of hacking the jpeg2000 encoded bitstream.  Function saves jp2 encoded image files contained within dicom which will be later decoded by DALI
def convert_dicom_to_j2k(file):
    patient = file.split('/')[-2]
    image = file.split('/')[-1][:-4]
    dcmfile = pydicom.dcmread(f'../input/rsna-breast-cancer-detection/train_images/{file}')
    if dcmfile.file_meta.TransferSyntaxUID=='1.2.840.10008.1.2.4.90':
        with open(f'../input/rsna-breast-cancer-detection/train_images/{file}', 'rb') as fp:
            raw = DicomBytesIO(fp.read())
            ds = pydicom.dcmread(raw)
        offset = ds.PixelData.find(b"\x00\x00\x00\x0C")  #<---- the jpeg2000 header info we're looking for
        hackedbitstream = bytearray()
        hackedbitstream.extend(ds.PixelData[offset:])
        with open(f"../working/{patient}_{image}.jp2", "wb") as binary_file:
            binary_file.write(hackedbitstream)
'''

def convert_dicom_to_j2k(inFile):
    fp, fn = os.path.split(inFile)
    _, ptID = os.path.split(fp)
    imgID = fn.rstrip('.dcm')
    outPath = os.path.join('/fast/rsna-breast/jp2/', ptID)
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    outFile = os.path.join(outPath, f'{imgID}.jp2')
    dcmfile = pydicom.dcmread(inFile)
    if dcmfile.file_meta.TransferSyntaxUID=='1.2.840.10008.1.2.4.90':
        with open(inFile, 'rb') as fp:
            raw = DicomBytesIO(fp.read())
            ds = pydicom.dcmread(raw)
        offset = ds.PixelData.find(b"\x00\x00\x00\x0C")  #<---- the jpeg2000 header info we're looking for
        hackedbitstream = bytearray()
        hackedbitstream.extend(ds.PixelData[offset:])
        with open(outFile, "wb") as binary_file:
            binary_file.write(hackedbitstream)



#allj2kdicoms = df[df['uid']=='1.2.840.10008.1.2.4.90']['dicom'].tolist()


#j2kfiles = [f'../working/{thing.split("/")[-2]}_{thing.split("/")[-1][:-4]}.jp2' for thing in allj2kdicoms[:32]]



testFile = '/fast/rsna-breast/train_images/10025/1365269360.dcm'            # confirmed a good file
#convert_dicom_to_j2k(testFile)                                             # SUCCESS!



def pydicom_benchmark(file):
    print(f'reading {file}')
    dicom = pydicom.dcmread(file)
    img = dicom.pixel_array
    return img

start = time.time()
#img1 = pydicom_benchmark(testFile)
#img1 = pydicom_benchmark(testFile)
print(f'took {time.time()-start} seconds for pydicom')
#print(img1.shape)




start = time.time()


testJP2 = '/fast/rsna-breast/jp2/10025/1365269360.jp2'

NUM_DCM = 32

@pipeline_def
def j2k_decode_pipeline():
    jpegs, _ = fn.readers.file(files=[testJP2]*NUM_DCM)
    images = fn.experimental.decoders.image(jpegs, device='mixed', output_type=types.ANY_DATA, dtype=DALIDataType.UINT16)
    return images

max_batch_size = NUM_DCM
pipe = j2k_decode_pipeline(batch_size=max_batch_size, num_threads=2, device_id=0, debug=True)
pipe.build()

pipe_out = pipe.run()           # pipe_out is a tuple

print(type(pipe_out))
print(len(pipe_out))

#batch, idxs = pipe_out
batch = pipe_out[0]
#images = batch.as_cpu().as_array()
#print(images.shape, '='*10)
#for img2 in images:
#    print(img2.shape)

print(f'took {time.time()-start} seconds for dali')
print(f'thats {(time.time()-start)/NUM_DCM} per dicom')


print(img2.shape)
img2 = img2[:,:,0]
print((img1-img2).sum())
print(img1==img2)


batch, idxs = pipe_out

print(idxs)