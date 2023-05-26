import os
from argparse import ArgumentParser
from random import shuffle

import numpy as np
import torch
from PIL import Image
from glob import glob

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from autoencoder import AutoencoderModel
from torchvision import transforms as T

from models import Model

'''
normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}
'''

normalize = dict(mean=[0.5], std=[0.25])

transform = [
    T.ToTensor(),
    T.Normalize(mean=normalize['mean'], std=normalize['std'])
]

transform = T.Compose(transform)

def norm(i):
    i = i-i.min()
    i = i/i.max()
    return i


mod1 = '/fast/rsna-breast/checkpoints/tileAutoencoder/zany-sweep-136/epoch=47-step=24197.ckpt'

#[T.ToPILImage(), T.Resize(224), T.ToTensor()]
invTransform = T.ToPILImage()


class MammogramDataset(Dataset):

    def __init__(self, args):
        imgFiles = glob('/fast/rsna-breast/pngfull/*/*.png')
        self.imgFiles = []
        for f in imgFiles:
            outFile = f.replace('pngfull', f'colorized/{args.out}')
            if not os.path.exists(outFile):
                self.imgFiles.append(f)

    def __len__(self):
        return len(self.imgFiles)

    def __getitem__(self, item):
        f = self.imgFiles[item]
        image = Image.open(f)
        tensor = transform(image)
        return tensor, f


def collate(batch):
    tensors, files = batch
    tensors = torch.stack(tensors)
    return tensors, files


if __name__ == '__main__':
    parser = ArgumentParser()
    #parser.add_argument("-model", default=mod1)
    parser.add_argument("-model", default='/fast/rsna-breast/checkpoints/tileClassifier/deit3_small_patch16_224_sweepy-sweep-45/epoch=83-step=21270.ckpt')
    #parser.add_argument("png")
    parser.add_argument("--levels", default=2, type=int)
    parser.add_argument("--kernel", default=5, type=int)
    parser.add_argument("--filters", default=8, type=int)
    #parser.add_argument("-out", default='zany-sweep-136')
    parser.add_argument("-out", default='sweepy-sweep-45')
    parser.add_argument("-colorize", default=1)
    parser.add_argument("-pool", default=0)
    parser.add_argument("-encoder", default='deit3_small_patch16_224')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #imgFile = glob('/fast/rsna-breast/pngfull/*/*.png')[0]



    #model = AutoencoderModel(args)      # zany sweep
    model = Model(args)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(args.model)['state_dict'])


    dataset = MammogramDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=True)

    for batch in tqdm(dataloader):
        imgs, files = batch
        #print(imgs.shape)
        tensor = imgs#[0]
        imgFile = files[0]

        outFile = imgFile.replace('pngfull', f'colorized/{args.out}')
        if os.path.exists(outFile):
            continue

        print('tensor shape', tensor.shape)
        #output = model.encoder(tensor.to(device))      # zany sweep
        output = model.colorizer(tensor.to(device))[0]       # sweepy sweep
        output = torch.permute(output, (1,2,0))
        output = output.cpu().detach().numpy()
        output = (norm(output)*255.0).astype(np.uint8)
        #print(output.shape)
        outPil = Image.fromarray(output)

        p, fn = os.path.split(outFile)
        if not os.path.exists(p):
            os.makedirs(p)
        #outPil.save('/fast/rsna-breast/test.png')
        outPil.save(outFile)


    '''
    def runImg(imgFile):
        outFile = imgFile.replace('pngfull', f'colorized/{args.out}')
        if os.path.exists(outFile):
            return

        image = Image.open(imgFile)
        tensor = transform(image)

        output = model.encoder(tensor.to(device))
        output = torch.permute(output, (1,2,0))
        output = output.cpu().detach().numpy()
        output = (norm(output)*255.0).astype(np.uint8)
        #print(output.shape)
        outPil = Image.fromarray(output)

        p, fn = os.path.split(outFile)
        if not os.path.exists(p):
            os.makedirs(p)
        #outPil.save('/fast/rsna-breast/test.png')
        outPil.save(outFile)



    for f in tqdm(imgFiles):
        runImg(f)

    '''