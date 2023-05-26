import argparse

import timm
import torch
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("-gpu", default=1, type=str)
parser.add_argument("-bs", default=32, type=int)
parser.add_argument("-imgSize", default=224)
parser.add_argument("-encoder", default='xcit_nano_12_p16_224_dist')
#parser.add_argument("-encoder", default='resnet18')
#parser.add_argument("-minAttn", default=None, help='If set, only train on images with this minimum attention score')


def getTimmModel(encoder):
    model = timm.create_model(encoder, pretrained=True, num_classes=0)
    o = model(torch.randn(2, 3, 224, 224))
    print(f'Original shape: {o.shape}')
    #o = model.forward_features(torch.randn(2, 3, 224, 224))
    #print(f'Unpooled shape: {o.shape}')
    return model, o.shape[1]


def wandb_log(**kwargs):
    for k, v in kwargs.items():
        wandb.log({k: v})
