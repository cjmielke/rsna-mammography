import sys
from collections import OrderedDict

import timm
import torch
import torchvision
from torch import nn


class Resnet18Model(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        #print(list(resnet.children()))
        layers = list(resnet.children())[:-2]
        layers.append(nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return x


class ColorizerModel(nn.Module):
    def __init__(self, k=5, n=3):
        super().__init__()
        assert n >= 1
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        layers = [nn.Conv2d(1, 3, (k, k), padding='same')]
        for x in range(n - 1):
            layers.append(nn.Conv2d(3, 3, (k, k), padding='same'))
        self.conv = nn.Sequential(*layers)

    def forward(self, bwImage):
        # model basically uses 3 convolutional filters to add residual color information on top of original image
        rgbBase = torch.cat([bwImage, bwImage, bwImage], dim=1)
        C = self.conv(bwImage)
        #return rgbBase, C
        #print(rgbBase.shape, C.shape)
        return rgbBase + C


def getModel(encoder, num_classes=0):
    model = timm.create_model(encoder, pretrained=True, num_classes=num_classes)
    o = model(torch.randn(2, 3, 224, 224))
    print(f'Original shape: {o.shape}')
    assert len(o.shape) == 2
    embSize = o.shape[1]
    return model, embSize


class TileClassifierEncoder(nn.Module):
    def __init__(self, argparse):
        super().__init__()
        self.args = argparse

        if self.args.colorize:
            self.colorizer = ColorizerModel(n=self.args.colorize)

        self.encoder, embSize = getModel(self.args.encoder)

    def forward(self, x):
        if self.args.colorize:
            if x.shape[1]==3:
                #print(x.shape)
                #print('converting RGB tensor to gray')
                x = x[:, 1:2, :, :]
                #print(x.shape)
            x = self.colorizer(x)

        embeddings = self.encoder(x)
        return embeddings


def loadModel(args, device=None):
    if args.encoder == 'resnet18':            # as defined in vicreg.py

        '''
        resnet = torchvision.models.resnet18()
        print(list(resnet.children()))
        layers = list(resnet.children())[:-2]
        layers.append(nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1), bias=False))
        backbone = nn.Sequential(*layers)
        model = backbone
        '''
        model = Resnet18Model()
        o = model(torch.randn(2, 3, 224, 224))
        print(f'Original shape: {o.shape}')
        #o = model.forward_features(torch.randn(2, 3, 224, 224))
        #print(f'Unpooled shape: {o.shape}')
    elif args.encoder == 'nyu':
        from nyuModel import ModifiedDenseNet121
        model = ModifiedDenseNet121(num_classes=4)
        o = model(torch.randn(2, 3, 224, 224))
        print(f'output : {o.shape}')
        #sys.exit()
    else:
        if args.weights is not None and 'tileClassifier' in args.weights:
            model = TileClassifierEncoder(args)
        elif args.weights is not None and 'regionClassifier' in args.weights and 'flowing-sweep' in args.weights:
            model = RegionModel('deit3_small_patch16_224', 64, 5, skipcon=True)
        elif args.weights is not None and 'regionClassifier' in args.weights and 'whole-sweep' in args.weights:
            model = RegionModel('deit3_small_patch16_224', 16, 5, skipcon=True)
        else:
            model = timm.create_model(args.encoder, pretrained=True, num_classes=0)
            ts = args.tileSize
            o = model(torch.randn(2, 3, ts, ts))
            print(f'Original shape: {o.shape}')
            o = model.forward_features(torch.randn(2, 3, ts, ts))
            print(f'Unpooled shape: {o.shape}')

    if args.weights:
        # remove "module." prefix from state dict keys
        state_dict = torch.load(args.weights, map_location=torch.device('cuda'))['state_dict']
        #state_dict = [k.lstrip('module.'):v for k,v in state_dict.items()]
        newDict = OrderedDict()
        for k, v in state_dict.items():
            #if k.startswith('classifier.'): continue
            #k = k.removeprefix('encoder.')          # for tileClassifier
            k = k.removeprefix('backbone.')         # for lightly SSL models
            newDict[k] = v
        missing_keys, unexpected_keys = model.load_state_dict(newDict, strict=False)
        if len(missing_keys) != 0:
            print(f'Missing keys :')
            print(missing_keys)
            print(f'Unexpected keys :')
            print(unexpected_keys)  # these are probably just projection heads from SSL models
            raise ValueError
        print(f'Unexpected keys :')
        print(unexpected_keys)          # these are probably just projection heads from SSL models

    if device: model.to(device)
    model.eval()
    #print(model)
    return model


class RegionModel(nn.Module):
    def __init__(self, encoder, f, k, skipcon=True):
        super().__init__()
        self.skipcon = skipcon

        #k1 = 5
        #self.colorize = nn.Conv2d(1, 3, (k1, k1), padding='same')
        # colorizer is a module that maps a 224*2 sized BW image to RGB 224

        layers = [
            nn.Conv2d(1, f, (k, k), padding='same'),
            nn.ReLU(),
            nn.Conv2d(f, 3, (k, k), padding='same'),
            #nn.ReLU(),
            #nn.MaxPool2d((2, 2)),
            nn.Sigmoid()
        ]
        self.colorizer = nn.Sequential(*layers)
        # encoder is RGB image at 1/4 the size!

        self.classifier, embSize = getModel(encoder, num_classes=0)

    def forward(self, x):
        x = x / 255         # mimmick ToTensor(), unavailable in DALI
        #print(x.min(), x.mean(), x.max())
        #sys.exit()
        #'''
        rgb = self.colorizer(x)

        if self.skipcon:
            #p = torch.max_pool2d(x, 2)
            p = x
            rgbBase = torch.cat([p,p,p], dim=1)
            #print(rgbBase.shape, rgb.shape)
            #sys.exit()
            rgb = rgb + rgbBase
        #'''
        #rgb = torch.cat([x, x, x], dim=1)

        #print(f'rgb shape {rgb.shape}')
        #preds = self.classifier(rgb).squeeze(1)
        #preds = torch.sigmoid(preds)
        #return rgb, preds
        #feats = self.classifier.forward_features(rgb)
        feats = self.classifier(rgb)
        #print(feats.shape)
        #sys.exit()
        return feats
