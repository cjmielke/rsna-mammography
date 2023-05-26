import dicomsdl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T
from PIL import Image



transform = T.Compose([
    T.RandomAffine(90, shear=(-20.0, 20.0), scale=(0.5, 1.5)),
    T.RandomCrop((4096, 3000), pad_if_needed=True),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.25])
])

class DicomDataset(Dataset):
    def __init__(self, df, transform=transform):
        self.transform = transform
        self.df = df
        #self.df = pd.read_csv('/fast/rsna-breast/train.csv').sample(frac=1).reset_index()

    def __len__(self):
        return len(self.df)

    def crop(self, img, f=2):
        #img = np.asarray(pil)

        av = np.mean(img, axis=0)
        mi = np.min(img, axis=0)
        ma = np.max(img, axis=0)
        img = img[:, (((av - mi) > f) + ((av - ma) > f))]

        av = np.mean(img, axis=1)
        mi = np.min(img, axis=1)
        ma = np.max(img, axis=1)
        img = img[(((av - mi) > f) + ((av - ma) > f)), :]

        return img
        #return Image.fromarray(img)

    def loadDCM(self, f):
        #patient = f.split('/')[-2]
        #image_name = f.split('/')[-1][:-4]

        dicom = dicomsdl.open(f)
        img = dicom.pixelData()
        img = (img - img.min()) / (img.max() - img.min())
        if dicom.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
            img = 1 - img

        image = (img * 255).astype(np.uint8)

        # img = cv2.resize(image, (size, size))

        return image

        #file_name = f'{save_folder}' + f"{patient}_{image_name}.{extension}"
        #cv2.imwrite(file_name, img)

    def __getitem__(self, item):
        R = self.df.iloc[item]
        fn = f'/kaggle/input/rsna-breast-cancer-detection/test_images/{int(R.patient_id)}/{int(R.image_id)}.dcm'
        img = self.loadDCM(fn)
        img = self.crop(img)
        pil = Image.fromarray(img)
        return pil
        tensor = self.transform(pil)
        return tensor, R.cancer, R


df = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/test.csv')
dataset = DicomDataset(df)

