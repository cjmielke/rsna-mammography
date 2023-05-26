from albumentations import Compose, Resize, HorizontalFlip, VerticalFlip, Normalize
from albumentations.pytorch import ToTensorV2

from config import Config


class Augments:
    """
    Contains Train, Validation Augments
    """
    train_augments = Compose([
        Resize(*Config['IMG_SIZE'], p=1.0),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(p=1.0),
    ],p=1.)

    valid_augments = Compose([
        Resize(*Config['IMG_SIZE'], p=1.0),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(p=1.0),
    ], p=1.)




