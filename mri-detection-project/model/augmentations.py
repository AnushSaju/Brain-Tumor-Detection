# model/augmentations.py
"""
Augmentation pipeline for MRI tumor detection dataset.
We use albumentations for strong and reproducible image augmentation.
"""

from albumentations import (
    Compose, Resize, HorizontalFlip, VerticalFlip, RandomRotate90,
    ShiftScaleRotate, RandomBrightnessContrast, Normalize
)
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size: int = 224):
    """
    Returns a transform pipeline for training images.
    Args:
        img_size (int): Final size to resize the image.
    """
    return Compose([
        Resize(img_size, img_size),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        RandomBrightnessContrast(p=0.3),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_valid_transforms(img_size: int = 224):
    """
    Returns a transform pipeline for validation/test images.
    """
    return Compose([
        Resize(img_size, img_size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
