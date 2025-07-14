import os
import shutil
import random

import torch
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import numpy as np

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


class AlbumentationsTransform:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, image):
        # image: PIL Image
        image = np.array(image)
        augmented = self.aug(image=image)
        return augmented['image']


def split_dataset(
        input_dir,
        output_dir,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
        "Ratios must sum to 1."

    random.seed(seed)

    # Get class names (e.g. ["cats", "dogs"])
    classes = [
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ]

    for class_name in classes:
        class_dir = os.path.join(input_dir, class_name)
        images = os.listdir(class_dir)

        # Shuffle images_datasets
        random.shuffle(images)

        # Compute split sizes
        total = len(images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        test_count = total - train_count - val_count

        splits = {
            'train': images[:train_count],
            'val': images[train_count:train_count + val_count],
            'test': images[train_count + val_count:]
        }

        for split_name, split_images in splits.items():
            split_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)

            for img_name in split_images:
                src = os.path.join(class_dir, img_name)
                dst = os.path.join(split_dir, img_name)
                shutil.copy2(src, dst)


split_dataset("images_datasets", "split_images")


class CustomizedDataset(ClassificationDataset):
    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """Initialize a customized classification dataset with enhanced data augmentation transforms."""
        super().__init__(root, args, augment, prefix)

        train_transforms = A.Compose([
            A.Resize(64, 64),
            # A.RandomResizedCrop((56, 56), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2(),
        ])
        train_transforms = AlbumentationsTransform(train_transforms)

        val_transforms = A.Compose([
            A.Resize(64, 64),
            # A.CenterCrop(56, 56),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2(),
        ])
        val_transforms = AlbumentationsTransform(val_transforms)
        self.torch_transforms = train_transforms if augment else val_transforms


class CustomizedTrainer(ClassificationTrainer):
    """A customized trainer class for YOLO classification models with enhanced dataset handling."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Build a customized dataset for classification training and the validation during training."""
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)


class CustomizedValidator(ClassificationValidator):
    """A customized validator class for YOLO classification models with enhanced dataset handling."""

    def build_dataset(self, img_path: str, mode: str = "val"):
        """Build a customized dataset for classification standalone validation."""
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=self.args.split)
