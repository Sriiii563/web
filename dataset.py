import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CULaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list_file, transform=None):
        """
        :param image_dir: path to CULane/images
        :param mask_dir: path to CULane/labels
        :param image_list_file: txt file with relative paths (e.g., train.txt)
        :param transform: albumentations transforms
        """
        with open(image_list_file, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.samples[idx])
        mask_path = os.path.join(self.mask_dir, self.samples[idx].replace('.jpg', '.png'))

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()
