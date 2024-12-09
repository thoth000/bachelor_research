import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import json
import numpy as np
import cv2
import random

class HorizontalFlip:
    """画像とマスクに確率的に水平方向の反転を適用するクラス"""
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        if np.random.rand() < self.flip_prob:
            sample['image'] = F.hflip(sample['image'])
            sample['mask'] = np.fliplr(sample['mask']).copy()  # NumPyでマスクを反転
        return sample

class RandomRotate:
    """画像とマスクに対してランダムな回転を適用するクラス"""
    def __init__(self, angle_range=(-10, 10)):
        self.angle_range = angle_range

    def __call__(self, sample):
        angle = np.random.uniform(*self.angle_range)
        sample['image'] = F.rotate(sample['image'], angle)
        sample['mask'] = np.array(F.rotate(Image.fromarray(sample['mask']), angle, interpolation=Image.NEAREST))
        return sample

class Resize:
    """画像とマスクをリサイズするクラス"""
    def __init__(self, size):
        self.size = size
        self.resize_image = transforms.Resize(size)
        self.resize_mask = transforms.Resize(size, interpolation=Image.NEAREST)

    def __call__(self, sample):
        sample['transformed_image'] = self.resize_image(sample['image'])
        sample['transformed_mask'] = np.array(self.resize_mask(Image.fromarray(sample['mask'])))
        return sample

class RandomResizedCrop:
    """画像とマスクにランダムなリサイズとクロップを適用するクラス"""
    def __init__(self, size, scale=(0.5, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, sample):
        i, j, h, w = self.get_params(sample['image'], self.scale)
        sample['transformed_image'] = F.resized_crop(sample['image'], i, j, h, w, self.size)
        sample['transformed_mask'] = np.array(F.resized_crop(Image.fromarray(sample['mask']), i, j, h, w, self.size, interpolation=Image.NEAREST))
        return sample

    @staticmethod
    def get_params(img, scale):
        width, height = F.get_image_size(img)
        area = height * width
        target_area = random.uniform(*scale) * area
        w = int(round(np.sqrt(target_area)))
        h = int(round(np.sqrt(target_area)))
        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
        else:  # フォールバック用の中心クロップ
            i = (height - h) // 2
            j = (width - w) // 2
        return i, j, h, w

class ToTensor:
    """画像とマスクをテンソルに変換するクラス"""
    def __call__(self, sample):
        sample['image'] = transforms.ToTensor()(sample['image'])
        sample['mask'] = torch.from_numpy(sample['mask']).long()
        sample['transformed_image'] = transforms.ToTensor()(sample['transformed_image'])
        sample['transformed_mask'] = torch.from_numpy(sample['transformed_mask']).long()
        return sample

class DRIVEDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))  # 画像ファイル名のリストを取得
        self.mask_filenames = sorted(os.listdir(mask_dir))    # マスクファイル名のリストを取得

    def __len__(self):
        return len(self.image_filenames)  # データセットのサイズを返す

    def __getitem__(self, idx):
        # 画像とマスクのパスを取得
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # 画像とマスクを読み込む
        image = Image.open(image_path).convert("RGB")  # 画像をRGBで読み込む
        mask = Image.open(mask_path).convert("L")      # マスクをグレースケールで読み込む

        sample = {'image': image, 'mask': mask, 'meta': {'img_path': image_path, 'mask_path': mask_path}}
        
        if self.transform:
            sample = self.transform(sample)

        return sample