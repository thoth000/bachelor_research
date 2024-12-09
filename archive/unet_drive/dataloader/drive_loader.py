import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

import scipy.ndimage as ndi
import cv2
from PIL import Image
import numpy as np
import random

import albumentations as A

class HorizontalFlip:
    """画像とマスクに確率的に水平方向の反転を適用するクラス"""
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        if np.random.rand() < self.flip_prob:
            sample['image'] = F.hflip(sample['image'])
            sample['mask'] = F.hflip(sample['mask'])
        return sample

class RandomRotate:
    """画像とマスクに対してランダムな回転を適用するクラス"""
    def __init__(self, angle_range=(-10, 10)):
        self.angle_range = angle_range

    def __call__(self, sample):
        angle = np.random.uniform(*self.angle_range)
        sample['image'] = F.rotate(sample['image'], angle)
        sample['mask'] = F.rotate(sample['mask'], angle, interpolation=Image.NEAREST)
        return sample

class RandomShift:
    """画像とマスクにランダムなシフト（平行移動）を適用するクラス"""
    def __init__(self, max_shift=(0.2, 0.2)):
        # 水平と垂直方向の最大シフト比率
        self.max_shift = max_shift

    def __call__(self, sample):
        image_width, image_height = F.get_image_size(sample['image'])
        max_dx = self.max_shift[0] * image_width
        max_dy = self.max_shift[1] * image_height
        dx = np.random.uniform(-max_dx, max_dx)
        dy = np.random.uniform(-max_dy, max_dy)
        
        # 画像とマスクを同じ量だけシフト
        sample['image'] = F.affine(sample['image'], angle=0, translate=(dx, dy), scale=1.0, shear=0)
        sample['mask'] = F.affine(sample['mask'], angle=0, translate=(dx, dy), scale=1.0, shear=0, interpolation=Image.NEAREST)
        return sample

class RandomScale:
    """画像とマスクにランダムなスケールを適用するクラス"""
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, sample):
        scale = np.random.uniform(*self.scale_range)
        sample['image'] = F.affine(sample['image'], angle=0, translate=(0, 0), scale=scale, shear=0)
        sample['mask'] = F.affine(sample['mask'], angle=0, translate=(0, 0), scale=scale, shear=0, interpolation=Image.NEAREST)
        return sample

class ElasticTransform:
    """画像とマスクに弾性変形を適用するクラス"""
    def __init__(self, alpha=34, sigma=4):
        self.alpha = alpha  # 変形の強さ
        self.sigma = sigma  # ガウシアンフィルタの標準偏差

    def __call__(self, sample):
        random_state = np.random.RandomState(None)
        
        # 画像とマスクの形状を取得
        shape = sample['image'].size[::-1]  # (高さ, 幅) の形状
        image_np = np.array(sample['image'])
        mask_np = np.array(sample['mask'])

        # 変形フィールドを生成
        dx = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

        # 変形後の座標を生成
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.array([y + dy, x + dx])

        # map_coordinatesに適した形状に変更
        distorted_image = ndi.map_coordinates(image_np, indices, order=1, mode='reflect')
        distorted_mask = ndi.map_coordinates(mask_np, indices, order=0, mode='reflect')

        # 変形後の結果をPIL Imageに変換
        sample['image'] = Image.fromarray(distorted_image.astype(np.uint8))
        sample['mask'] = Image.fromarray(distorted_mask.astype(np.uint8))

        return sample

class HistogramEqualization:
    """画像にヒストグラム平坦化を適用するクラス"""
    def __call__(self, sample):
        image = np.array(sample['image'])
        for i in range(image.shape[-1]):
            image[..., i] = cv2.equalizeHist(image[..., i])
        sample['image'] = Image.fromarray(image)
        return sample

class RandomBrightnessContrast:
    """画像にランダムな明るさとコントラストの調整を適用するクラス"""
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, sample):
        brightness = np.random.uniform(*self.brightness_range)
        contrast = np.random.uniform(*self.contrast_range)
        
        # 明るさとコントラストの調整
        sample['image'] = F.adjust_brightness(sample['image'], brightness)
        sample['image'] = F.adjust_contrast(sample['image'], contrast)
        return sample

class RandomGamma:
    """画像にランダムなガンマ変換を適用するクラス"""
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range

    def __call__(self, sample):
        gamma = np.random.uniform(*self.gamma_range)
        sample['image'] = F.adjust_gamma(sample['image'], gamma)
        return sample

class FullTransform:
    """すべての変換をまとめたクラス"""
    def __init__(self):
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10),
            # A.RandomScale(scale_limit=0.2),
            # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
            A.GridDistortion(),
            A.OpticalDistortion(),
            A.GaussNoise(),
            A.RandomGamma(),
            A.Equalize(),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16),
            A.Sharpen(),
            A.GaussianBlur(blur_limit=3),
            A.RandomBrightnessContrast(),
        ])

    def __call__(self, sample):
        # PIL.Image を NumPy 配列に変換
        image_np = np.array(sample['image'])
        mask_np = np.array(sample['mask'])

        # Albumentations での変換を適用
        augmented = self.transforms(image=image_np, mask=mask_np)

        # 変換後の NumPy 配列を PIL.Image に戻す
        sample['image'] = Image.fromarray(augmented['image'])
        sample['mask'] = Image.fromarray(augmented['mask'])

        return sample

class Resize:
    """画像とマスクをリサイズするクラス"""
    def __init__(self, size):
        self.size = size
        self.resize_image = transforms.Resize(size)
        self.resize_mask = transforms.Resize(size, interpolation=Image.NEAREST)

    def __call__(self, sample):
        sample['transformed_image'] = self.resize_image(sample['image'])
        sample['transformed_mask'] = self.resize_mask(sample['mask'])
        
        return sample


class ToTensor:
    """画像とマスクをテンソルに変換するクラス"""
    def __call__(self, sample):
        sample['image'] = transforms.ToTensor()(sample['image'])
        sample['mask'] = transforms.ToTensor()(sample['mask'])
        sample['transformed_image'] = transforms.ToTensor()(sample['transformed_image'])
        sample['transformed_mask'] = transforms.ToTensor()(sample['transformed_mask'])
        
        return sample

class DRIVEDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))  # 画像ファイル名のリストを取得
        self.mask_filenames = sorted(os.listdir(mask_dir))    # マスクファイル名のリストを取得
        
        if len(self.image_filenames) != len(self.mask_filenames):
            raise ValueError(f"Number of images ({len(self.image_filenames)}) and masks ({len(self.mask_filenames)}) do not match.")

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

# For Test

class Resize_Test:
    """画像とマスクをリサイズするクラス"""
    def __init__(self, size):
        self.size = size
        self.resize_image = transforms.Resize(size)

    def __call__(self, sample):
        sample['transformed_image'] = self.resize_image(sample['image'])
        
        return sample

class ToTensor_Test:
    """画像とマスクをテンソルに変換するクラス"""
    def __call__(self, sample):
        sample['image'] = transforms.ToTensor()(sample['image'])
        sample['transformed_image'] = transforms.ToTensor()(sample['transformed_image'])
        
        return sample
