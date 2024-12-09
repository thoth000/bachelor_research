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

class PadToSquare:
    """画像とマスクを正方形のnumpy配列にパディングするクラス"""
    def __init__(self, fill_value=0):
        self.fill_value = fill_value

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        h, w = image.shape[:2]
        max_side = max(h, w)
        
        # パディングの量を計算して保存
        pad_top = (max_side - h) // 2
        pad_bottom = max_side - h - pad_top
        pad_left = (max_side - w) // 2
        pad_right = max_side - w - pad_left

        sample['padding'] = (pad_top, pad_bottom, pad_left, pad_right)
        
        # パディングの適用
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=self.fill_value)
        padded_mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=self.fill_value)
        
        sample['image'] = padded_image
        sample['mask'] = padded_mask
        
        return sample

def unpad_to_original(output, padding):
    # ネットワークの出力からパディングを解除し、元のサイズに戻す関数。
    pad_top, pad_bottom, pad_left, pad_right = padding

    if isinstance(output, torch.Tensor):
        # テンソルの場合
        unpadded_output = output[:, pad_top:output.shape[1]-pad_bottom, pad_left:output.shape[2]-pad_right]
    else:
        # NumPy 配列の場合
        unpadded_output = output[pad_top:output.shape[0]-pad_bottom, pad_left:output.shape[1]-pad_right]
    
    return unpadded_output


class StandardTransform:
    """すべての変換をまとめたクラス"""
    def __init__(self):
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10),
            # A.RandomScale(scale_limit=0.2),
            # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0),
            A.ElasticTransform(alpha=1, sigma=50),
            A.GridDistortion(),
            A.OpticalDistortion(),
            A.GaussNoise(),
            A.RandomGamma(),
            A.Equalize(),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16),
            A.Sharpen(),
            A.GaussianBlur(blur_limit=(3,7)),
            A.RandomBrightnessContrast(),
        ])

    def __call__(self, sample):
        # Albumentations での変換を適用
        augmented = self.transforms(image=sample['image'], mask=sample['mask'])

        # 変換後の NumPy 配列を PIL.Image に戻す
        sample['image'] = augmented['image']
        sample['mask'] = augmented['mask']

        return sample


class Resize:
    """画像とマスクをnumpy配列でリサイズするクラス"""
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, sample):
        sample['transformed_image'] = cv2.resize(sample['image'], self.size, interpolation=cv2.INTER_LINEAR)
        sample['transformed_mask'] = cv2.resize(sample['mask'], self.size, interpolation=cv2.INTER_NEAREST)
        return sample


class ConvertToGrayscale:
    """カラー画像をグレースケールに変換するクラス"""
    def __call__(self, sample):
        sample['image'] = np.array(sample['image'].convert("L"))
        sample['mask'] = np.array(sample['mask'])
        return sample


class Normalize:
    """正規化するクラス"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['image'] = (sample['image'] / 255.0 - self.mean) / self.std
        sample['mask'] = sample['mask'] / 255.0
        return sample


class AlbumentationsTransform:
    """Albumentationsでデータ拡張（フリップと回転）を行うクラス"""
    def __init__(self):
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0)
        ])

    def __call__(self, sample):
        augmented = self.transforms(image=sample['transformed_image'], mask=sample['transformed_mask'])
        sample['transformed_image'] = augmented['image']
        sample['transformed_mask'] = augmented['mask']
        return sample


class ProbabilisticPatchExtraction:
    """画像とマスクから確率的に48x48のパッチを抽出し、transformed_imageとtransformed_maskを生成するクラス"""
    def __init__(self, patch_size=48):
        self.patch_size = patch_size

    def __call__(self, sample):
        h, w = sample['image'].shape[:2]
        if h < self.patch_size or w < self.patch_size:
            raise ValueError("Image dimensions must be larger than the patch size.")

        # ランダムな位置からパッチを抽出
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        sample['transformed_image'] = sample['image'][top:top + self.patch_size, left:left + self.patch_size]
        sample['transformed_mask'] = sample['mask'][top:top + self.patch_size, left:left + self.patch_size]
        
        return sample


class ToTensor:
    """画像とマスクをテンソルに変換するクラス"""
    def __call__(self, sample):
        sample['image'] = transforms.ToTensor()(sample['image']).float()
        sample['mask'] = transforms.ToTensor()(sample['mask']).float()
        sample['transformed_image'] = transforms.ToTensor()(sample['transformed_image']).float()
        sample['transformed_mask'] = transforms.ToTensor()(sample['transformed_mask']).float()
        
        return sample


def get_transform(args, mode='train'):
    if args.transform == 'fr_unet':
        num_channels = 1
        if mode == 'train':
            transform = transforms.Compose([
                ConvertToGrayscale(),
                Normalize(mean=0.5, std=0.5), # [-1, 1]に正規化
                ProbabilisticPatchExtraction(patch_size=48),
                AlbumentationsTransform(),
                ToTensor()
            ])
        else:
            transform = transforms.Compose([
                ConvertToGrayscale(),
                Normalize(mean=0.5, std=0.5), # [-1, 1]に正規化
                Resize(size=args.resolution), # (512, 512)
                ToTensor()
            ])
    else:
        num_channels = 3
        if mode == 'train':
            transform = transforms.Compose([
                PadToSquare(),
                StandardTransform(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Resize(size=args.resolution), # (512, 512)
                ToTensor()
            ])
        else:
            transform = transforms.Compose([
                PadToSquare(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                Resize(size=args.resolution),
                ToTensor(),
            ])
    
    return transform, num_channels


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
        image = np.array(Image.open(image_path).convert("RGB"))  # 画像をRGBで読み込む
        mask = np.array(Image.open(mask_path).convert("L"))      # マスクをグレースケールで読み込む

        sample = {'image': image, 'mask': mask, 'meta': {'img_path': image_path, 'mask_path': mask_path}}
        
        if self.transform:
            sample = self.transform(sample)

        return sample