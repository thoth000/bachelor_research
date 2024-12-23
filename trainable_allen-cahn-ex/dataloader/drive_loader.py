import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt
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
        
        # 画像の次元数に応じたパディングの適用
        if image.ndim == 3:  # カラー画像の場合
            padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=self.fill_value)
        else:  # グレースケール画像の場合
            padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=self.fill_value)
        
        # マスクの次元数に応じたパディングの適用
        if mask.ndim == 3:
            padded_mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=self.fill_value)
        else:
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
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
        ])

    def __call__(self, sample):
        # Albumentations での変換を適用
        augmented = self.transforms(image=sample['image'], mask=sample['mask'])
        sample['image'] = augmented['image']
        sample['mask'] = augmented['mask']

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


class Resize:
    """画像とマスクをnumpy配列でリサイズするクラス"""
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, sample):
        sample['transformed_image'] = cv2.resize(sample['image'], self.size, interpolation=cv2.INTER_LINEAR)
        sample['transformed_mask'] = cv2.resize(sample['mask'], self.size, interpolation=cv2.INTER_NEAREST)
        return sample


class Keep:
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['transformed_image'] = sample['image']
        sample['transformed_mask'] = sample['mask']
        return sample


class ToTensor:
    """画像とマスクをテンソルに変換するクラス"""
    def __call__(self, sample):
        sample['image'] = transforms.ToTensor()(sample['image']).float()
        sample['mask'] = transforms.ToTensor()(sample['mask']).float()
        sample['transformed_image'] = transforms.ToTensor()(sample['transformed_image']).float()
        sample['transformed_mask'] = transforms.ToTensor()(sample['transformed_mask']).float()
        
        return sample


def get_transform(args, mode='training'):
    if mode == 'training':
        transform = transforms.Compose([
            StandardTransform(),
            Keep(),
            ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            PadToSquare(),
            Keep(),
            ToTensor(),
        ])
    
    return transform


class DRIVEDataset(Dataset):
    def __init__(self, mode, path, opt ,is_val=False, split=None, transform=None):
        self.path = path
        self.transform = transform
        self.data_path = os.path.join(path, f"{mode}_{opt}")
        self.data_file = os.listdir(self.data_path)
        self.image_file = self._select_img(self.data_file)
        if split is not None and mode == "training":
            assert split > 0 and split < 1
            if not is_val:
                self.image_file = self.image_file[:int(split*len(self.image_file))]
            else:
                self.image_file = self.image_file[int(split*len(self.image_file)):]

    def __len__(self):
        return len(self.image_file)  # データセットのサイズを返す

    def __getitem__(self, idx):
        image_file = self.image_file[idx]
        with open(file=os.path.join(self.data_path, image_file), mode='rb') as file:
            image = pickle.load(file)
            image = np.transpose(image, (1, 2, 0))
        gt_file = "gt" + image_file[3:]
        with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
            mask = pickle.load(file)
            mask = np.transpose(mask, (1, 2, 0))

        sample = {'image': image, 'mask': mask, 'meta': {'img_path': image_file, 'mask_path': gt_file}}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)

        return img_list