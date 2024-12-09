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

# 自作の設定ファイルから設定を読み込み
from myconfig import config

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

class_mapping = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
}

class CityscapesDataset(Dataset):
    """Cityscapesデータセットを読み込むクラス"""
    def __init__(self, root, split='train', transform=None):
        self.images_dir = os.path.join(root, split)
        self.mask_dir = self.images_dir.replace('leftImg8bit', 'gtFine')
        self.transform = transform
        self.images, self.masks = self.load_image_mask_paths()

    def load_image_mask_paths(self):
        images, masks = [], []
        for city in os.listdir(self.images_dir):
            city_images = os.listdir(os.path.join(self.images_dir, city))
            for image in city_images:
                img_path = os.path.join(self.images_dir, city, image)
                mask_path = os.path.join(self.mask_dir, city, image.replace('_leftImg8bit.png', '_gtFine_polygons.json'))
                images.append(img_path)
                masks.append(mask_path)
        return images, masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        original_image = Image.open(img_path).convert("RGB")
        original_mask = self.load_mask(mask_path)

        sample = {'image': original_image, 'mask': original_mask, 'meta': {'img_path': img_path, 'mask_path': mask_path}}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def load_mask(self, mask_path):
        with open(mask_path) as f:
            mask_json = json.load(f)
        mask = np.ones((mask_json['imgHeight'], mask_json['imgWidth']), dtype=np.uint8) * 255
        for obj in mask_json['objects']:
            label = self.get_label_id(obj['label'])
            if label != 255:  # 無視クラスを除外
                polygon = np.array(obj['polygon'], dtype=np.int32)
                cv2.fillPoly(mask, [polygon], label)
        return mask
    
    def get_label_id(self, label):
        label_map = {
            "unlabeled": 0, "ego vehicle": 1, "rectification border": 2, "out of roi": 3,
            "static": 4, "dynamic": 5, "ground": 6, "road": 7, "sidewalk": 8, "parking": 9,
            "rail track": 10, "building": 11, "wall": 12, "fence": 13, "pole": 17,
            "traffic light": 19, "traffic sign": 20, "vegetation": 21, "terrain": 22,
            "sky": 23, "person": 24, "rider": 25, "car": 26, "truck": 27, "bus": 28,
            "train": 31, "motorcycle": 32, "bicycle": 33
        }
        original_label = label_map.get(label, 255)  # 未定義ラベルは255を返す
        return class_mapping.get(original_label, 255)

def get_dataloader(split='train'):
    if split == 'train':
        transform = transforms.Compose([
            RandomRotate(angle_range=(-10, 10)),  # ランダム回転
            HorizontalFlip(flip_prob=0.5),        # 水平反転
            RandomResizedCrop(size=(256, 512)),   # ランダムクロップとリサイズ
            ToTensor()                            # テンソルに変換
        ])
    else:
        transform = transforms.Compose([
            Resize(size=(256, 512)),              # リサイズ
            ToTensor()                            # テンソルに変換
        ])

    dataset = CityscapesDataset(root=config[f"{split}_data"], split=split, transform=transform)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])


def get_dataloader(split='train'):
    """transform = transforms.Compose([
        RandomRotate(angle_range=(-10, 10)),
        HorizontalFlip(flip_prob=0.5),
        Resize(size=(512, 1024)),
        ToTensor()
    ])"""
    
    if split=='train':
        transform = transforms.Compose([
            RandomRotate(angle_range=(-10, 10)),  # ランダム回転
            HorizontalFlip(flip_prob=0.5),        # 水平反転
            RandomResizedCrop(size=(256, 512)),  # ランダムクロップとリサイズ
            ToTensor()                            # テンソルに変換
        ])
    else:
        transform = transforms.Compose([
            Resize(size=(256, 512)),             # 固定サイズにリサイズ
            ToTensor()                            # テンソルに変換
        ])

    if split == "train":
        dataset = CityscapesDataset(root=config["train_data"], split=split, transform=transform)
        return DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    elif split == "val":
        dataset = CityscapesDataset(root=config["val_data"], split=split, transform=transform)
        return DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    elif split == "test":
        dataset = CityscapesDataset(root=config["test_data"], split=split, transform=transform)
        return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config['num_workers'])
