from __future__ import division

import os
import numpy as np
import cv2
import PIL
# from scipy.misc import imresize

from dataloaders.helpers import *
import dataloaders.custom_transforms as tr
from torch.utils.data import Dataset

from mypath import Path


class DAVIS2016(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 db_root_dir=Path.db_root_dir('davis2016'),
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 retname=True):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.retname = retname

        # offline training
        fname = 'train' if self.train else 'val'
        with open(os.path.join(db_root_dir, 'ImageSets/2016', fname + '.txt')) as f:
            seqs = f.readlines()
            img_list = []
            labels = []
            for seq in seqs:
                images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))
                images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
                img_list.extend(images_path)
                lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
                lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                labels.extend(lab_path)

        assert (len(labels) == len(img_list))
        self.img_list = img_list
        self.labels = labels

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)
        sample = {'image': img, 'gt': gt}

        if self.retname:
            cat = self.img_list[idx].split('/')[-2]
            frame_id = self.img_list[idx].split('/')[-1][:-4]
            sample['meta'] = {'image': cat + '+' + frame_id,
                              'object': str(0),
                              'category': cat,
                              'im_size': (img.shape[0], img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = np.array(PIL.Image.open(os.path.join(self.db_root_dir, self.img_list[idx])).convert('RGB')).astype(
            np.float32)

        if self.labels[idx] is not None:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)
            gt = np.array(label, dtype=np.float32)
            gt = (gt > 0.5).astype(np.float32)
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        return img, gt


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import torchvision
    import torch

    transforms = torchvision.transforms.Compose([tr.RandomHorizontalFlip(), tr.ToTensor()])

    dataset = DAVIS2016(train=True, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for i, data in enumerate(dataloader):
        plt.figure(1)
        img = tens2image(data['image']) / 255
        J = img[:, :, 0]
        J[tens2image(data['gt']) > 0.5] = 1
        plt.imshow(img)
        plt.pause(1)
        if i == 10:
            break
