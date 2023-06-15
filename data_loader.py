from __future__ import print_function, division
import torch
from skimage import io, transform
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import transform

class RescaleT(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant',
                               order = 0, preserve_range = True)
        return {'imidx': imidx, 'image': img, 'label': lbl}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = random.randrange(0, h - new_h , 4)
        left = random.randrange(0, w - new_w , 4)
        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]
        return {'imidx': imidx, 'image': image, 'label': label}


class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None, test=False):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform
        self.test = test

    # def __len__(self):
    #     return len(self.image_name_list)
    #
    # def __getitem__(self, idx):
    #     img_name = self.image_name_list[idx]
    #     lbl_name = self.label_name_list[idx]
    #     image = io.imread(img_name)
    #     label = io.imread(lbl_name)
    #     sample = {'imidx': idx, 'image': image, 'label': label}
    #     if self.transform:
    #         sample = self.transform(sample)
    #     return sample


    def __len__(self):
         return len(self.image_name_list)


    def __getitem__(self, idx):
         image = io.imread(self.image_name_list[idx])
         imidx = np.array([idx])
         if len(self.label_name_list) == 0:
            label = np.zeros_like(image)
         else:
            label = io.imread(self.label_name_list[idx])
         sample = {'imidx': imidx, 'image': image, 'label': label}
         if self.transform:
            sample = self.transform(sample)
         return sample


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        if np.max(label) <1e-6:
            label = label
        else:
            label = label / np.max(label)
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        image = image / np.max(image)
        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        imidx = imidx.copy()
        tmpImg = transforms.ToTensor()(tmpImg.copy())
        tmpLbl = transforms.ToTensor()(label.copy())
        return {'imidx': torch.from_numpy(imidx), 'image': tmpImg, 'label': tmpLbl}