import os

import albumentations as albu
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from albumentations import Compose, ElasticTransform, RandomRotate90, RandomCrop, Transpose, OneOf, GridDistortion, \
    OpticalDistortion, RandomBrightnessContrast, RandomGamma, Resize
from torch.utils.data import Dataset

depth_aug = albu.Compose([
    # Color augmentation
    albu.OneOf([
        albu.Compose([
            albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            albu.RandomGamma(gamma_limit=(80, 120), p=0.5),
            albu.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=0, val_shift_limit=0, p=0.5)]),
        albu.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=0.5)
    ]),
    # Image quality augmentation
    albu.OneOf([
        albu.Blur(p=0.5),
        albu.MedianBlur(p=0.5),
        albu.MotionBlur(p=0.5),
        albu.JpegCompression(quality_lower=20, quality_upper=100, p=0.5)
    ]),
    # Noise augmentation
    albu.OneOf([
        albu.GaussNoise(var_limit=(10, 30), p=0.5),
        albu.IAAAdditiveGaussianNoise(loc=0, scale=(0.005 * 255, 0.02 * 255), p=0.5)
    ]),
    albu.Resize(256, 256, p=1),
])

seg_aug = Compose([
    Transpose(p=0.5),
    RandomRotate90(p=1),
    # RandomBrightnessContrast(p=1),
    # RandomGamma(p=1),
    OneOf([
        # GaussNoise(p=1),
        RandomBrightnessContrast(p=1),
        RandomGamma(p=1),
    ]),
    # OneOf([
    #     ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    #     ShiftScaleRotate(p=1),
    # ]),
    # ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    # # ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    OneOf([
        ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(p=1),
        # ShiftScaleRotate(p=1),
        OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
    ]),
    RandomCrop(256, 256, p=1)
])

seg_aug_2 = Compose([
    Resize(256, 256, p=1),
])


# class JointLoader(Dataset):
#     def __init__(self, seg_dir, simu_dir, mode, data_transform):
#         seg_datasets = [SegLoader(os.path.join(seg_dir, i), mode, transform=data_transform) for i in
#                         os.listdir(seg_dir)]
#         self.seg_dataset = ConcatDataset(seg_datasets)
#         self.simulation_dataset = SimulationLoader(simu_dir, mode, transform=data_transform)
#
#     def __len__(self):
#         return len(self.seg_dataset) + len(self.simulation_dataset)
#
#     def __getitem__(self, item):


class SegLoader(Dataset):
    def __init__(self, root_dir, mode, transform):
        img_dir = list()
        label_dir = list()

        for r, d, f in os.walk(root_dir + '/image/'):
            for file in f:
                if '.png' in file:
                    img_dir.append(os.path.join(r, file))
                    label_dir.append(os.path.join(os.path.join(root_dir + '/index/'), file))

        self.transform = transform
        self.mode = mode
        self.img_dir = sorted(img_dir)
        self.label_dir = sorted(label_dir)

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, item):
        image = cv2.imread(self.img_dir[item], 0)
        image = (255 / (np.max(image) - np.min(image)) * (image - np.min(image)) + 0.5).astype("uint8")
        image = np.repeat(image[..., np.newaxis], 3, -1)
        image_label = cv2.imread(self.label_dir[item], 0)
        # image = Image.open(self.img_dir[item])
        # image_label = Image.open(self.label_dir[item]).convert('L')
        sample = {'image': image, 'label': image_label}
        tmp = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if self.mode == 'train':
            augmented = seg_aug(image=image, mask=image_label)
            image = augmented['image']
            mask = augmented['mask']

            sample = {'left_image': tmp(image), 'left_label': mask,
                      'left_index': np.squeeze(np.eye(5)[np.asarray(mask).reshape(-1)]).reshape(256, 256,
                                                                                                5)}
        elif self.mode == 'val':
            # augmented = seg_aug_2(image=image, mask=image_label)
            # image = augmented['image']
            # mask = augmented['mask']
            image = cv2.imread(self.img_dir[item])
            sample = {'left_image': tmp(image), 'left_label': image_label,
                      'left_index': np.squeeze(np.eye(5)[np.asarray(image_label).reshape(-1)]).reshape(384, 384,
                                                                                                       5)}
        else:
            sample = tmp(image)
            # sample = {'image': transforms.ToTensor()(image), 'label': image_label,
            #           'index': np.squeeze(np.eye(5)[np.asarray(image_label).reshape(-1)]).reshape(384, 384,
            #                                                                                       5)}
        # sample = {'img': image, 'label': image_label,
        #           'index': np.squeeze(np.eye(5)[np.asarray(sample['left_label']).reshape(-1)]).reshape(256, 256,
        #                                                                                                5)}
        return sample


class EndoscopyLoader(Dataset):
    def __init__(self, root_dir, mode, transform):
        left_dir = list()
        right_dir = list()

        if mode == 'val' or mode == 'train':
            for r, d, f in os.walk(root_dir):
                for file in f:
                    if '_L' in file:
                        left_dir.append(os.path.join(r, file))
                    elif '_R' in file:
                        right_dir.append(os.path.join(r, file))
        elif mode == 'test':
            for r, d, f in os.walk(root_dir):
                for file in f:
                    if '_L.jpg' in file:
                        left_dir.append(os.path.join(r, file))
        self.transform = transform
        self.mode = mode

        self.left_paths = sorted(left_dir)
        self.right_paths = sorted(right_dir)

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, item):
        left_image = cv2.imread(self.left_paths[item])
        # left_image = Image.open(self.left_paths[item])
        tmp = transforms.Compose(
            [transforms.ToTensor()])
        if self.mode == 'train':
            right_image = cv2.imread(self.right_paths[item])
            # right_image = Image.open(self.right_paths[item])
            aug_left_image = depth_aug(image=left_image)['image']
            aug_right_image = depth_aug(image=right_image)['image']
            # augmented = depth_aug(image=left_image, mask=right_image)
            # aug_left_image = augmented['image']
            # aug_right_image = augmented['mask']

            # right_image = transforms.RandomHorizontalFlip(p=1)(right_image)
            sample = {'left_image': tmp(aug_left_image), 'right_image': tmp(aug_right_image)}

            # if self.transform:
            #     sample = self.transform(sample)
            #     return sample
            # else:
            return sample
        elif self.mode == 'val':
            right_image = Image.open(self.right_paths[item])
            sample = {'left_image': tmp(left_image), 'right_image': tmp(right_image)}
            return sample
        elif self.mode == 'test':
            if self.transform:
                left_image = self.transform(left_image)
            return left_image


class ArthroscopyLoader(Dataset):
    def __init__(self, root_dir, mode, transform):
        left_dir = list()
        right_dir = list()

        left_label_dir = list()
        right_label_dir = list()
        if mode == 'val' or mode == 'train':
            for r, d, f in os.walk(root_dir):
                for file in f:
                    if '_L' in file:
                        left_dir.append(os.path.join(r, file))
                        # left_index_dir.append(os.path.join(root_dir + 'index/', file))
                        left_label_dir.append(os.path.join(r + '/index/', file))
                    elif '_R' in file:
                        right_dir.append(os.path.join(r, file))
                        # right_index_dir.append(os.path.join(root_dir + 'index/', file))
                        right_label_dir.append(os.path.join(r + '/index/', file))
        elif mode == 'test':
            for r, d, f in os.walk(root_dir + 'image/'):
                for file in f:
                    if '_L' in file:
                        left_dir.append(os.path.join(r, file))
                        # left_index_dir.append(os.path.join(root_dir + 'index/', file))
                        left_label_dir.append(os.path.join(root_dir + '/index/', file))
        self.transform = transform
        self.mode = mode
        self.left_paths = sorted(left_dir)
        self.right_paths = sorted(right_dir)

        self.left_label = sorted(left_label_dir)
        self.right_label = sorted(right_label_dir)

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, item):
        left_image = Image.open(self.left_paths[item])
        left_label = Image.open(self.left_label[item]).convert('L')

        if self.mode == 'val':
            right_image = Image.open(self.right_paths[item])
            right_label = Image.open(self.right_label[item]).convert('L')
            sample = {
                'left_image': left_image,
                'right_image': right_image,
                'left_label': left_label,
                'right_label': right_label
            }
            if self.transform:
                sample = self.transform(sample)
                sample = {
                    'left_image': sample['left_image'],
                    'right_image': sample['right_image'],
                    'left_label': np.asarray(sample['left_label']),
                    'right_label': np.asarray(sample['right_label']),
                    'left_index': np.squeeze(np.eye(5)[np.asarray(sample['left_label']).reshape(-1)]).reshape(256, 256,
                                                                                                              5),
                    'right_index': np.squeeze(np.eye(5)[np.asarray(sample['right_label']).reshape(-1)]).reshape(
                        256, 256, 5)
                }
                return sample
            else:
                return sample
        elif self.mode == 'test':
            sample = left_image
            if self.transform:
                sample = self.transform(sample)
            # sample = {
            #     'left_image': left_image,
            #     'left_label': left_label,
            # }
            # if self.transform:
            #     sample = self.transform(sample)
            #     sample = {
            #         'left_image': sample['left_image'],
            #         'left_label': sample['left_label'],
            #         'left_index': np.squeeze(np.eye(5)[sample['left_label'].reshape(-1)]).reshape(256, 256, 5)
            #     }
            return sample


class SimulationLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        left_dir = list()
        right_dir = list()
        if mode == 'val' or mode == 'train':
            for r, d, f in os.walk(root_dir):
                for file in f:
                    if 'right' in r:
                        right_dir.append(os.path.join(r, file))
                    elif 'left' in r:
                        left_dir.append(os.path.join(r, file))
        elif mode == 'test':
            for r, d, f in os.walk(root_dir):
                for file in f:
                    if '.png' in file:
                        left_dir.append(os.path.join(r, file))

        self.transform = transform
        self.mode = mode

        self.left_paths = sorted(left_dir)
        self.right_paths = sorted(right_dir)

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, item):
        # read RGB left and right
        left_image = Image.open(self.left_paths[item])
        tmp = transforms.Compose(
            [transforms.ToTensor()])
        if self.mode == 'train':
            right_image = Image.open(self.right_paths[item])
            sample = {'left_image': left_image, 'right_image': right_image}

            if self.transform:
                # augmented = depth_aug(image=sample['left_image'], mask=sample['right_image'])
                sample = self.transform(sample)
                return sample
            else:
                return sample
        elif self.mode == 'val':
            right_image = Image.open(self.right_paths[item])
            sample = {'left_image': tmp(left_image), 'right_image': tmp(right_image)}
            # sample = self.transform(sample)

            return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image
