import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# from albumentations import RandomCrop, RandomRotate90, Transpose, ElasticTransform, Compose


def image_transforms(mode='val', augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                     do_augmentation=True, transformations=None, size=(256, 256)):
    if mode == 'val' or mode == 'train':
        data_transform = transforms.Compose([
            RandomCrop_wtseg(train=True, size=size),
            # ResizeImage(train=True, size=size),
            RandomFlip_wtseg(do_augmentation),
            AugmentImagePair(augment_parameters, do_augmentation),
            # ToTensor_wtseg(train=True),
            # AugmentImagePair(augment_parameters, do_augmentation)
            # RandomCrop(train=True, size=size),
            # RandomFlip(do_augmentation),
            # ToTensor(train=True),
            # AugmentImagePair(augment_parameters, do_augmentation)
        ])
        return data_transform
    elif mode == 'simu':
        data_transform = transforms.Compose([
            RandomCrop_wtseg(train=True, size=size),
            # ResizeImage(train=True, size=size),
            RandomFlip_wtseg(do_augmentation),
            ToTensor_wtseg(train=True),
            AugmentImagePair(augment_parameters, do_augmentation)
            # RandomCrop(train=True, size=size),
            # RandomFlip(do_augmentation),
            # ToTensor(train=True),
            # AugmentImagePair(augment_parameters, do_augmentation)
        ])
        return data_transform
    elif mode == 'test':
        data_transform = transforms.Compose([
            ResizeImage(train=False, size=size),
            ToTensor_wtseg(train=False),
            DoTest(),
        ])
        return data_transform
    elif mode == 'custom':
        data_transform = transforms.Compose(transformations)
        return data_transform
    else:
        print('Wrong mode')


class RandomCrop_wtseg(object):
    def __init__(self, train=True, size=(256, 256)):
        self.train = train
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        if self.train:
            left_image = sample['left_image']
            right_image = sample['right_image']
            i, j, h, w = transforms.RandomCrop.get_params(left_image, output_size=(256, 256))
            new_left_image = TF.crop(left_image, i, j, h, w)
            new_right_image = TF.crop(right_image, i, j, h, w)
            sample = {'left_image': new_left_image, 'right_image': new_right_image}
        else:
            left_image = sample
            new_left_image = self.transform(left_image)
            sample = new_left_image
        return sample


class RandomFlip_wtseg(object):
    def __init__(self, do_augmentation):
        self.transform = transforms.RandomHorizontalFlip(p=1)
        self.do_augmentation = do_augmentation

    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']
        k = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if k > 0.5:
                fliped_left = self.transform(right_image)
                fliped_right = self.transform(left_image)
                sample = {'left_image': fliped_left, 'right_image': fliped_right}
        else:
            sample = {'left_image': left_image, 'right_image': right_image}
        return sample


class ToTensor_wtseg(object):
    def __init__(self, train):
        self.train = train
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __call__(self, sample):
        if self.train:
            left_image = sample['left_image']
            right_image = sample['right_image']
            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
            sample = {'left_image': new_left_image,
                      'right_image': new_right_image}
        else:
            left_image = sample
            sample = self.transform(left_image)
        return sample


class RandomCrop(object):
    def __init__(self, train=True, size=(256, 256)):
        self.train = train
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        if self.train:
            left_image = sample['left_image']
            left_label = sample['left_label']
            right_image = sample['right_image']
            right_label = sample['right_label']
            # new_left_image = transforms.CenterCrop(256)(left_image)
            # new_left_label = transforms.CenterCrop(256)(left_label)
            # new_right_image = transforms.CenterCrop(256)(right_image)
            # new_right_label = transforms.CenterCrop(256)(right_label)
            i, j, h, w = transforms.RandomCrop.get_params(left_image, output_size=(256, 256))
            new_left_image = TF.crop(left_image, i, j, h, w)
            new_left_label = TF.crop(left_label, i, h, h, w)
            new_right_image = TF.crop(right_image, i, j, h, w)
            new_right_label = TF.crop(right_label, i, j, h, w)
            sample = {
                'left_image': new_left_image,
                'right_image': new_right_image,
                'left_label': new_left_label,
                'right_label': new_right_label
            }
        else:
            left_image = sample['left_image']
            left_label = sample['left_label']
            new_left_image = self.transform(left_image)
            new_left_label = self.transform(left_label)
            sample = {'left_image': new_left_image, 'left_label': new_left_label}
        return sample


class ResizeImage(object):
    def __init__(self, train=True, size=(256, 512)):
        self.train = train
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
        ])

    def __call__(self, sample):
        if self.train:
            left_image = sample['left_image']
            right_image = sample['right_image']
            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
            sample = {'left_image': new_left_image, 'right_image': new_right_image}
        else:
            # left_image = sample['left_image']
            new_left_image = self.transform(sample)
            sample = new_left_image
        return sample


class DoTest(object):
    def __call__(self, sample):
        # new_sample = torch.reshape(sample, (1, -1, 384, 384))
        new_sample = torch.stack((sample, torch.flip(sample, [2])))
        return new_sample


class ToTensor(object):
    def __init__(self, train):
        self.train = train
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __call__(self, sample):
        if self.train:
            left_image = sample['left_image']
            left_label = sample['left_label']
            right_image = sample['right_image']
            right_label = sample['right_label']
            new_right_image = self.transform(right_image)
            # new_right_label = self.transform(right_label)
            new_left_image = self.transform(left_image)
            # new_left_label = self.transform(left_label)
            sample = {
                'left_image': new_left_image,
                'right_image': new_right_image,
                'left_label': left_label,
                'right_label': right_label
            }
        else:
            left_image = sample['left_image']
            sample = {'left_image': self.transform(left_image), 'left_label': sample['left_label']}
        return sample


class RandomFlip(object):
    def __init__(self, do_augmentation):
        self.transform = transforms.RandomHorizontalFlip(p=1)
        self.do_augmentation = do_augmentation

    def __call__(self, sample):
        left_image = sample['left_image']
        left_label = sample['left_label']
        right_image = sample['right_image']
        right_label = sample['right_label']
        k = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if k > 0.5:
                fliped_left = self.transform(right_image)
                fliped_left_label = self.transform(right_label)
                fliped_right = self.transform(left_image)
                fliped_right_label = self.transform(left_label)
                sample = {
                    'left_image': fliped_left,
                    'right_image': fliped_right,
                    'left_label': fliped_left_label,
                    'right_label': fliped_right_label
                }
        else:
            sample = {
                'left_image': left_image,
                'right_image': right_image,
                'left_label': left_label,
                'right_label': right_label
            }
        return sample


class AugmentImagePair(object):
    def __init__(self, augment_parameters, do_augmentation):
        self.do_augmentation = do_augmentation
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2
        self.tmp = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        left_image = self.tmp(sample['left_image'])
        right_image = self.tmp(sample['right_image'])
        p = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if p > 0.5:
                # randomly shift gamma
                random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
                left_image_aug = left_image ** random_gamma
                right_image_aug = right_image ** random_gamma

                # randomly shift brightness
                random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
                left_image_aug = left_image_aug * random_brightness
                right_image_aug = right_image_aug * random_brightness

                # randomly shift color
                random_colors = np.random.uniform(self.color_low, self.color_high, 3)
                for i in range(3):
                    left_image_aug[i, :, :] *= random_colors[i]
                    right_image_aug[i, :, :] *= random_colors[i]

                # saturate
                # left_image_aug = np.clip(left_image_aug, 0, 1)
                left_image_aug = torch.clamp(left_image_aug, 0, 1)
                # right_image_aug = np.clip(right_image_aug, 0, 1)
                right_image_aug = torch.clamp(right_image_aug, 0, 1)

                sample['left_image'] = left_image_aug
                sample['right_image'] = right_image_aug
            else:
                sample['left_image'] = left_image
                sample['right_image'] = right_image
        return sample
