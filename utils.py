import collections
import os

import torch
from torch.utils.data import DataLoader, ConcatDataset

from data_loader import SimulationLoader, EndoscopyLoader, SegLoader
from models_resnet import Resnet18_md, Resnet50_md, ResnetModel, UNet, NestedUNet, Model, ResidualBlock, UpProj_Block
from transforms import image_transforms


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(
            f"Input must contain tensor, dict or list, found {type(input)}")


def get_model(model, input_channels=3, pretrained=True):
    if model == 'resnet50_md':
        out_model = Resnet50_md(input_channels)
    elif model == 'resnet18_md':
        out_model = Resnet18_md(input_channels)
    elif model == 'AlbUNet':
        out_model = UNet()
    elif model == 'nested':
        out_model = NestedUNet()
    elif model == 'ldid':
        out_model = Model(ResidualBlock, UpProj_Block, 8)

    else:
        out_model = ResnetModel(
            input_channels, encoder=model, pretrained=pretrained)
    return out_model


def prepare_multi_dataloader(data_directory, mode, augment_parameters,
                             do_augmentation, batch_size, size, num_workers, train):
    seg_dir = data_directory + 'segmentation/'
    simu_dir = data_directory + 'simulation'
    arth_dir = data_directory + 'endoscopy/'

    data_transform = image_transforms(
        mode=mode,
        augment_parameters=augment_parameters,
        do_augmentation=do_augmentation,
        size=size, )
    seg_datasets = [SegLoader(os.path.join(seg_dir, i), mode, transform=data_transform) for i in
                    os.listdir(seg_dir)]
    seg_dataset = ConcatDataset(seg_datasets)

    simulation_dataset = EndoscopyLoader(arth_dir, mode, transform=data_transform)

    # sampler = BatchSampler(random_sampler, batch_size, drop_last=True)
    # arthroscopy_dataset = EndoscopyLoader(arth_dir, mode, transform=data_transform)
    # if mode == 'val' or mode == 'train':
    #     seg_datasets = [SegLoader(os.path.join(seg_dir, i), mode, transform=data_transform) for i in
    #                     os.listdir(seg_dir)]
    #     seg_dataset = ConcatDataset(seg_datasets)
    #     simulation_dataset = SimulationLoader(simu_dir, mode, transform=data_transform)
    # elif mode == 'test':
    #     seg_dataset = SegLoader(seg_dir, mode, transform=data_transform)
    #     simulation_dataset = SimulationLoader(data_directory, mode, transform=data_transform)
    seg_n_img = len(seg_dataset)
    # simu_n_img = len(simulation_dataset)
    arth_n_img = len(simulation_dataset)
    print('{} mode, {} images'.format(mode, seg_n_img + arth_n_img))
    if mode == 'val' or mode == 'train':
        seg_loader = DataLoader(seg_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                pin_memory=True, drop_last=True)
        arth_loader = DataLoader(simulation_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers,
                                 pin_memory=True, drop_last=True)

    else:
        seg_loader = DataLoader(seg_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                pin_memory=True)
        arth_loader = DataLoader(simulation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)

    return [arth_n_img, seg_n_img], [arth_loader, seg_loader]
    # return seg_n_img + arth_n_img, DataLoader(whole_dataset, batch_size=batch_size, shuffle=False,
    #                                           sampler=random_sampler,
    #                                           num_workers=num_workers, pin_memory=True, drop_last=True)


def prepare_dataloader(data_directory, mode, augment_parameters,
                       do_augmentation, batch_size, size, num_workers, train):
    # data_dirs = os.listdir(data_directory)
    data_transform = image_transforms(
        mode=mode,
        augment_parameters=augment_parameters,
        do_augmentation=do_augmentation,
        size=size, )
    if mode == 'val' or mode == 'train':
        datasets = [SegLoader(os.path.join(data_directory, i), mode, transform=data_transform) for i in
                    os.listdir(data_directory)]
        dataset = ConcatDataset(datasets)
    elif mode == 'test':
        dataset = SimulationLoader('data/test/', mode, transform=data_transform)
    # if mode == 'val' or mode == 'train':
    #     dataset = EndoscopyLoader(data_directory, mode, transform=data_transform)
    #     # dataset = SimulationLoader(data_directory, mode, transform=data_transform)
    #     # dataset = ArthroscopyLoader(data_directory, mode, transform=data_transform)
    # elif mode == 'test':
    #     dataset = EndoscopyLoader(data_directory, mode, transform=data_transform)
    #     # dataset = SimulationLoader(data_directory, mode, transform=data_transform)
    # # dataset = ArthroscopyLoader(data_directory, mode, transform=data_transform)

    n_img = len(dataset)
    print('Use a dataset with', n_img, 'images')
    if mode == 'val':
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True, drop_last=True)
    elif mode == 'train':
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                            drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    return n_img, loader
