import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import torch
from easydict import EasyDict as edict

from main_monodepth_pytorch import Model

torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.empty_cache()
import multiprocessing

multiprocessing.set_start_method('spawn', True)

# dict_parameters = edict({'data_dir': 'data/unified/train/',
#                          'val_data_dir': 'data/unified/test/',
#                          'model_path': 'data/models/monodepth_resnet50_001.pth',
#                          'output_directory': 'data/output/',
#                          'input_height': 256,
#                          'input_width': 256,
#                          'model': 'resnet50_md',
#                          'task': 'depth',
#                          'pretrained': True,
#                          'mode': 'val',
#                          'train_mode': 'train',
#                          'epochs': 140,
#                          'learning_rate': 1e-4,
#                          'batch_size': 20,
#                          'adjust_lr': True,
#                          'device': 'cuda:0',
#                          'do_augmentation': True,
#                          'augment_parameters': [0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
#                          'print_images': False,
#                          'print_weights': False,
#                          'input_channels': 3,
#                          'num_workers': 0,
#                          'use_multiple_gpu': False})
# model = SemanticAware(dict_parameters)
# # model.load('data/models/monodepth_resnet50_001_last.pth')
#
# model.train()

dict_parameters_arch = edict({'data_dir': 'data/unified/train/',
                              'val_data_dir': 'data/unified/test/',
                              'model_path': 'data/models/monodepth_resnet50_001.pth',
                              'output_directory': 'data/output/',
                              'input_height': 256,
                              'input_width': 256,
                              'model': 'resnet50_md',
                              'task': 'depth',
                              'pretrained': True,
                              'mode': 'val',
                              'train_mode': 'train',
                              'epochs': 50,
                              'learning_rate': 1e-4,
                              'batch_size': 20,
                              'adjust_lr': True,
                              'device': 'cuda:0',
                              'do_augmentation': True,
                              'augment_parameters': [0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                              'print_images': False,
                              'print_weights': False,
                              'input_channels': 3,
                              'num_workers': 0,
                              'use_multiple_gpu': False})
# model = SemanticAware(dict_parameters_arch)
# model.load('data/models/monodepth_resnet50_001.pth')
#
# model.train()
# model = Model(dict_parameters)
# # model.load('data/models/simulation_unet++_skipdisp/monodepth_resnet50_001_last.pth')
#
# model.train()

dict_parameters_test = edict({'data_dir': 'data/unified/test/',
                              'model_path': 'data/models/monodepth_resnet50_001.pth',
                              'output_directory': 'data/output/test',
                              'input_height': 256,
                              'input_width': 256,
                              'task': 'depth',
                              'model': 'resnet50_md',
                              'pretrained': False,
                              'mode': 'test',
                              'device': 'cuda:0',
                              'input_channels': 3,
                              'num_workers': 0,
                              'use_multiple_gpu': False})
model_test = Model(dict_parameters_test)

model_test.test()
# model_test.seg_inference()

disp = np.load('data/output/test/disparities.npy')  # Or disparities.npy for output without post-processing
# disp.shape

# disp_to_img = cv2.resize(disp[0].squeeze(), (384, 384))
# depth = 0.00152 * 304.02197222 / (384 * disp_to_img)
# plt.imshow(depth * 1000, vmax=15, cmap='gray')
# plt.show()
disp_to_img = skimage.transform.resize(disp[0].squeeze(), [384, 384], mode='constant')

# plt.imshow(depth * 1000,cmap='plasma')
# plt.show()
plt.imsave(os.path.join(dict_parameters_test.output_directory,
                        dict_parameters_test.model_path.split('/')[-1][:-4] + '_test_output.png'), disp_to_img,
           cmap='plasma')

for i in range(disp.shape[0]):
    # cv2.imwrite(os.path.join(dict_parameters_test.output_directory, str(i) + '.png'), disp[i])
    # plt.imsave(os.path.join(dict_parameters_test.output_directory,
    #                         str(i) + '.png'), disp[i], cmap='gray')

    # disp_to_img = cv2.resize(disp[i].squeeze(), (384, 384))
    # depth = 0.00152 * 304.02197222 / (384 * disp_to_img)
    # plt.imsave(os.path.join(dict_parameters_test.output_directory,
    #                         str(i) + '.png'), depth * 1000, vmax=20)

    disp_to_img = cv2.resize(disp[i].squeeze(), (384, 384))
    plt.imsave(os.path.join(dict_parameters_test.output_directory,
                            str(i) + '.png'), disp_to_img, cmap='plasma')

plt.imsave(os.path.join(dict_parameters_test.output_directory,
                        dict_parameters_test.model_path.split('/')[-1][:-4] + '_gray.png'), disp_to_img, cmap='gray')
