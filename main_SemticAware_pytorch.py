import argparse
import random
import time

import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from loss import MonodepthLoss
from utils import get_model, to_device, prepare_dataloader, prepare_multi_dataloader

# custom modules
# plot params

mpl.rcParams['figure.figsize'] = (15, 10)
writer = SummaryWriter('log/')
task = {0: 'depth', 1: 'seg', 2: 'both'}


def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')

    parser.add_argument('data_dir',
                        help='path to the dataset folder. \
                        It should contain subfolders with following structure:\
                        "image_02/data" for left images and \
                        "image_03/data" for right images'
                        )
    parser.add_argument('val_data_dir',
                        help='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02/data" for left images and \
                            "image_03/data" for right images'
                        )
    parser.add_argument('model_path', help='path to the trained model')
    parser.add_argument('output_directory',
                        help='where save dispairities\
                        for tested images'
                        )
    parser.add_argument('--input_height', type=int, help='input height',
                        default=256)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=512)
    parser.add_argument('--model', default='resnet18_md',
                        help='encoder architecture: ' +
                             'resnet18_md or resnet50_md ' +
                             '(default: resnet18)'
                             + 'or torchvision version of any resnet model'
                        )
    parser.add_argument('--pretrained', default=False,
                        help='Use weights of pretrained model'
                        )
    parser.add_argument('--mode', default='val',
                        help='mode: val or 120_epochs (default: val)')
    parser.add_argument('--epochs', default=50,
                        help='number of total epochs to run')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', default=256,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)'
                        )
    parser.add_argument('--device',
                        default='cuda:0',
                        help='choose cpu or cuda:0 device"'
                        )
    parser.add_argument('--do_augmentation', default=True,
                        help='do augmentation of images or not')
    parser.add_argument('--augment_parameters', default=[
        0.8,
        1.2,
        0.5,
        2.0,
        0.8,
        1.2,
    ],
                        help='lowest and highest values for gamma,\
                        brightness and color respectively'
                        )
    parser.add_argument('--print_images', default=False,
                        help='print disparity and image\
                        generated from disparity on every iteration'
                        )
    parser.add_argument('--print_weights', default=False,
                        help='print weights of every layer')
    parser.add_argument('--input_channels', default=3,
                        help='Number of channels in input tensor')
    parser.add_argument('--num_workers', default=4,
                        help='Number of workers in dataloader')
    parser.add_argument('--use_multiple_gpu', default=False)
    args = parser.parse_args()
    return args


def init_learning_rate(optimizer, epoch, learning_rate):
    decay = (learning_rate - 0) * (1 - float(epoch) / 30) ** 2 + 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches"""

    # if epoch >= 30 and epoch < 100:
    #     lr = learning_rate / 2
    # elif epoch >= 100:
    #     lr = learning_rate / 4
    # else:
    #     lr = learning_rate
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    decay = (learning_rate - 0) * (1 - float(epoch) / 140) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay


def post_process_disparity(disp_left, disp_right):
    (_, h, w) = disp_left.shape
    l_disp = disp_left[0, :, :]
    r_disp = disp_right[0, :, :]
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


class SemanticAware:

    def __init__(self, args):
        self.args = args

        # Set up model
        self.device = args.device
        self.model = get_model(
            args.model, input_channels=args.input_channels, pretrained=args.pretrained)
        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        if args.mode == 'val':
            self.loss_function = MonodepthLoss(
                n=4,
                SSIM_w=0.85,
                disp_gradient_w=0.1, lr_w=1).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.learning_rate)
            self.val_n_img, self.val_loader = prepare_multi_dataloader(args.val_data_dir, args.mode,
                                                                       args.augment_parameters,
                                                                       False, args.batch_size,
                                                                       (args.input_height,
                                                                        args.input_width),
                                                                       args.num_workers, False)

            self.n_img, self.loader = prepare_multi_dataloader(args.data_dir, 'train', args.augment_parameters,
                                                               args.do_augmentation, args.batch_size,
                                                               (args.input_height,
                                                                args.input_width),
                                                               args.num_workers, True)
        else:
            self.model.load_state_dict(torch.load(args.model_path))
            args.augment_parameters = None
            args.do_augmentation = False
            args.batch_size = 1

            self.n_img, self.loader = prepare_dataloader(args.data_dir, args.mode,
                                                         args.augment_parameters,
                                                         False, args.batch_size,
                                                         (args.input_height,
                                                          args.input_width),
                                                         args.num_workers, False)
        self.output_directory = args.output_directory
        self.input_height = args.input_height
        self.input_width = args.input_width

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def train(self):
        index = {0: 'both', 1: 'seg', 2: 'both'}
        losses = []
        val_losses = []
        best_loss = float('Inf')
        best_seg_val_loss = float('Inf')

        best_simu_val_loss = float('Inf')
        # best_val_loss = float('Inf')

        running_val_loss = 0.0
        running_ce_loss = 0.0
        running_dice_loss = 0.0

        # self.model.eval()
        # with torch.no_grad():
        #     for idx, loader in enumerate(self.val_loader):
        #         for data in loader:
        #             data = to_device(data, self.device)
        #             left = data['left_image']
        #             if idx == 0:
        #                 # simulation
        #                 right = data['right_image']
        #                 flip_right = torch.flip(right, [3])
        #                 collection = self.model([left, flip_right], task[idx])
        #                 loss = self.loss_function(collection[0], data, task[idx], input_right=collection[1])
        #                 val_losses.append(loss.item())
        #                 running_val_loss += loss.item()
        #             # elif idx == 1:
        #             #
        #             #     # semantic
        #             #     collection = self.model([left], task[idx])
        #             #     tmp_loss = 0
        #             #     tmp_ce = 0
        #             #     tmp_dice = 0
        #             #     for i in collection[0]:
        #             #         loss, ce, dice = self.loss_function(i, data, task[idx])
        #             #         tmp_loss += loss.item()
        #             #         tmp_ce += ce.item()
        #             #         tmp_dice += dice.item()
        #             #     val_losses.append(tmp_loss / 1)
        #             #     running_ce_loss += tmp_ce / 1
        #             #     running_dice_loss += tmp_dice / 1
        #             #     running_val_loss += tmp_loss / 1
        #             elif idx == 2:
        #                 # joint
        #                 right = data['right_image']
        #                 flip_right = torch.flip(right, [3])
        #                 collection = self.model([left, flip_right], task[idx])
        #                 loss = self.loss_function(collection[0], data, task[idx], input_right=collection[1])
        #                 val_losses.append(loss.item())
        #                 running_val_loss += loss.item()
        #         running_val_loss /= self.val_n_img[idx] / self.args.batch_size
        #         if idx == 1:
        #             running_ce_loss /= self.val_n_img[idx] / self.args.batch_size
        #             running_dice_loss /= self.val_n_img[idx] / self.args.batch_size
        #             print('segmentation, val loss: {}, cross entropy: {}, dice: {}'.format(running_val_loss,
        #                                                                                    running_ce_loss,
        #                                                                                    running_dice_loss))
        #         else:
        #             print('{},val loss: {}'.format(index[idx], running_val_loss))
        # print(
        #     '--------------------------------------------------------------------------------------------------------')
        # print('training simulation for model initialise')
        choice = 0
        # for epoch in range(40):
        #     if self.args.adjust_lr:
        #         init_learning_rate(self.optimizer, epoch,
        #                            self.args.learning_rate)
        #     c_time = time.time()
        #     running_loss = 0.0
        #     self.model.train()
        #     print('simulation initialise')
        #     for data in self.loader[0]:
        #         data = to_device(data, self.device)
        #         left = data['left_image']
        #
        #         self.optimizer.zero_grad()
        #         right = data['right_image']
        #         flip_right = torch.flip(right, [3])
        #         collection = self.model([left, flip_right], task[choice])
        #         loss = self.loss_function(collection[0], data, task[choice], input_right=collection[1])
        #         loss.backward()
        #         self.optimizer.step()
        #         losses.append(loss.item())
        #         running_loss += loss.item()
        #     running_loss /= self.n_img[choice] / self.args.batch_size
        #     if epoch % 5 == 0:
        #         print(
        #             'Epoch:',
        #             epoch + 1,
        #             'train_loss:',
        #             running_loss,
        #             'time:',
        #             round(time.time() - c_time, 3),
        #             's',
        #         )
        # self.save(self.args.model_path[:-4] + '_simu.pth')
        for epoch in range(self.args.epochs):
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch,
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            running_ce = 0.0
            running_dice = 0.0
            running_val_loss = 0.0
            running_seg_loss = 0.0
            running_simu_loss = 0.0
            self.model.train()
            # if epoch % 5 != 0:
            #     choice = 1
            #     adjust_learning_rate(self.optimizer, epoch, self.args.learning_rate)
            # else:
            #     choice = 0
            # choice = random.randint(0, 1)
            # if epoch <= 20:
            #     choice = epoch % 2
            # else:
            #     choice = epoch % 3
            choice = random.randint(0, 1)
            # choice = epoch % 2
            print('training depth') if choice == 0 else print('training segmentation')
            for idx, (simu, seg) in enumerate(zip(self.loader[0], self.loader[1])):
                # self.optimizer.zero_grad()
                simu = to_device(simu, self.device)
                # seg = to_device(seg, self.device)
                data_simu_left = simu['left_image']
                data_simu_right = simu['right_image']
                # data_seg_left = seg['left_image']
                # collection_seg = self.model([data_seg_left], 'seg')
                # tmp_loss = 0
                # tmp_ce = 0
                # tmp_dice = 0
                # for i in collection_seg[0]:
                #     loss_seg, ce, dice = self.loss_function(i, seg, 'seg')
                #     tmp_loss += loss_seg
                #     tmp_ce += ce
                #     tmp_dice += dice
                # (tmp_loss / 1).backward()
                # self.optimizer.step()
                # self.optimizer.zero_grad()

                data_simu_flip_right = torch.flip(data_simu_right, [3])
                collection_simu = self.model([data_simu_left, data_simu_flip_right], 'depth')
                loss_simu = self.loss_function(collection_simu[0], simu, 'depth', input_right=collection_simu[1])
                loss_simu.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # running_dice += (tmp_dice / 1).item()
                # running_ce += (tmp_ce / 1).item()
                running_simu_loss += loss_simu.item()
                # running_seg_loss += (tmp_loss / 1).item()
            # for data in self.loader[choice]:
            #     data = to_device(data, self.device)
            #     left = data['left_image']
            #     self.optimizer.zero_grad()
            #     if choice == 0:
            #         # simulation
            #         right = data['right_image']
            #         flip_right = torch.flip(right, [3])
            #         collection = self.model([left, flip_right], task[choice])
            #         loss = self.loss_function(collection[0], data, task[choice], input_right=collection[1])
            #         losses.append(loss.item())
            #         running_loss += loss.item()
            #     elif choice == 1:
            #         # semantic
            #         collection = self.model([left], task[choice])
            #         loss, ce, dice = self.loss_function(collection[0], data, task[choice])
            #         losses.append(loss.item())
            #         running_dice += dice.item()
            #         running_ce += ce.item()
            #         running_loss += loss.item()
            #     elif choice == 2:
            #         # joint
            #         right = data['right_image']
            #         flip_right = torch.flip(right, [3])
            #         collection = self.model([left, flip_right], task[choice])
            #         loss = self.loss_function(collection[0], data, task[choice], input_right=collection[1])
            #         losses.append(loss.item())
            #         running_loss += loss.item()
            #     loss.backward()
            #     self.optimizer.step()
            running_simu_loss /= (idx + 1)
            running_seg_loss /= (idx + 1)
            running_dice /= (idx + 1)
            running_ce /= (idx + 1)
            print('simulation: {}, segmentation: {}, ce: {}, dice: {}'.format(running_simu_loss, running_seg_loss,
                                                                              running_ce, running_dice))
            # running_loss /= self.n_img[choice] / self.args.batch_size
            # if choice == 1:
            #     running_ce /= self.n_img[choice] / self.args.batch_size
            #     running_dice /= self.n_img[choice] / self.args.batch_size
            #     print('segmentation, epoch: {}, train loss: {}, train ce: {}, train dice:{}'.format(
            #         epoch,
            #         running_loss,
            #         running_ce,
            #         running_dice_loss))
            # else:
            #     print('simulation, epoch: {},train loss:{}'.format(epoch, running_loss))
            running_val_loss = 0.0
            running_ce_loss = 0.0
            running_dice_loss = 0.0
            running_val_seg_loss = 0.0
            # self.model.eval()
            # with torch.no_grad():
            #     for data in self.val_loader[1]:
            #         data = to_device(data, self.device)
            #         left = data['left_image']
            #
            #         # semantic
            #         collection = self.model([left], 'seg')
            #         tmp_loss = 0
            #         tmp_ce = 0
            #         tmp_dice = 0
            #         for i in collection[1]:
            #             loss, ce, dice = self.loss_function(i, data, 'seg')
            #             tmp_loss += loss.item()
            #             tmp_ce += ce.item()
            #             tmp_dice += dice.item()
            #         val_losses.append(tmp_loss / 1)
            #         running_ce_loss += tmp_ce / 1
            #         running_dice_loss += tmp_dice / 1
            #         running_val_loss += tmp_loss / 1
            #
            #     running_val_loss /= self.val_n_img[1] / self.args.batch_size
            #
            #     running_val_seg_loss /= self.val_n_img[1] / self.args.batch_size
            #     running_ce_loss /= self.val_n_img[1] / self.args.batch_size
            #     running_dice_loss /= self.val_n_img[1] / self.args.batch_size
            #     print('Segmentation, val loss: {}, cross entropy: {}, dice: {}'.format(running_val_loss,
            #                                                                            running_ce_loss,
            #                                                                            running_dice_loss))
            #
            # print(
            #     '----------------------------------------------------------------------------------------------------')

            self.save(self.args.model_path[:-4] + '_last.pth')

            if running_dice_loss < best_seg_val_loss:
                self.save(self.args.model_path[:-4] + '_cpt_seg_dice.pth')
                best_seg_val_loss = running_dice_loss
                print(
                    '**********************************************************************************************')
                print('Seg model_saved')
                print(
                    '**********************************************************************************************')
            if running_val_loss < best_loss:
                self.save(self.args.model_path[:-4] + '_cpt_seg_all.pth')
                best_loss = running_val_loss
                print(
                    '**********************************************************************************************')
                print('Seg model_saved')
                print(
                    '**********************************************************************************************')
        print('Finished Training. Best loss:', best_loss)
        self.save(self.args.model_path)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def seg_inference(self):
        self.model.eval()
        segmentations = np.zeros((self.n_img,
                                  384, 384),
                                 dtype=np.float32)
        with torch.no_grad():
            for idx, data in enumerate(self.loader):
                data = to_device(data, self.device)
                left = data['image']
                # left = data['left']
                # right = data['right']

                # if self.args.task == 'depth':
                #     target_list = {'left': left, 'right': right}
                # elif self.args.task == 'seg' or self.args.task == 'both':
                #     left_label = data['left_label']
                #     right_label = data['right_label']
                #     left_index = data['left_index']
                #     right_index = data['right_index']
                #     target_list = {'left': left, 'right': right, 'left_label': left_label,
                #                    'right_label': right_label,
                #                    'left_index': left_index, 'right_index': right_index}
                # flip_right = torch.flip(right, [3])
                # left = data['image']
                #
                collection = self.model([left], self.args.task)

                # loss = self.loss_function(collection[0], data,
                #                           self.args.task, input_right=collection[1])
                prediction = nn.Softmax2d()(collection[3].unsqueeze(0)).squeeze().permute(1, 2,
                                                                                          0).detach().cpu().numpy()
                target = np.zeros(384 * 384).reshape((384, 384))
                for i in range(prediction.shape[0]):
                    for j in range(prediction.shape[1]):
                        anatomy = np.argmax(prediction[i][j])
                        if anatomy == 1:
                            target[i][j] = 80
                        elif anatomy == 2:
                            target[i][j] = 100
                        elif anatomy == 3:
                            target[i][j] = 140
                        elif anatomy == 4:
                            target[i][j] = 220
                # plt.imshow(target,cmap='gray')
                # plt.show()
                segmentations[idx] = target
                # print(prediction[0].shape)
                # exit(0)

                # right_prediction = collection[1].detach().cpu().numpy()
            np.save(self.output_directory + '/segmentation.npy', segmentations)
            print('Finished Testing')

    def test(self):
        self.model.eval()
        disparities = np.zeros((self.n_img,
                                self.input_height, self.input_width),
                               dtype=np.float32)
        disparities_pp = np.zeros((self.n_img,
                                   self.input_height, self.input_width),
                                  dtype=np.float32)
        with torch.no_grad():
            for i, data in enumerate(self.loader):
                # Get the inputs
                data = to_device(data, self.device)
                left = data.squeeze()
                # Do a forward pass
                collection = self.model([left], self.args.task)
                disp = collection[0][:, 0, :, :].unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
                disparities_pp[i] = \
                    post_process_disparity(collection[0][:, 0, :, :]
                                           .cpu().numpy())

        np.save(self.output_directory + '/disparities.npy', disparities)
        np.save(self.output_directory + '/disparities_pp.npy',
                disparities_pp)
        print('Finished Testing')


def main(args):
    args = return_arguments()
    if args.mode == 'val':
        model = Model(args)
        model.train()
    elif args.mode == '120_epochs':
        model_test = Model(args)
        model_test.test()


if __name__ == '__main__':
    main()
