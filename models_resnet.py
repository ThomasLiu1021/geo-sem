from __future__ import absolute_import, division, print_function

import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, d1, d2, skip=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.skip = skip

        self.conv1 = nn.Conv2d(in_channels, d1, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(d1)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(d1, d1, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(d1)

        self.conv3 = nn.Conv2d(d1, d2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(d2)

        if not self.skip:
            self.conv4 = nn.Conv2d(in_channels, d2, 1, stride=stride, bias=False)
            self.bn4 = nn.BatchNorm2d(d2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.skip:
            residual = x
        else:
            residual = self.conv4(x)
            residual = self.bn4(residual)

        out += residual
        out = self.relu(out)

        return out


class UpProj_Block(nn.Module):
    def __init__(self, in_channels, out_channels, batch_size):
        super(UpProj_Block, self).__init__()
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv4 = nn.Conv2d(in_channels, out_channels, (2, 2))

        self.conv5 = nn.Conv2d(in_channels, out_channels, (3, 3))
        self.conv6 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv7 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv8 = nn.Conv2d(in_channels, out_channels, (2, 2))

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def prepare_indices(self, before, row, col, after, dims):

        x0, x1, x2, x3 = np.meshgrid(before, row, col, after)
        dtype = torch.cuda.FloatTensor
        x_0 = torch.from_numpy(x0.reshape([-1]))
        x_1 = torch.from_numpy(x1.reshape([-1]))
        x_2 = torch.from_numpy(x2.reshape([-1]))
        x_3 = torch.from_numpy(x3.reshape([-1]))

        linear_indices = x_3 + dims[3] * x_2 + 2 * dims[2] * dims[3] * x_0 * 2 * dims[1] + 2 * dims[2] * dims[3] * x_1
        linear_indices_int = linear_indices.int()
        return linear_indices_int

    def forward(self, x, BN=True):
        out1 = self.unpool_as_conv(x, id=1)
        out1 = self.conv9(out1)

        if BN:
            out1 = self.bn2(out1)

        out2 = self.unpool_as_conv(x, ReLU=False, id=2)

        out = out1 + out2

        out = self.relu(out)
        return out

    def unpool_as_conv(self, x, BN=True, ReLU=True, id=1):
        if (id == 1):
            out1 = self.conv1(torch.nn.functional.pad(x, (1, 1, 1, 1)))
            out2 = self.conv2(torch.nn.functional.pad(x, (1, 1, 1, 0)))
            out3 = self.conv3(torch.nn.functional.pad(x, (1, 0, 1, 1)))
            out4 = self.conv4(torch.nn.functional.pad(x, (1, 0, 1, 0)))
        else:
            out1 = self.conv5(torch.nn.functional.pad(x, (1, 1, 1, 1)))
            out2 = self.conv6(torch.nn.functional.pad(x, (1, 1, 1, 0)))
            out3 = self.conv7(torch.nn.functional.pad(x, (1, 0, 1, 1)))
            out4 = self.conv8(torch.nn.functional.pad(x, (1, 0, 1, 0)))

        out1 = out1.permute(0, 2, 3, 1)
        out2 = out2.permute(0, 2, 3, 1)
        out3 = out3.permute(0, 2, 3, 1)
        out4 = out4.permute(0, 2, 3, 1)

        dims = out1.size()
        dim1 = dims[1] * 2
        dim2 = dims[2] * 2

        A_row_indices = range(0, dim1, 2)
        A_col_indices = range(0, dim2, 2)
        B_row_indices = range(1, dim1, 2)
        B_col_indices = range(0, dim2, 2)
        C_row_indices = range(0, dim1, 2)
        C_col_indices = range(1, dim2, 2)
        D_row_indices = range(1, dim1, 2)
        D_col_indices = range(1, dim2, 2)

        all_indices_before = range(int(self.batch_size))
        all_indices_after = range(dims[3])

        A_linear_indices = self.prepare_indices(all_indices_before, A_row_indices, A_col_indices, all_indices_after,
                                                dims)
        B_linear_indices = self.prepare_indices(all_indices_before, B_row_indices, B_col_indices, all_indices_after,
                                                dims)
        C_linear_indices = self.prepare_indices(all_indices_before, C_row_indices, C_col_indices, all_indices_after,
                                                dims)
        D_linear_indices = self.prepare_indices(all_indices_before, D_row_indices, D_col_indices, all_indices_after,
                                                dims)

        A_flat = (out1.permute(1, 0, 2, 3)).contiguous().view(-1)
        B_flat = (out2.permute(1, 0, 2, 3)).contiguous().view(-1)
        C_flat = (out3.permute(1, 0, 2, 3)).contiguous().view(-1)
        D_flat = (out4.permute(1, 0, 2, 3)).contiguous().view(-1)

        size_ = A_linear_indices.size()[0] + B_linear_indices.size()[0] + C_linear_indices.size()[0] + \
                D_linear_indices.size()[0]

        Y_flat = torch.cuda.FloatTensor(size_).zero_()

        Y_flat.scatter_(0, A_linear_indices.type(torch.cuda.LongTensor).squeeze(), A_flat.data)
        Y_flat.scatter_(0, B_linear_indices.type(torch.cuda.LongTensor).squeeze(), B_flat.data)
        Y_flat.scatter_(0, C_linear_indices.type(torch.cuda.LongTensor).squeeze(), C_flat.data)
        Y_flat.scatter_(0, D_linear_indices.type(torch.cuda.LongTensor).squeeze(), D_flat.data)

        Y = Y_flat.view(-1, dim1, dim2, dims[3])
        Y = Variable(Y.permute(0, 3, 1, 2))
        Y = Y.contiguous()

        if (id == 1):
            if BN:
                Y = self.bn1_1(Y)
        else:
            if BN:
                Y = self.bn1_2(Y)

        if ReLU:
            Y = self.relu(Y)

        return Y


class SkipUp(nn.Module):
    def __init__(self, in_size, out_size, scale):
        super(SkipUp, self).__init__()
        self.unpool = nn.Upsample(scale_factor=scale, mode='bilinear')
        self.conv = nn.Conv2d(in_size, out_size, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_size, out_size, 3, 1, 1)

    def forward(self, inputs):
        outputs = self.unpool(inputs)
        outputs = self.conv(outputs)
        # outputs = self.conv2(outputs)
        return outputs


##########################################


class Model(nn.Module):
    def __init__(self, block1, block2, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.encoder = torchvision.models.resnet50(pretrained=True)
        # Layers for Backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(3, stride=2)

        # self.proj_layer1 = self.make_proj_layer(block1, 64, d1=64, d2=256, stride=1)
        self.proj_layer1 = self.encoder.layer1
        self.skip_layer1_1 = self.make_skip_layer(block1, 256, d1=64, d2=256, stride=1)
        self.skip_layer1_2 = self.make_skip_layer(block1, 256, d1=64, d2=256, stride=1)

        # self.proj_layer2 = self.make_proj_layer(block1, 256, d1=128, d2=512, stride=2)
        self.proj_layer2 = self.encoder.layer2
        self.skip_layer2_1 = self.make_skip_layer(block1, 512, d1=128, d2=512)
        self.skip_layer2_2 = self.make_skip_layer(block1, 512, d1=128, d2=512)
        self.skip_layer2_3 = self.make_skip_layer(block1, 512, d1=128, d2=512)

        # self.proj_layer3 = self.make_proj_layer(block1, 512, d1=256, d2=1024, stride=2)
        self.proj_layer3 = self.encoder.layer3
        self.skip_layer3_1 = self.make_skip_layer(block1, 1024, d1=256, d2=1024)
        self.skip_layer3_2 = self.make_skip_layer(block1, 1024, d1=256, d2=1024)
        self.skip_layer3_3 = self.make_skip_layer(block1, 1024, d1=256, d2=1024)
        self.skip_layer3_4 = self.make_skip_layer(block1, 1024, d1=256, d2=1024)
        self.skip_layer3_5 = self.make_skip_layer(block1, 1024, d1=256, d2=1024)

        # self.proj_layer4 = self.make_proj_layer(block1, 1024, d1=512, d2=2048, stride=2)
        self.proj_layer4 = self.encoder.layer4
        self.skip_layer4_1 = self.make_skip_layer(block1, 2048, d1=512, d2=2048)
        self.skip_layer4_2 = self.make_skip_layer(block1, 2048, d1=512, d2=2048)

        self.conv2 = nn.Conv2d(2048, 1024, 1)
        self.bn2 = nn.BatchNorm2d(1024)

        # depth
        self.dep_up_conv1 = self.make_up_conv_layer(block2, 1024, 512, self.batch_size)
        self.dep_up_conv2 = self.make_up_conv_layer(block2, 512, 256, self.batch_size)
        self.dep_up_conv3 = self.make_up_conv_layer(block2, 256, 128, self.batch_size)
        self.dep_up_conv4 = self.make_up_conv_layer(block2, 128, 64, self.batch_size)
        self.dep_skip_up1 = SkipUp(512, 64, 8)
        self.dep_skip_up2 = SkipUp(256, 64, 4)
        self.dep_skip_up3 = SkipUp(128, 64, 2)
        self.dep_conv3 = nn.Conv2d(64, 2, 3, padding=1)  # 64,1
        # self.upsample = nn.UpsamplingBilinear2d(size = (480,640))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # sem
        self.sem_up_conv1 = self.make_up_conv_layer(block2, 1024, 512, self.batch_size)
        self.sem_up_conv2 = self.make_up_conv_layer(block2, 512, 256, self.batch_size)
        self.sem_up_conv3 = self.make_up_conv_layer(block2, 256, 128, self.batch_size)
        self.sem_up_conv4 = self.make_up_conv_layer(block2, 128, 64, self.batch_size)
        self.sem_skip_up1 = SkipUp(512, 64, 8)
        self.sem_skip_up2 = SkipUp(256, 64, 4)
        self.sem_skip_up3 = SkipUp(128, 64, 2)
        self.sem_conv3 = nn.Conv2d(64, 5, 3, padding=1)  # 64,1

        self.LS_d11 = nn.Conv2d(512, 512, 1)
        self.LS_s11 = nn.Conv2d(512, 512, 1)
        self.LS_d12 = nn.Conv2d(512, 512, 1)
        self.LS_s12 = nn.Conv2d(512, 512, 1)
        self.LS_d21 = nn.Conv2d(256, 256, 1)
        self.LS_s21 = nn.Conv2d(256, 256, 1)
        self.LS_d22 = nn.Conv2d(256, 256, 1)
        self.LS_s22 = nn.Conv2d(256, 256, 1)
        self.LS_d31 = nn.Conv2d(128, 128, 1)
        self.LS_s31 = nn.Conv2d(128, 128, 1)
        self.LS_d32 = nn.Conv2d(128, 128, 1)
        self.LS_s32 = nn.Conv2d(128, 128, 1)
        self.act = torch.nn.Sigmoid()

    def make_proj_layer(self, block, in_channels, d1, d2, stride=1, pad=0):
        return block(in_channels, d1, d2, skip=False, stride=stride)

    def make_skip_layer(self, block, in_channels, d1, d2, stride=1, pad=0):
        return block(in_channels, d1, d2, skip=True, stride=stride)

    def make_up_conv_layer(self, block, in_channels, out_channels, batch_size):
        return block(in_channels, out_channels, batch_size)

    def forward(self, x_1):
        out_1 = self.conv1(x_1)
        out_1 = self.bn1(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.max_pool(out_1)
        out_1 = self.proj_layer1(out_1)
        out_1 = self.skip_layer1_1(out_1)
        out_1 = self.skip_layer1_2(out_1)
        out_1 = self.proj_layer2(out_1)
        out_1 = self.skip_layer2_1(out_1)
        out_1 = self.skip_layer2_2(out_1)
        out_1 = self.skip_layer2_3(out_1)
        out_1 = self.proj_layer3(out_1)
        out_1 = self.skip_layer3_1(out_1)
        out_1 = self.skip_layer3_2(out_1)
        out_1 = self.skip_layer3_3(out_1)
        out_1 = self.skip_layer3_4(out_1)
        out_1 = self.skip_layer3_5(out_1)
        out_1 = self.proj_layer4(out_1)
        out_1 = self.skip_layer4_1(out_1)
        out_1 = self.skip_layer4_2(out_1)
        out_1 = self.conv2(out_1)
        out_1 = self.bn2(out_1)

        # Upconv section
        # Depth Prediction Branch
        dep_out_1up1 = self.dep_up_conv1(out_1)
        sem_out_1up1 = self.sem_up_conv1(out_1)
        LS_dep_out_1up1 = dep_out_1up1 + self.LS_d11(dep_out_1up1) + self.LS_s11(sem_out_1up1)
        LS_sem_out_1up1 = sem_out_1up1 + self.LS_s12(sem_out_1up1) + self.LS_d12(dep_out_1up1)

        dep_out_1up2 = self.dep_up_conv2(LS_dep_out_1up1)
        sem_out_1up2 = self.sem_up_conv2(LS_sem_out_1up1)
        LS_dep_out_1up2 = dep_out_1up2 + self.LS_d21(dep_out_1up2) + self.LS_s21(sem_out_1up2)
        LS_sem_out_1up2 = sem_out_1up2 + self.LS_s22(sem_out_1up2) + self.LS_d22(dep_out_1up2)

        dep_out_1up3 = self.dep_up_conv3(LS_dep_out_1up2)
        sem_out_1up3 = self.sem_up_conv3(LS_sem_out_1up2)
        LS_dep_out_1up3 = dep_out_1up3 + self.LS_d31(dep_out_1up3) + self.LS_s31(sem_out_1up3)
        LS_sem_out_1up3 = sem_out_1up3 + self.LS_s32(sem_out_1up3) + self.LS_d32(dep_out_1up3)

        dep_out_1 = self.dep_up_conv4(LS_dep_out_1up3)
        sem_out_1 = self.sem_up_conv4(LS_sem_out_1up3)

        dep_skipup1 = self.dep_skip_up1(dep_out_1up1)
        dep_skipup2 = self.dep_skip_up2(dep_out_1up2)
        dep_skipup3 = self.dep_skip_up3(dep_out_1up3)
        dep_out_1 = dep_out_1 + dep_skipup1 + dep_skipup2 + dep_skipup3
        dep_out_1 = 0.3 * self.act(self.dep_conv3(dep_out_1))
        dep_out_1 = self.upsample(dep_out_1)

        sem_skipup1 = self.sem_skip_up1(sem_out_1up1)
        sem_skipup2 = self.sem_skip_up2(sem_out_1up2)
        sem_skipup3 = self.sem_skip_up3(sem_out_1up3)
        sem_out_1 = sem_out_1 + sem_skipup1 + sem_skipup2 + sem_skipup3
        sem_out_1 = self.sem_conv3(sem_out_1)
        sem_out_1 = self.upsample(sem_out_1)

        return dep_out_1, sem_out_1


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return F.elu(x, inplace=True)


class seg_conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(seg_conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        x = F.relu(x, inplace=True)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        x = F.relu(x, inplace=True)
        # x = self.normalize(x)
        return x


class convblock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(convblock, self).__init__()
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, kernel_size, 2)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        p2d = (p, p, p, p)
        return F.max_pool2d(F.pad(x, p2d), self.kernel_size, stride=2)


class resconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 1, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, stride)
        self.conv3 = nn.Conv2d(num_out_layers, 4 * num_out_layers, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(num_in_layers, 4 * num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(4 * num_out_layers)

    def forward(self, x):
        # do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)
        # return F.elu(x_out + shortcut, inplace=True)


class resconv_basic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_basic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 3, stride)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, 1)
        self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        #         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if do_proj:
            shortcut = self.conv3(x)
        else:
            shortcut = x
        #  return F.elu(self.normalize(x_out + shortcut), inplace=True)
        return F.elu(x_out + shortcut, inplace=True)


def resblock(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks - 1):
        layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


def resblock_basic(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_basic(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.block = nn.Sequential(
            Interpolate(scale_factor=scale, mode='bilinear'),
            ConvRelu(num_in_layers, num_out_layers),
            ConvRelu(num_out_layers, num_out_layers)
        )
        # self.scale = scale
        # # self.conv1 =
        # self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        # x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        # # x = ConvRelu(x,)
        # return self.conv1(x)
        return self.block(x)


class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size=3, stride=1)
        # self.normalize = nn.BatchNorm2d(2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        # x = self.normalize(x)
        return 0.6 * self.sigmoid(x)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


def with_he_normal_weights(layer):
    nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
    return layer


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            with_he_normal_weights(nn.Conv2d(in_channels, out_channels, 3, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AlbUNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder

        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

        """

    def __init__(self, num_classes=5, num_filters=32, pretrained=True, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = self.pool

        self.encoder = torchvision.models.resnet50(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        # 3, 64
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        # 64,64
        # 64 256
        self.conv2 = self.encoder.layer1

        # 64,128
        # 256, 512
        self.conv3 = self.encoder.layer2

        # 128,256
        # 512 1024
        self.conv4 = self.encoder.layer3

        # 256, 512
        # 1024 2048
        self.conv5 = self.encoder.layer4

        # 512, 512, 256
        # 2048, 2048,1024
        self.center = DecoderBlockV2(2048, 2048, 1024, is_deconv)

        # 1024 + 2048, 2048, 1024
        self.dec5 = DecoderBlockV2(1024 + 2048, 2048, 1024, is_deconv)
        # 1024 + 1024, 1024, 512
        self.dec4 = DecoderBlockV2(1024 + 1024, 1024, 512, is_deconv)
        # 512 + 512, 512, 256
        self.dec3 = DecoderBlockV2(512 + 512, 512, 256, is_deconv)
        # 256 + 256, 256, 64
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2, is_deconv)
        # 64, 64,32
        self.dec1 = DecoderBlockV2(num_filters * 2, num_filters * 2, num_filters, is_deconv)
        # 32,32
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(32, 5, kernel_size=1)

    def forward(self, x, task):
        conv1 = self.conv1(x[0])
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        x_out = self.final(dec0)

        return [x_out]


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU6(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)
        # out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)
        # out = self.bn2(out)

        return out


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet50(pretrained=True)

        self.relu = nn.ReLU6(inplace=True)

        nb_filter = [64, 256, 512, 1024, 2048]

        self.conv0_0 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
        self.up = Interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv0_0 = VGGBlock(args.input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self.encoder.layer1
        self.conv2_0 = self.encoder.layer2
        self.conv3_0 = self.encoder.layer3
        self.conv4_0 = self.encoder.layer4

        # segmentation decoder
        self.seg_conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.seg_conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.seg_conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.seg_conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        # depth decoder
        self.disp_conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.disp_conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3] + 2, nb_filter[2], nb_filter[2])
        self.disp_conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2] + 2, nb_filter[1], nb_filter[1])
        self.disp_conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1] + 2, nb_filter[0], nb_filter[0])
        # depth decoder
        self.dec = ConvRelu(64, 32)
        self.disp1_layer = get_disp(64)
        self.disp2_layer = get_disp(256)
        self.disp3_layer = get_disp(512)
        self.disp4_layer = get_disp(1024)
        self.final1 = nn.Conv2d(32, 5, kernel_size=1)

    def forward(self, x, task):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)

        seg_x3_1 = self.seg_conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        seg_x2_2 = self.seg_conv2_2(torch.cat([x2_0, self.up(seg_x3_1)], 1))
        seg_x1_3 = self.seg_conv1_3(torch.cat([x1_0, self.up(seg_x2_2)], 1))
        seg_x0_4 = self.seg_conv0_4(torch.cat([x0_0, self.up(seg_x1_3)], 1))

        disp_x3_1 = self.disp_conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        disp4 = self.up(self.disp4_layer(disp_x3_1))
        disp_x2_2 = self.disp_conv2_2(torch.cat([x2_0, self.up(disp_x3_1), disp4], 1))
        disp3 = self.up(self.disp3_layer(disp_x2_2))
        disp_x1_3 = self.disp_conv1_3(torch.cat([x1_0, self.up(disp_x2_2), disp3], 1))
        disp2 = self.up(self.disp2_layer(disp_x1_3))
        disp_x0_4 = self.disp_conv0_4(torch.cat([x0_0, self.up(disp_x1_3), disp2], 1))
        disp1 = self.up(self.disp1_layer(disp_x0_4))
        if task == 'depth':
            return disp1, disp2, disp3, disp4
        elif task == 'seg':
            return self.final1(self.dec(self.up(seg_x0_4)))


class att4(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F, F_int):
        super(att4, self).__init__()

        self.W_g1 = nn.Sequential(
            nn.Conv2d(F[0], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F[1], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g1(g[0])
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class att3(nn.Module):
    def __init__(self, F, F_int):
        super(att3, self).__init__()

        self.W_g1 = nn.Sequential(
            nn.Conv2d(F[0], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F[1], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_g2 = nn.Sequential(
            nn.Conv2d(F[2], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g1(g[0])
        x1 = self.W_x(x)
        g2 = self.W_g2(g[1])
        psi = self.relu(g1 + x1 + g2)
        psi = self.psi(psi)
        out = x * psi
        return out


class att2(nn.Module):
    def __init__(self, F, F_int):
        super(att2, self).__init__()

        self.W_g1 = nn.Sequential(
            nn.Conv2d(F[0], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F[1], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_g2 = nn.Sequential(
            nn.Conv2d(F[2], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_g3 = nn.Sequential(
            nn.Conv2d(F[3], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g1(g[0])
        x1 = self.W_x(x)
        g2 = self.W_g2(g[1])
        g3 = self.W_g3(g[2])
        psi = self.relu(g1 + x1 + g2 + g3)
        psi = self.psi(psi)
        out = x * psi
        return out


class att1(nn.Module):
    def __init__(self, F, F_int):
        super(att1, self).__init__()

        self.W_g1 = nn.Sequential(
            nn.Conv2d(F[0], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F[1], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_g2 = nn.Sequential(
            nn.Conv2d(F[2], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_g3 = nn.Sequential(
            nn.Conv2d(F[3], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_g4 = nn.Sequential(
            nn.Conv2d(F[4], F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g1(g[0])
        x1 = self.W_x(x)
        g2 = self.W_g2(g[1])
        g3 = self.W_g3(g[2])
        g4 = self.W_g4(g[3])
        psi = self.relu(g1 + x1 + g2 + g3 + g4)
        psi = self.psi(psi)
        out = x * psi
        return out


class NestedUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet50(pretrained=True)

        self.relu = nn.ReLU(inplace=True)

        nb_filter = [64, 256, 512, 1024, 2048]

        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up = Interpolate(scale_factor=2, mode='bilinear')
        self.conv0_0 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
        self.conv1_0 = self.encoder.layer1
        self.conv2_0 = self.encoder.layer2
        self.conv3_0 = self.encoder.layer3
        self.conv4_0 = self.encoder.layer4

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.att3_1 = att4([nb_filter[4], nb_filter[3]], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3] + 2, nb_filter[2], nb_filter[2])
        self.att2_2 = att3([nb_filter[3], nb_filter[2], nb_filter[2]], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2] + 2, nb_filter[1], nb_filter[1])
        self.att1_3 = att2([nb_filter[2], nb_filter[1], nb_filter[1], nb_filter[1]], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1] + 2, nb_filter[0], nb_filter[0])
        self.att0_4 = att3([nb_filter[1], nb_filter[0], nb_filter[0], nb_filter[0]], nb_filter[0])
        self.dec = ConvRelu(64, 32)
        self.disp1_layer = get_disp(nb_filter[0])
        self.disp2_layer = get_disp(nb_filter[1])
        self.disp3_layer = get_disp(nb_filter[2])
        self.disp4_layer = get_disp(nb_filter[3])
        # self.dec = nn.Sequential(
        #     nn.Conv2d(64, 32, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(32)
        # )

        self.final1 = nn.Conv2d(32, 5, kernel_size=1)
        self.final2 = nn.Conv2d(32, 5, kernel_size=1)
        self.final3 = nn.Conv2d(32, 5, kernel_size=1)
        self.final4 = nn.Conv2d(32, 5, kernel_size=1)

    def forward_branch(self, input, branch, task):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(x3_0)
        att3_0 = self.att3_1([self.up(x4_0)], x3_0)
        x3_1 = self.conv3_1(torch.cat([att3_0, self.up(x4_0)], 1))
        self.disp4 = self.up(self.disp4_layer(x3_1))

        att2_1 = self.att2_2([self.up(x3_1), x2_1], x2_0)
        x2_2 = self.conv2_2(torch.cat([att2_1, self.up(x3_1), self.disp4], 1))
        self.disp3 = self.up(self.disp3_layer(x2_2))

        att1_2 = self.att1_3([self.up(x2_2), x1_2, x1_1], x1_0)
        x1_3 = self.conv1_3(torch.cat([att1_2, self.up(x2_2), self.disp3], 1))
        self.disp2 = self.up(self.disp2_layer(x1_3))

        att0_3 = self.att0_4([self.up(x1_3), x0_2, x0_3, x0_0], x0_0)
        x0_4 = self.conv0_4(torch.cat([att0_3, self.up(x1_3), self.disp2], 1))
        self.disp1 = self.up(self.disp1_layer(x0_4))

        self.seg_output1 = self.final1(self.dec(self.up(x0_1)))
        self.seg_output2 = self.final2(self.dec(self.up(x0_2)))
        self.seg_output3 = self.final3(self.dec(self.up(x0_3)))
        self.seg_output4 = self.final4(self.dec(self.up(x0_4)))
        if task == 'depth':
            if branch == 'left':
                return self.disp1, \
                       self.disp2, \
                       self.disp3, \
                       self.disp4
            elif branch == 'right':
                return torch.flip(self.disp1, [3]), \
                       torch.flip(self.disp2, [3]), \
                       torch.flip(self.disp3, [3]), \
                       torch.flip(self.disp4, [3])
        elif task == 'seg':
            if branch == 'left':
                return [self.seg_output4, self.seg_output3, self.seg_output2, self.seg_output1]
            elif branch == 'right':
                return [torch.flip(self.seg_output4, [3])]
        elif task == 'both':
            if branch == 'left':
                return self.disp1, \
                       self.disp2, \
                       self.disp3, \
                       self.disp4, \
                       self.seg_output1
            elif branch == 'right':
                return torch.flip(self.disp1, [3]), \
                       torch.flip(self.disp2, [3]), \
                       torch.flip(self.disp3, [3]), \
                       torch.flip(self.disp4, [3]), \
                       torch.flip(self.seg_output1, [3])
        # output1 = self.final1(self.dec(self.up(x0_1)))
        # output2 = self.final2(self.dec(self.up(x0_2)))
        # output3 = self.final3(self.dec(self.up(x0_3)))
        # output4 = self.final4(self.dec(self.up(x0_4)))
        # return [disp1, disp2, disp3, disp4],
        # return [output1, output2, output3]

    def forward(self, x, task):
        left_collection = self.forward_branch(x[0], 'left', task)
        # return [left_collection]
        if task != 'seg':
            right_collection = self.forward_branch(x[1], 'right', task)
            return [left_collection, right_collection]
        else:
            return [left_collection]


class Resnet50_md(nn.Module):
    def __init__(self, num_in_layers):
        super(Resnet50_md, self).__init__()
        # encoder
        self.conv1 = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1 = maxpool(3)  # H/4  -   64D
        self.conv2 = resblock(64, 64, 3, 2)  # H/8  -  256D
        self.conv3 = resblock(256, 128, 4, 2)  # H/16 -  512D
        self.conv4 = resblock(512, 256, 6, 2)  # H/32 - 1024D
        self.conv5 = resblock(1024, 512, 3, 2)  # H/64 - 2048D

        # decoder
        self.upconv6 = upconv(2048, 512, 3, 2)
        self.iconv6 = conv(1024 + 512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(512 + 256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(256 + 128, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64 + 64 + 2, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(32 + 64 + 2, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16 + 2, 16, 3, 1)
        self.disp1_layer = get_disp(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward_branch(self, input, branch, task):
        # encoder
        x1 = self.conv1(input)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4

        # decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        if task == 'depth':
            if branch == 'left':
                return [self.disp1, self.disp2, self.disp3, self.disp4]
            elif branch == 'right':
                return [torch.flip(self.disp1, [3]), torch.flip(self.disp2, [3]), torch.flip(self.disp3, [3]),
                        torch.flip(self.disp4, [3])]
        # return self.disp1, self.disp2, self.disp3, self.disp4

    def forward(self, x, task):
        left_collection = self.forward_branch(x[0], 'left', task)
        # return [left_collection]
        if task != 'seg':
            right_collection = self.forward_branch(x[1], 'right', task)
            return [left_collection, right_collection]
        else:
            return [left_collection]


class Resnet18_md(nn.Module):
    def __init__(self, num_in_layers):
        super(Resnet18_md, self).__init__()
        # encoder
        self.conv1 = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1 = maxpool(3)  # H/4  -   64D
        self.conv2 = resblock_basic(64, 64, 2, 2)  # H/8  -  64D
        self.conv3 = resblock_basic(64, 128, 2, 2)  # H/16 -  128D
        self.conv4 = resblock_basic(128, 256, 2, 2)  # H/32 - 256D
        self.conv5 = resblock_basic(256, 512, 2, 2)  # H/64 - 512D

        # decoder
        self.upconv6 = upconv(512, 512, 3, 2)
        self.iconv6 = conv(256 + 512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(128 + 256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(64 + 128, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64 + 64 + 2, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64 + 32 + 2, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16 + 2, 16, 3, 1)
        self.disp1_layer = get_disp(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4

        # decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        return self.disp1, self.disp2, self.disp3, self.disp4


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)


class ResnetModel(nn.Module):
    def __init__(self, num_in_layers, encoder='resnet18', pretrained=True):
        super(ResnetModel, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', \
                           'resnet101', 'resnet152'], \
            "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        resnet = class_for_name("torchvision.models", encoder) \
            (pretrained=pretrained)
        if num_in_layers != 3:  # Number of input channels
            self.firstconv = nn.Conv2d(num_in_layers, 64,
                                       kernel_size=(7, 7), stride=(2, 2),
                                       padding=(3, 3), bias=False)
        else:
            self.firstconv = resnet.conv1  # H/2
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # H/4

        # encoder
        self.encoder1 = resnet.layer1  # H/4
        self.encoder2 = resnet.layer2  # H/8
        self.encoder3 = resnet.layer3  # H/16
        self.encoder4 = resnet.layer4  # H/32

        # decoder
        self.upconv6 = upconv(filters[3], 512, 3, 2)
        self.iconv6 = conv(filters[2] + 512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(filters[1] + 256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(filters[0] + 128, 128, 3, 1)
        # self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 1)  #
        self.iconv3 = conv(64 + 64 + 2, 64, 3, 1)
        # self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64 + 32 + 2, 32, 3, 1)
        # self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16, 16 + 2, 3, 1)
        # self.disp1_layer = get_disp(16)

        self.output1 = nn.Conv2d(16, 5, 1)
        self.seg_output = seg_conv(5, 5, 1, 1)

        self.output2 = nn.Conv2d(32, 5, 1)

        self.output3 = nn.Conv2d(64, 5, 1)

        self.output4 = nn.Conv2d(128, 5, 1)
        self.disp_layer = get_disp(5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward_branch(self, x, branch, task):
        # encoder
        x_first_conv = self.firstconv(x)
        x = self.firstbn(x_first_conv)
        x = self.firstrelu(x)
        x_pool1 = self.firstmaxpool(x)
        x1 = self.encoder1(x_pool1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        # skips
        skip1 = x_first_conv
        skip2 = x_pool1
        skip3 = x1
        skip4 = x2
        skip5 = x3

        # decoder
        upconv6 = self.upconv6(x4)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        # self.seg4 = self.output4(iconv4)
        # self.seg_output4 = self.seg_output(self.seg4)
        # self.disp4 = self.disp_layer(self.seg4)
        # self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=1, mode='bilinear', align_corners=True)
        # self.disp4 = nn.functional.interpolate(self.disp4, scale_factor=0.5, mode='bilinear', align_corners=True)
        # self.useg4 = nn.functional.interpolate(nn.Softmax2d()(self.seg_output4), scale_factor=1, mode='bilinear',
        #                                        align_corners=True)
        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2), 1)
        iconv3 = self.iconv3(concat3)
        # self.seg3 = self.output3(iconv3)
        # self.seg_output3 = self.seg_output(self.seg3)
        # self.disp3 = self.disp_layer(self.seg3)
        # self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)
        # self.useg3 = nn.functional.interpolate(nn.Softmax2d()(self.seg_output3), scale_factor=2, mode='bilinear',
        #                                        align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1), 1)
        iconv2 = self.iconv2(concat2)
        # self.seg2 = self.output2(iconv2)
        # self.seg_output2 = self.seg_output(self.seg2)
        # self.disp2 = self.disp_layer(self.seg2)
        # self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)
        # self.useg2 = nn.functional.interpolate(nn.Softmax2d()(self.seg_output2), scale_factor=2, mode='bilinear',
        #                                        align_corners=True)

        upconv1 = self.upconv1(iconv2)
        # concat1 = torch.cat((upconv1), 1)
        iconv1 = self.iconv1(upconv1)
        self.seg1 = self.output1(iconv1)
        self.seg_output1 = self.seg_output(self.seg1)
        self.disp1 = self.disp_layer(self.seg1)
        # self.disp1 = self.disp1_layer(iconv1)
        if task == 'depth':
            if branch == 'left':
                return self.disp1, \
                       self.disp2, \
                       self.disp3, \
                       self.disp4
            elif branch == 'right':
                return torch.flip(self.disp1, [3]), \
                       torch.flip(self.disp2, [3]), \
                       torch.flip(self.disp3, [3]), \
                       torch.flip(self.disp4, [3])
        elif task == 'seg':
            if branch == 'left':
                return self.seg_output1
                # self.seg_output2, \
                # self.seg_output3, \
                # self.seg_output4
            elif branch == 'right':
                return torch.flip(self.seg_output1, [3]), \
                       torch.flip(self.seg_output2, [3]), \
                       torch.flip(self.seg_output3, [3]), \
                       torch.flip(self.seg_output4, [3])
        elif task == 2:
            if branch == 'left':
                return self.disp1, \
                       self.disp2, \
                       self.disp3, \
                       self.disp4, \
                       self.seg_output1, \
                       self.seg_output2, \
                       self.seg_output3, \
                       self.seg_output4
            elif branch == 'right':
                return torch.flip(self.disp1, [3]), \
                       torch.flip(self.disp2, [3]), \
                       torch.flip(self.disp3, [3]), \
                       torch.flip(self.disp4, [3]), \
                       torch.flip(self.seg_output1, [3]), \
                       torch.flip(self.seg_output2, [3]), \
                       torch.flip(self.seg_output3, [3]), \
                       torch.flip(self.seg_output4, [3])

    def forward(self, x, task):
        left_collection = self.forward_branch(x[0], 'left', task)
        return [left_collection]
        if task != 'seg':
            right_collection = self.forward_branch(x[1], 'right', task)
            return [left_collection, right_collection]
        else:
            return [left_collection]
