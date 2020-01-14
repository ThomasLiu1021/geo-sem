import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N, H, W = target.size(0), target.size(1), target.size(2)
        smooth = 1

        input_flat = input.view(N, -1).float()
        target_flat = target.view(N, -1).float()

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):

        b = self.soft_dice_loss(y_true, y_pred)
        return b


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):

            diceLoss = dice(input[:, i, :, :], target[:, i, :, :])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


class MonodepthLoss(nn.modules.Module):
    def __init__(self, n=4, SSIM_w=0.85, disp_gradient_w=1.0, lr_w=1.0):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img,
                                                         size=[nh, nw], mode='bilinear',
                                                         align_corners=True))
        return scaled_imgs

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                                                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                                                     width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def seg_disp_smoothness(self, disp, pyramid):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(pyramid)
        image_gradients_y = self.gradient_y(pyramid)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))
        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y
        return torch.abs(smoothness_x) + torch.abs(smoothness_y)

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                                           keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                                           keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]
                        for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(self.n)]

    def seg_loss(self, input_left, target):
        '''

        :param input_left: un-softmax left image output seg
        :param input_right: un-softmax right image output seg
        :param target: [index_left, index_right, one_hot_left,one_hot_right]
        :return:
        '''

        # seg_left_pyramid = [d[:, 0, :, :].unsqueeze(1) for d in input_left[4:8]]
        # seg_right_pyramid = [d[:, 0, :, :].unsqueeze(1) for d in input_right[4:8]]
        #
        # target_left_pyramid = self.scale_pyramid(targe,t['left_label'].view(24, 1, 256, 256).float(), self.n)
        # target_right_pyramid = self.scale_pyramid(target['right_label'].view(24, 1, 256, 256).float(), self.n)
        #
        # left_ce = sum(
        #     [nn.CrossEntropyLoss()(seg_left_pyramid[i], target_left_pyramid[i].squeeze().long()) for i in
        #      range(self.n)]) / 4
        # right_ce = sum(
        #     [nn.CrossEntropyLoss()(seg_right_pyramid[i], target_right_pyramid[i].squeeze().long()) for i in
        #      range(self.n)]) / 4

        # total_ce = 0
        # total_dice = 0
        # total_var = 0
        # for i in input_left:
        #     left_ce = nn.CrossEntropyLoss()(i, target['left_label'].long())
        #     # right_ce = nn.CrossEntropyLoss()(input_right[0], target['right_label'].long())
        #
        #     input_left_softmax = nn.Softmax(dim=1)(i)
        #     # input_right_softmax = nn.Softmax2d()(input_right[0])
        #
        #     # intersection: torch.Tensor = torch.einsum("bcwh,bcwh->bc", input_left_softmax.type(torch.float32),
        #     #                                           target['left_index'].permute(0, 3, 1, 2).type(torch.float32))
        #     # union: torch.Tensor = (torch.einsum("bcwh->bc", target['left_index'].permute(0, 3, 1, 2).type(
        #     #     torch.float32)) + torch.einsum("bcwh->bc",
        #     #                                    input_left_softmax.type(
        #     #                                        torch.float32)))
        #     # mean_dice = (1 - (2 * intersection + 1) / (union + 1)).mean()
        #     power = 2
        #
        #     x_loss = torch.sum(torch.abs(input_left_softmax[:, :, 1:, :] - input_left_softmax[:, :, :-1, :]) ** power)
        #     y_loss = torch.sum((torch.abs(input_left_softmax[:, :, :, 1:] - input_left_softmax[:, :, :, :-1])) ** power)
        #
        #     tv_x_size = input_left_softmax[:, :, 1:, :].size()[1] * input_left_softmax[:, :, 1:, :].size()[2] * \
        #                 input_left_softmax[:, :, 1:, :].size()[3]
        #     tv_y_size = input_left_softmax[:, :, :, 1:].size()[1] * input_left_softmax[:, :, :, 1:].size()[2] * \
        #                 input_left_softmax[:, :, :, 1:].size()[3]
        #
        #     tv_loss = (x_loss / tv_x_size + y_loss / tv_y_size) / i[0]
        #     total_ce += left_ce
        #     total_dice += (MulticlassDiceLoss()(input_left_softmax, target['left_index'].permute(0, 3, 1, 2))) / 5
        #     total_var += tv_loss
        # dice_left = MulticlassDiceLoss()(input_left_softmax, target['left_index'].permute(0, 3, 1, 2)) / 5
        # dice_right = MulticlassDiceLoss()(input_right_softmax, target['right_index'].permute(0, 3, 1, 2)) / 5
        # return 0.1 * (left_ce + right_ce) / 2 + (dice_left + dice_right) / 2, [left_ce, right_ce], [dice_left,
        #                                                                                             dice_right]

        left_ce = nn.CrossEntropyLoss()(input_left, target['left_label'].long())
        input_left_softmax = nn.Softmax(dim=1)(input_left)
        # intersection: torch.Tensor = torch.einsum("bcwh,bcwh->bc", input_left_softmax.type(torch.float32),
        #                                           target['left_index'].permute(0, 3, 1, 2).type(torch.float32))
        # union: torch.Tensor = (torch.einsum("bcwh->bc", target['left_index'].permute(0, 3, 1, 2).type(
        #     torch.float32)) + torch.einsum("bcwh->bc",
        #                                    input_left_softmax.type(
        #                                        torch.float32)))
        # mean_dice = (1 - (2 * intersection + 1) / (union + 1)).mean()
        power = 2

        x_loss = torch.sum(torch.abs(input_left_softmax[:, :, 1:, :] - input_left_softmax[:, :, :-1, :]) ** power)
        y_loss = torch.sum((torch.abs(input_left_softmax[:, :, :, 1:] - input_left_softmax[:, :, :, :-1])) ** power)

        tv_x_size = input_left_softmax[:, :, 1:, :].size()[1] * input_left_softmax[:, :, 1:, :].size()[2] * \
                    input_left_softmax[:, :, 1:, :].size()[3]
        tv_y_size = input_left_softmax[:, :, :, 1:].size()[1] * input_left_softmax[:, :, :, 1:].size()[2] * \
                    input_left_softmax[:, :, :, 1:].size()[3]

        tv_loss = (x_loss / tv_x_size + y_loss / tv_y_size) / input_left.shape[0]
        # mean_dice = MulticlassDiceLoss()(input_left_softmax, target['left_index'].permute(0, 3, 1, 2)) / 5
        intersection: torch.Tensor = torch.einsum("bcwh,bcwh->bc", input_left_softmax.type(torch.float32),
                                                  target['left_index'].permute(0, 3, 1, 2).type(torch.float32))
        union: torch.Tensor = (torch.einsum("bcwh->bc", target['left_index'].permute(0, 3, 1, 2).type(
            torch.float32)) + torch.einsum("bcwh->bc", input_left_softmax.type(torch.float32)))
        mean_dice = (1 - (2 * intersection + 1) / (union + 1)).mean()
        # ssim = 0.85 * torch.mean(self.SSIM(torch.argmax(input_left_softmax, dim=1).unsqueeze(1).float(),
        #                                    target['left_label'].unsqueeze(1).float())) + 0.15 * torch.mean(
        #     torch.abs(torch.argmax(input_left_softmax, dim=1).unsqueeze(1).float() - target['left_label'].unsqueeze(
        #         1).float()))
        # return 0.5 * total_ce / len(input_left) + total_dice / len(input_left) + total_var / len(
        #     input_left), total_ce / len(input_left), total_dice / len(input_left)
        return left_ce * 0.5 + mean_dice + tv_loss, left_ce, mean_dice

    def depth_loss(self, input_left, input_right, target):
        """
                Args:
                    input [disp1, disp2, disp3, disp4]
                    target [left, right]

                Return:
                    (float): The loss
                """
        left = target['left_image']
        right = target['right_image']
        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)

        # Prepare disparities
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input_left]
        disp_right_est = [d[:, 0, :, :].unsqueeze(1) for d in input_right]

        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est
        # Generate images
        left_est = [self.generate_image_left(right_pyramid[i],
                                             disp_left_est[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(left_pyramid[i],
                                               disp_right_est[i]) for i in range(self.n)]
        self.left_est = left_est
        self.right_est = right_est

        # L-R Consistency
        right_left_disp = [self.generate_image_left(disp_right_est[i],
                                                    disp_left_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i],
                                                     disp_right_est[i]) for i in range(self.n)]

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_left_est,
                                                    left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est,
                                                     right_pyramid)

        # L1
        l1_left = [torch.mean(torch.abs(left_est[i] - left_pyramid[i]))
                   for i in range(self.n)]
        l1_right = [torch.mean(torch.abs(right_est[i]
                                         - right_pyramid[i])) for i in range(self.n)]

        # SSIM
        ssim_left = [torch.mean(self.SSIM(left_est[i],
                                          left_pyramid[i])) for i in range(self.n)]
        ssim_right = [torch.mean(self.SSIM(right_est[i],
                                           right_pyramid[i])) for i in range(self.n)]

        image_loss_left = [self.SSIM_w * ssim_left[i]
                           + (1 - self.SSIM_w) * l1_left[i]
                           for i in range(self.n)]
        image_loss_right = [self.SSIM_w * ssim_right[i]
                            + (1 - self.SSIM_w) * l1_right[i]
                            for i in range(self.n)]
        image_loss = sum(image_loss_left + image_loss_right)

        # L-R Consistency
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i]
                                             - disp_left_est[i])) for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i]
                                              - disp_right_est[i])) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_loss = [torch.mean(torch.abs(
            disp_left_smoothness[i])) / 2 ** i
                          for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(
            disp_right_smoothness[i])) / 2 ** i
                           for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        loss = image_loss + self.disp_gradient_w * disp_gradient_loss \
               + self.lr_w * lr_loss
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        return loss

    def combin(self, input_left, input_right, target):

        """
                        Args:
                            input [disp1, disp2, disp3, disp4]
                            target [left, right]

                        Return:
                            (float): The loss
                        """

        left = target['left_image']
        right = target['right_image']

        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)

        seg_left = nn.Softmax(dim=1)(input_left[4])
        seg_right = nn.Softmax(dim=1)(input_right[4])
        seg_left = torch.argmax(seg_left, dim=1).unsqueeze(1).float()
        seg_right = torch.argmax(seg_right, dim=1).unsqueeze(1).float()
        # seg_left = nn.Softmax2d()( input_left[4].unsqueeze(1))
        # seg_right = input_right[4].unsqueeze(1)
        # seg_left_pyramid = [d[:, 0, :, :].unsqueeze(1) for d in input_left[5]]
        # seg_right_pyramid = [d[:, 0, :, :].unsqueeze(1) for d in input_right[4:8]]

        # Prepare disparities
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input_left[0:4]]
        disp_right_est = [d[:, 0, :, :].unsqueeze(1) for d in input_right[0:4]]

        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est
        # Generate images
        left_est = [self.generate_image_left(right_pyramid[i],
                                             disp_left_est[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(left_pyramid[i],
                                               disp_right_est[i]) for i in range(self.n)]

        # Generate segmentation images
        left_seg_est = self.generate_image_left(seg_right, disp_left_est[0])
        right_seg_est = self.generate_image_right(seg_left, disp_right_est[0])
        # left_seg_est = [self.generate_image_left(seg_right_pyramid[i], disp_left_est[i]) for i in range(self.n)]
        # right_seg_est = [self.generate_image_right(seg_left_pyramid[i], disp_right_est[i]) for i in range(self.n)]
        # left_seg_est = self.generate_image_left(seg_right_est, disp_left_est[0])
        # right_seg_est = self.generate_image_right(seg_left_est, disp_right_est[0])

        self.left_est = left_est
        self.right_est = right_est

        # L-R Consistency
        right_left_disp = [self.generate_image_left(disp_right_est[i],
                                                    disp_left_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i],
                                                     disp_right_est[i]) for i in range(self.n)]

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_left_est,
                                                    left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est,
                                                     right_pyramid)

        seg_left_smoothness = self.seg_disp_smoothness(disp_left_est[0], seg_left)
        seg_right_smoothness = self.seg_disp_smoothness(disp_right_est[0], seg_right)

        # L1
        l1_left = [torch.mean(torch.abs(left_est[i] - left_pyramid[i]))
                   for i in range(self.n)]
        l1_right = [torch.mean(torch.abs(right_est[i]
                                         - right_pyramid[i])) for i in range(self.n)]

        # SSIM
        ssim_left = [torch.mean(self.SSIM(left_est[i],
                                          left_pyramid[i])) for i in range(self.n)]
        ssim_right = [torch.mean(self.SSIM(right_est[i],
                                           right_pyramid[i])) for i in range(self.n)]

        image_loss_left = [self.SSIM_w * ssim_left[i]
                           + (1 - self.SSIM_w) * l1_left[i]
                           for i in range(self.n)]
        image_loss_right = [self.SSIM_w * ssim_right[i]
                            + (1 - self.SSIM_w) * l1_right[i]
                            for i in range(self.n)]
        image_loss = sum(image_loss_left + image_loss_right)

        # L1 seg
        seg_l1_left = torch.mean(torch.abs(left_seg_est - seg_left))
        seg_l1_right = torch.mean(torch.abs(right_seg_est - seg_right))
        # seg_l1_left = [torch.mean(torch.abs(left_seg_est[i] - seg_left_pyramid[i]))
        #                for i in range(self.n)]
        # seg_l1_right = [torch.mean(torch.abs(right_seg_est[i]
        #                                      - seg_right_pyramid[i])) for i in range(self.n)]
        # SSIM seg
        seg_ssim_left = torch.mean(self.SSIM(left_seg_est, seg_left))
        seg_ssim_right = torch.mean(self.SSIM(right_seg_est, seg_right))
        # seg_ssim_left = [torch.mean(self.SSIM(left_seg_est[i],
        #                                       seg_left_pyramid[i])) for i in range(self.n)]
        # seg_ssim_right = [torch.mean(self.SSIM(right_seg_est[i],
        #                                        seg_right_pyramid[i])) for i in range(self.n)]
        seg_loss_left = self.SSIM_w * seg_ssim_left + (1 - self.SSIM_w) * seg_l1_left
        seg_loss_right = self.SSIM_w * seg_ssim_right + (1 - self.SSIM_w) * seg_l1_right
        # seg_loss_left = [self.SSIM_w * seg_ssim_left[i]
        #                  + (1 - self.SSIM_w) * seg_l1_left[i]
        #                  for i in range(self.n)]
        # seg_loss_right = [self.SSIM_w * seg_ssim_right[i]
        #                   + (1 - self.SSIM_w) * seg_l1_right[i]
        #                   for i in range(self.n)]
        seg_loss = seg_loss_left + seg_loss_right

        # L-R Consistency
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i]
                                             - disp_left_est[i])) for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i]
                                              - disp_right_est[i])) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_loss = [torch.mean(torch.abs(
            disp_left_smoothness[i])) / 2 ** i
                          for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(
            disp_right_smoothness[i])) / 2 ** i
                           for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        # semantic smoothness
        seg_left_loss = torch.mean(torch.abs(seg_left_smoothness[0] / 2 ** 0))
        seg_right_loss = torch.mean(torch.abs(seg_right_smoothness[0] / 2 ** 0))
        # seg_left_loss = [torch.mean(torch.abs(
        #     seg_left_smoothness[i])) / 2 ** i
        #                  for i in range(self.n)]
        # seg_right_loss = [torch.mean(torch.abs(
        #     seg_right_smoothness[i])) / 2 ** i
        #                   for i in range(self.n)]

        seg_graident_loss = seg_left_loss + seg_right_loss
        semantic_loss = seg_loss + self.disp_gradient_w * seg_graident_loss + self.lr_w * lr_loss
        disp_loss = image_loss + self.disp_gradient_w * disp_gradient_loss + self.lr_w * lr_loss
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss

        # left_ce = nn.CrossEntropyLoss()(input_left[4], target['left_label'].long())
        # right_ce = nn.CrossEntropyLoss()(input_right[0], target['right_label'].long())

        # input_left_softmax = nn.Softmax2d()(input_left[4])
        # input_right_softmax = nn.Softmax2d()(input_right[0])

        # dice_left = MulticlassDiceLoss()(input_left_softmax, target['left_index'].permute(0, 3, 1, 2)) / 5

        total_loss = semantic_loss + disp_loss
        return total_loss

    def forward(self, input_left, target, task, input_right=None):
        if task == 'depth':
            return self.depth_loss(input_left, input_right, target)
        elif task == 'seg':
            return self.seg_loss(input_left, target)
        elif task == 'both':
            return self.combin(input_left, input_right, target).float()
