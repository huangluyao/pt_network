import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class DetailAggregateLoss(nn.Module):
    def __init__(self):
        super(DetailAggregateLoss, self).__init__()
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False)

        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                                                           dtype=torch.float32).reshape(1, 3, 1, 1))

    def forward(self, boundary_logits, seg_labels, dtype):
        self.laplacian_kernel = self.laplacian_kernel.type(dtype)
        gtmasks = seg_labels.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks, self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks, self.laplacian_kernel,
                                       stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        boundary_targets_x4 = F.conv2d(gtmasks, self.laplacian_kernel,
                                       stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks, self.laplacian_kernel,
                                       stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)

        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
                                               dim=1)

        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)

        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = self.dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return bce_loss,  dice_loss

    def dice_loss_func(self, input, target):
        smooth = 1.
        n = input.size(0)
        iflat = input.view(n, -1)
        tflat = target.view(n, -1)
        intersection = (iflat * tflat).sum(1)
        loss = 1 - ((2. * intersection + smooth) /
                    (iflat.sum(1) + tflat.sum(1) + smooth))
        return loss.mean()
