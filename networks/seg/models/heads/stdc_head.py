# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/8/9 下午6:37
import torch
import torch.nn as nn
import torch.nn.functional as F
from .decode_head import BaseDecodeHead
from .bisenet_head import FeatureFusion, AttentionRefinement, SegHead
from base.cnn import ConvModule
from ..builder import HEADS
from ..criterions import DiceLoss

@HEADS.register_module()
class STDCHead(BaseDecodeHead):

    def __init__(self,
                 conv_out_inplanes=128,
                 final_drop=0,
                 **kwargs):
        super(STDCHead, self).__init__(**kwargs)

        # context path
        self.arm_names = []
        self.conv_head_names = []
        self.seg_head_names = []
        for index, in_channel in enumerate(self.in_channels[1:]):
            arm_name = "arm%d" % (index)
            conv_head_name = "conv_head%d" % (index)
            self.add_module(arm_name, AttentionRefinement(in_channel, conv_out_inplanes))
            self.add_module(conv_head_name, ConvModule(conv_out_inplanes, conv_out_inplanes, kernel_size=1,
                                                       stride=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
            self.arm_names.append(arm_name)
            self.conv_head_names.append(conv_head_name)

        for i in range(len(self.in_channels)):
            seg_head_name = "seg_head_name%d" % (i)
            self.add_module(seg_head_name, SegHead(conv_out_inplanes, self.num_classes,
                                                   conv_out_inplanes//2, final_drop=final_drop))
            self.seg_head_names.append(seg_head_name)

        self.conv_avg = ConvModule(self.in_channels[-1], conv_out_inplanes, kernel_size=1,
                                   stride=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.ffm = FeatureFusion(in_channels=self.in_channels[0]+conv_out_inplanes,
                                 out_channels=conv_out_inplanes)

        self.conv_out_sp8 = SegHead(self.in_channels[0], 1, 64)

        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False)

        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                                                           dtype=torch.float32).reshape(1, 3, 1, 1))

    def forward(self, inputs):

        H, W = inputs[-1].shape[2:]
        global_context = self.conv_avg(F.avg_pool2d(inputs[-1], (H, W)))
        global_context = F.interpolate(global_context, (H, W), mode='nearest')

        last_feature = global_context
        features = inputs[1:][::-1]
        pred_out = []
        for i, (feature, arm_model, conv_head) in enumerate(zip(features,
                                                        reversed(self.arm_names),
                                                        reversed(self.conv_head_names))):
            feature = getattr(self, arm_model)(feature)
            feature += last_feature
            H, W = H*2, W*2
            feature = F.interpolate(feature, (H, W), mode='nearest')
            last_feature = getattr(self, conv_head)(feature)
            pred_out.append(last_feature)
        context_out = last_feature

        feat_fuse = self.ffm(inputs[0], context_out)
        pred_out.append(feat_fuse)

        if self.training:
            seg_logit = [getattr(self, seg_head)(pred) for (seg_head, pred) in zip(self.seg_head_names, pred_out)]
            if len(seg_logit) == 1:
                return seg_logit[0]
            seg_logit.append(self.conv_out_sp8(inputs[0]))
            return seg_logit
        else:
            return getattr(self, self.seg_head_names[-1])(pred_out[-1])

    def losses(self, seg_logits, seg_labels):
        H, W = seg_labels.shape[-2:]
        outs = []
        for seg_logit in seg_logits[:-1]:
            outs.append(F.interpolate(seg_logit, (H, W), mode='bilinear', align_corners=True))

        for i, out in enumerate(outs):
            if i == 0 :
                loss = self.loss(out, seg_labels)
            else:
                loss += self.loss(out, seg_labels)

        bce_loss,  dice_loss = self.get_detail_aggregate_loss(seg_logits[-1], seg_labels, out.type())

        return dict(
            ce_loss=loss + bce_loss,
            dice_loss=dice_loss
        )

    def get_detail_aggregate_loss(self, boundary_logits, seg_labels, dtype):
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