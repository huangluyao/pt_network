# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/28 上午9:28
import torch
import torch.nn as nn
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from base.cnn.utils.weight_init import bias_init_with_prob, normal_init
from ..utils import multi_apply

@HEADS.register_module()
class CenterNetHead(BaseDenseHead):

    def __init__(self, in_channel, feat_channel, num_classes,
                 loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1_loss', loss_weight=1.0),
                 loss_offset=dict(type='L1_loss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs
                 ):
        super(CenterNetHead, self).__init__()

        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _build_head(self, in_channel, feat_channel, out_channel):
        layer = nn.Sequential(nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(feat_channel, out_channel, kernel_size=1)
                              )

        return layer

    def init_weights(self):
        """ initialize weights of the head """
        base_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(base_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        return center_heatmap_pred, wh_pred, offset_pred

    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             gt_labels,
             gt_bboxes_ignore=None):


        pass
