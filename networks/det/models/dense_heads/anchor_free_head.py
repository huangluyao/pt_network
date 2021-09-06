from abc import abstractmethod
import torch
import torch.nn as nn
from .base_dense_head import BaseDenseHead
from base.cnn.components.conv_module import ConvModule
from base.cnn.components.blocks import C3
from ..builder import HEADS, build_loss
from base.cnn.utils import normal_init, bias_init_with_prob
from ..utils import multi_apply
@HEADS.register_module()
class AnchorFreeHead(BaseDenseHead):


    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=[4, 8 , 16, 32, 64],
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 loss_cls=dict(type='FocalLoss',
                               gamma=2.0,
                               alpha=0.25,
                               loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN2d'),
                 act_cfg=dict(type='ReLU'),
                 train_cfg=None,
                 test_cfg=None, **kwargs):

        super(AnchorFreeHead, self).__init__()
        self.num_classes = num_classes
        if isinstance(in_channels, int):
            self.in_channels = [in_channels for _ in range(len(strides))]
        else:
            self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.fp16_enabled = False

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

    def _init_cls_convs(self):
        """ stacked convs"""

        self.cls_convs = nn.ModuleList()
        for i in range(len(self.in_channels)):
            convs = []
            for j in range(self.stacked_convs):
                chn = self.in_channels[i] if j == 0 else self.feat_channels
                if self.dcn_on_last_conv and j == self.stacked_convs - 1:
                    conv_cfg = dict(type='DCNv2')
                else:
                    conv_cfg = self.conv_cfg
                convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias))
            self.cls_convs.append(nn.Sequential(*convs))

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""

        self.reg_convs = nn.ModuleList()
        for i in range(len(self.in_channels)):
            convs = []
            for j in range(self.stacked_convs):
                chn = self.in_channels[i] if j == 0 else self.feat_channels
                if self.dcn_on_last_conv and j == self.stacked_convs - 1:
                    conv_cfg = dict(type='DCNv2')
                else:
                    conv_cfg = self.conv_cfg
                convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=self.conv_bias))
            self.reg_convs.append(nn.Sequential(*convs))

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.ModuleList([nn.Conv2d(self.feat_channels, self.num_classes, 3, padding=1) for _ in range(len(self.in_channels))])
        self.conv_reg = nn.ModuleList([nn.Conv2d(self.feat_channels, 4, 3, padding=1)for _ in range(len(self.in_channels))])

    def init_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                normal_init(m, std=0.01)
        #     elif t is nn.BatchNorm2d:
        #         m.eps = 1e-3
        #         m.momentum = 0.03
        #     elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
        #         m.inplace = True
        for m in self.conv_cls:
            bias_cls = bias_init_with_prob(0.01)
            normal_init(m, std=0.01, bias=bias_cls)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually contain classification scores and bbox predictions.
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * 4.
        """
        return multi_apply(self.forward_single, feats, self.cls_convs, self.reg_convs, self.conv_cls, self.conv_reg)

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: Scores for each class, bbox predictions, features
                after classification and regression conv layers, some
                models needs these features like FCOS.
        """
        cls_feat = x
        reg_feat = x

        cls_feat = cls_convs(cls_feat)
        cls_score = conv_cls(cls_feat)

        reg_feat = reg_convs(reg_feat)
        bbox_pred = conv_reg(reg_feat)
        return cls_score, bbox_pred, cls_feat, reg_feat

    @abstractmethod
    def loss(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_bboxes(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_targets(self, **kwargs):
        raise NotImplementedError


    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        "Get points of a single scale level"

        h, w = featmap_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return y, x

    def get_points(self, featmap_sizes, dtype, device, flatten=False):

        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, device, flatten))
        return mlvl_points
