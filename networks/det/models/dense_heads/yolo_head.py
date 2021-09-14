import numpy as np

import torch
import torch.nn as nn

from base.cnn import ConvModule, normal_init

from .base_dense_head import BaseDenseHead
from networks.det.models.builder import HEADS
from ..utils import generate_yolov3_targets, yolov3_decoder, yolov3_losses
from ...specific import multi_apply


@HEADS.register_module()
class YOLOV3Head(BaseDenseHead):
    """YoloV3: https://arxiv.org/abs/1804.02767.

    Parameters
    ----------
    num_classes : int
        The number of object classes (w/o background)
    in_channels : List[int]
        Number of input channels per scale.
    out_channels : List[int]
        The number of output channels per scale
        before the final 1x1 layer. Default: (1024, 512, 256).
    anchor_cfg : dict
        Config dict for anchor generator
    featmap_strides : List[int]
        The stride of each scale.
        Should be in descending order. Default: (32, 16, 8).
    conv_cfg : dict
        Config dict for convolution layer. Default: None.
    norm_cfg : dict
        Dictionary to construct and config norm layer.
        Default: dict(type='BN', requires_grad=True)
    act_cfg : dict
        Config dict for activation layer.
        Default: dict(type='LeakyReLU', negative_slope=0.1).
    train_cfg : dict
        Training config of YOLOV3 head. Default: None.
    test_cfg : dict
        Testing config of YOLOV3 head. Default: None.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 out_channels=(1024, 512, 256),
                 anchor_cfg=dict(
                     base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(10, 13), (16, 30), (33, 23)]],
                     strides=[32, 16, 8]),
                 featmap_strides=[32, 16, 8],
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(YOLOV3Head, self).__init__()
        assert (len(in_channels) == len(out_channels) == len(featmap_strides))

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featmap_strides = np.array(featmap_strides)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.anchor_cfg = anchor_cfg
        self.num_anchors = len(self.anchor_cfg['base_sizes'][0])
        self.anchors = np.array(self.anchor_cfg['base_sizes']).astype(np.float32)
        self._init_layers()

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """number of attributes in pred_map:
        (dx, dy, log(dw), log(dh), confidence, prob0, ..., probN)
        """
        return 5 + self.num_classes

    def _init_layers(self):
        self.convs_bridge = nn.ModuleList()
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_bridge = ConvModule(
                self.in_channels[i],
                self.out_channels[i],
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            conv_pred = nn.Conv2d(self.out_channels[i],
                                  self.num_anchors * self.num_attrib, 1)

            self.convs_bridge.append(conv_bridge)
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.convs_pred:
            normal_init(m, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Parameters
        ----------
        feats : tuple[Tensor]
            Features from the upstream network, each is
            a 4D-tensor.

        Returns
        -------
        tuple[Tensor]
            A tuple of multi-level predication map, each is a
            4D-tensor of shape (batch_size, (5+num_classes)*3, height, width).
        """

        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            x = self.convs_bridge[i](x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)

        return tuple(pred_maps)

    def get_bboxes(self, pred_maps):
        """Transform network output for a batch into bbox predictions.

        Parameters
        ----------
        pred_maps : list[Tensor]
            Raw predictions for small, medium, large objections.
            [(B, (5+num_classes)*3, H_large, W_large),
             (B, (5+num_classes)*3, H_medium, W_medium),
             (B, (5+num_classes)*3, H_small, W_small)]

        Returns
        -------
        Tensor
            (B, K, (x1, y1, x2, y2, cls_id, confidence))
        """
        K = self.test_cfg.get('topk', 1000)

        results = []
        for i, pred_map in enumerate(pred_maps):
            N, C, H, W = pred_maps[i].shape
            result = yolov3_decoder(pred_map=pred_map,
                                    anchors=self.anchors[i],
                                    num_anchors=self.num_anchors,
                                    num_classes=self.num_classes,
                                    stride=self.anchor_cfg['strides'][i])
            results.append(result)
        results = torch.cat(results, dim=1)

        results, indices = torch.topk(results, K, dim=1, sorted=True)

        return results

    def _compose_bboxes(self, bboxes_list):
        _boxes_list = []
        for bboxes in bboxes_list:
            num_padding = self.test_cfg['instances_per_img'] - bboxes.shape[0]
            if  num_padding > 0:
                bboxes = np.pad(bboxes, ((0, num_padding), (0, 0)), mode='constant')
            _boxes_list.append(bboxes)

        return np.stack(_boxes_list, axis=0)

    def loss(self, pred_maps, ground_truth):
        targets = multi_apply(generate_yolov3_targets,
                              ground_truth['gt_bboxes'],
                              ground_truth['gt_labels'],
                              num_classes=self.num_classes,
                              anchors=self.anchors,
                              featmap_sizes=[fm.shape[3] for fm in pred_maps],
                              featmap_strides=self.featmap_strides)

        gt_bboxes = self._compose_bboxes(ground_truth['gt_bboxes'])

        all_level_losses = []
        #all_level_boxes_pred = []
        for i, pred_map in enumerate(pred_maps):
            losses = yolov3_losses(pred_map=pred_map,
                                   target=targets[i],
                                   num_classes=self.num_classes,
                                   anchors=self.anchors[i],
                                   stride=self.featmap_strides[i],
                                   gt_bboxes=gt_bboxes,
                                   pos_iou_thr=self.train_cfg['pos_iou_thr'])
            all_level_losses.append(losses)
            #all_level_boxes_pred.append(pred_bbox)

        all_level_losses = list(zip(*all_level_losses))
        loc_loss, conf_loss, cls_loss = [sum(losses) for losses in all_level_losses]

        return dict(
            loc_loss=loc_loss,
            conf_loss=conf_loss,
            cls_loss=cls_loss,
            #all_level_boxes_pred=all_level_boxes_pred,
            #all_level_targets=targets
        )
