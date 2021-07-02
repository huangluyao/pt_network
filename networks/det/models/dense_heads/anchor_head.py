# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/29 上午9:49
import torch
import torch.nn as nn
from .base_dense_head import BaseDenseHead
from ..builder import build_loss, HEADS
from ..utils.bbox import build_bbox_coder, build_sampler, build_assigner, multiclass_nms
from ..utils.anchor import build_anchor_generator
from ..utils import multi_apply


@HEADS.register_module()
class AnchorHead(BaseDenseHead):

    def __init__(self, num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     strides=[4, 8, 16, 32, 64]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=True,
                    target_means=(.0, .0, .0, .0),
                    target_stds=(1.0, 1.0, 1.0, 1.0)),
                reg_decoded_bbox=False,
                loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                train_cfg=None,
                test_cfg=None,
                 **kwargs):

        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        # TODO better way to determine whether sample or not
        self.sampling = loss_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg["assigner"])
            # use PseudoSampler when sampling is False
            if self.sampling and 'sampler' in self.train_cfg:
                sampler_cfg = self.train_cfg["sampler"]
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.anchor_generator = build_anchor_generator(anchor_generator)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv2d(self.in_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 4, 1)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """Forward feature of a single scale level.
        Args:
            x (Tensor): Features of a single scale level.
        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_anchors * 4.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def get_anchors(self, featmap_sizes, img_info, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_info (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_info["bboxes"])

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_info['src_shape']):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta, device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def get_bboxes(self,
                   preds,
                   cfg=None,
                   rescale=False,
                   with_nms=True
                   ):
        cls_scores = preds[0]
        bbox_preds = preds[1]

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, rescale, with_nms)
        if len(result_list)==0:
            pass

        return torch.cat(result_list, dim=0)

    def _get_bboxes(self,
                    mlvl_cls_scores,
                    mlvl_bbox_preds,
                    mlvl_anchors,
                    rescale=False,
                    with_nms=True
                    ):

        cfg = self.test_cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
            mlvl_anchors)

        batch_size = mlvl_cls_scores[0].shape[0]
        max_per_img = cfg.get('max_per_img', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores,
                                                 mlvl_bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            anchors = anchors.expand_as(bbox_pred)

            if self.use_sigmoid_cls:
                max_scores, _ = scores.max(-1)
            else:
                max_scores, _ = scores[..., :-1].max(-1)

            max_per_img = min(max_per_img, len(max_scores[0]))
            nms_pre_tensor = torch.tensor(
                max_per_img,
                device=mlvl_cls_scores[0].device,
                dtype=torch.long)
            _, topk_inds = max_scores.topk(nms_pre_tensor)
            batch_inds = torch.arange(batch_size).view(
                -1, 1).expand_as(topk_inds)

            anchors = anchors[batch_inds, topk_inds, :]
            bbox_pred = bbox_pred[batch_inds, topk_inds, :]
            scores = scores[batch_inds, topk_inds, :]

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)

        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = batch_mlvl_scores.new_zeros(batch_size,
                                                  batch_mlvl_scores.shape[1],
                                                  1)
            batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes,
                                                  batch_mlvl_scores):
                result = multiclass_nms(mlvl_bboxes,
                                         mlvl_scores,
                                         cfg["score_thr"],
                                         cfg["nms"],
                                         cfg["max_per_img"])
                det_results.append(result)
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]
        return det_results

