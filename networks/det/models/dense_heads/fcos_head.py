import os
import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from ..utils.bbox import distance2bbox, multiclass_nms
from ..builder import HEADS, build_loss
from base.cnn.components.conv_module import ConvModule
from ....base.cnn.utils.weight_init import normal_init
from .anchor_free_head import AnchorFreeHead
from ..utils import multi_apply, Scale, bbox_overlaps
INF = 1e8

@HEADS.register_module()
class FCOSHead(AnchorFreeHead):

    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501


    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls = dict(type='FocalLoss', gamma=2.0,alpha=0.25, loss_weight=1.0),
                 loss_bbox = dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True,loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):

        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        super(FCOSHead, self).__init__(num_classes, in_channels,
                                       loss_cls=loss_cls,
                                       loss_bbox=loss_bbox,
                                       norm_cfg=norm_cfg,
                                       **kwargs
                                       )
        self.loss_centerness = build_loss(loss_centerness)

        if kwargs['train_cfg'] is not None:
            self.debug = kwargs['train_cfg'].get('debug', False)
            output_dir = kwargs["train_cfg"].get("output_dir", None)
            if output_dir is not None:
                self.output_file = os.path.join(output_dir, "vis_point")
                os.makedirs(self.output_file)
            else:
                self.debug = False
        else:
            self.debug = False
            self.output_file = None


    def _init_layers(self):
        super(FCOSHead, self)._init_layers()
        self.conv_centerness = nn.ModuleList([nn.Conv2d(self.feat_channels, 1, 3, padding=1) for _ in range(len(self.in_channels))])
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        super(FCOSHead, self).init_weights()
        for m in self.conv_centerness:
            normal_init(m, std=0.01)

    def forward(self, feats):

        return multi_apply(self.forward_single, feats, self.scales, self.strides, self.conv_centerness,
                           self.cls_convs, self.reg_convs, self.conv_cls, self.conv_reg)

    def forward_single(self, x, scale, stride, conv_centerness,
                       cls_convs, reg_convs, conv_cls, conv_reg
                       ):

        cls_feat = x
        reg_feat = x

        cls_feat = cls_convs(cls_feat)
        cls_score = conv_cls(cls_feat)

        reg_feat = reg_convs(reg_feat)
        bbox_pred = conv_reg(reg_feat)

        centerness = conv_centerness(reg_feat) if self.centerness_on_reg else conv_centerness(cls_feat)

        bbox_pred = scale(bbox_pred).float()

        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness

    def loss(self, preds, gt_labels, **kwargs):

        cls_scores, bbox_preds, centernesses = preds[0], preds[1], preds[2]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        num_imgs = cls_scores[0].shape[0]

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cls_score in cls_scores]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]

        labels, bbox_targets, pred_ious = self.get_targets(all_level_points, gt_labels, preds)

        if self.debug:
            import numpy as np
            import cv2
            circle_ratio = [2, 4, 6]
            batch_size = cls_scores[0].shape[0]
            for i in range(batch_size):
                gt_bboxes_list = gt_labels['bboxes'][i]
                img = gt_labels["image"][i].copy()
                mean = gt_labels["mean"][i]
                std = gt_labels["std"][i]
                image_name = gt_labels["image_path"][i].split('/')[-1]
                img = (img * std + mean).astype(np.uint8)

                # draw boxes
                for bbox in gt_bboxes_list:
                    bbox_f = np.array(bbox[:4], np.int32)
                    img = cv2.rectangle(img, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), (255, 255, 255), 1)

                # for each feature
                for j in range(len(labels)):
                    label = labels[j]
                    level_label = label[i * len(label) // batch_size:(i + 1) * len(label) // batch_size]
                    # print(bbox_target[(level_label >= 0) & (level_label < self.num_classes)])
                    level_label = level_label.view(featmap_sizes[j][0], featmap_sizes[j][1])
                    # get positive point mask
                    pos_mask = (level_label >= 0) & (level_label < self.num_classes)
                    pos_mask = pos_mask.detach().cpu().numpy().astype(np.uint8)
                    index = pos_mask.nonzero()
                    index_yx = np.stack(index, axis=1)
                    # resize to org image size
                    pos_img_yx = index_yx * self.strides[j] + self.strides[j] // 2
                    # draw point
                    _img = img.copy()
                    for z in range(pos_img_yx.shape[0]):
                        point = (int(pos_img_yx[z, 1]), int(pos_img_yx[z, 0]))
                        cv2.circle(_img, point, 1, (0, 0, 255), circle_ratio[j])

                    # save image
                    save_name = "feature_size_%dx%d_%s" % (featmap_sizes[j][0], featmap_sizes[j][1], image_name)
                    save_path = os.path.join(self.output_file, save_name)
                    cv2.imwrite(save_path, _img)


        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = ((flatten_labels >= 0) & (flatten_labels < self.num_classes)).nonzero().view(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = torch.clamp(num_pos, min=1.0)


        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)

        centerness_denorm = max(pos_centerness_targets.sum().detach(), 1e-6)

        # pos_cls_preds = flatten_cls_scores[pos_inds]
        # pos_cls_targets = flatten_labels[pos_inds]
        # loss_cls = self.loss_cls(pos_cls_preds, pos_cls_targets, avg_factor=num_pos)

        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
                    loss_cls=loss_cls,
                    loss_bbox=loss_bbox,
                    loss_centerness=loss_centerness
                    )

    def get_bboxes(self, preds):
        cls_scores, bbox_preds, centernesses = preds[0], preds[1], preds[2]

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.shape[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]

        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, mlvl_points)

        if len(result_list)==0:
            pass

        return torch.cat(result_list, dim=0)

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    mlvl_points,
                    rescale=False,
                    with_nms=True
                    ):

        cfg = self.train_cfg
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(cfg.get('nms_pre', -1), device=device, dtype=torch.long)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []

        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.shape[-2:] == bbox_pred.shape[-2:]

            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.num_classes).sigmoid()
            centerness = centerness.permute(0, 2, 3, 1).reshape(batch_size, -1).sigmoid()

            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

            # Always keep topk op for dynamic input in onnx
            if nms_pre_tensor > 0 and (torch.onnx.is_in_onnx_export()
                                       or scores.shape[-2] > nms_pre_tensor):
                from torch import _shape_as_tensor
                # keep shape as tensor and get k
                num_anchor = _shape_as_tensor(scores)[-2].to(device)
                nms_pre = torch.where(nms_pre_tensor < num_anchor,
                                      nms_pre_tensor, num_anchor)

                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]
                centerness = centerness[batch_inds, topk_inds]

            bboxes = distance2bbox(points, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)

        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)


        # Set max number of box to be feed into nms in deployment
        deploy_nms_pre = cfg.get('deploy_nms_pre', -1)
        if deploy_nms_pre > 0 and torch.onnx.is_in_onnx_export():
            batch_mlvl_scores, _ = (
                batch_mlvl_scores *
                batch_mlvl_centerness.unsqueeze(2).expand_as(batch_mlvl_scores)
            ).max(-1)
            _, topk_inds = batch_mlvl_scores.topk(deploy_nms_pre)
            batch_inds = torch.arange(batch_mlvl_scores.shape[0]).view(
                -1, 1).expand_as(topk_inds)
            batch_mlvl_scores = batch_mlvl_scores[batch_inds, topk_inds, :]
            batch_mlvl_bboxes = batch_mlvl_bboxes[batch_inds, topk_inds, :]
            batch_mlvl_centerness = batch_mlvl_centerness[batch_inds,
                                                          topk_inds]

        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness):
                result = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg["score_thr"],
                    cfg["nms"],
                    cfg["max_per_img"],
                    score_factors=mlvl_centerness)

                det_results.append(result)
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results

    def get_targets(self,points, gt_labels, preds=None):

        gt_bboxes_list = gt_labels['bboxes']
        gt_labels_list = gt_labels['label_index']

        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)

        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]

        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        num_points = [center.size(0) for center in points]

        if preds is not None:
            cls_scores, bbox_preds, centernesses = preds[0], preds[1], preds[2]
            batch_size = cls_scores[0].shape[0]
            batch_cls_scoures = [cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
                                 for cls_score in cls_scores]
            batch_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
                                for bbox_pred in bbox_preds]
            batch_centernesses = [centerness.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
                               for centerness in centernesses]

            batch_cls_scoures = torch.cat(batch_cls_scoures, dim=1)
            batch_bbox_preds = torch.cat(batch_bbox_preds, dim=1)
            batch_centernesses = torch.cat(batch_centernesses, dim=1)

            outputs = torch.cat([batch_bbox_preds, batch_centernesses, batch_cls_scoures], dim=-1)
        else:
            outputs = None
        # get labels and bbox_targets of each images
        labels_list, bbox_targets_list, pred_ious_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            outputs,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl = num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets =torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])

            concat_lvl_bbox_targets.append(bbox_targets)

        return concat_lvl_labels, concat_lvl_bbox_targets, torch.cat(pred_ious_list)

    def _get_target_single(self, gt_bboxes, gt_labels, preds, points,
                           regress_ranges, num_points_per_lvl):
        num_points = points.shape[0]
        num_gts = gt_labels.shape[0]

        if not isinstance(gt_bboxes, torch.Tensor):
            gt_bboxes = regress_ranges.new_tensor(gt_bboxes)
        if not isinstance(gt_labels, torch.Tensor):
            gt_labels = regress_ranges.new_tensor(gt_labels)

        gt_bboxes_per_image = gt_bboxes
        if num_gts == 0:
            return gt_labels.new_full([num_points], self.num_classes), \
                   gt_bboxes.new_zeros([num_points, 4])

        # compute boxes area
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)

        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        # condition 2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes      # set as Background
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        is_in_boxes_and_regression = (
            inside_gt_bbox_mask.T[:, (min_area != INF)] & inside_regress_range.T[:, (min_area != INF)]
        )

        pos_inds = (min_area != INF)

        fg_mask_inboxes, pred_ious_this_matching, gt_matched_classes = \
            self.get_assignments(preds, points, gt_bboxes_per_image, gt_labels, pos_inds,
                             is_in_boxes_and_regression, num_gts)

        pos_inds[pos_inds.clone()] = fg_mask_inboxes
        labels[~pos_inds] = self.num_classes

        return labels, bbox_targets, pred_ious_this_matching

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        y, x = super(FCOSHead, self)._get_points_single(featmap_size, stride,dtype,
                                                 device)

        points = torch.stack([x.reshape(-1) * stride, y.reshape(-1) * stride],dim=-1)+stride//2

        return points

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """

        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]

        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])

        return torch.sqrt(centerness_targets)


    def get_assignments(self, preds, points, gt_bboxes_per_image, gt_labels,
                        pos_inds, is_in_boxes_and_regression, num_gts):
        # calc cost
        bbox_preds = preds[:, :4]  # [batch, n_anchors_all, 4]
        obj_preds = preds[:, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = preds[:, 5:]  # [batch, n_anchors_all, n_cls]

        bbox_preds = distance2bbox(points, bbox_preds)[pos_inds]
        obj_preds = obj_preds[pos_inds]
        cls_preds = cls_preds[pos_inds]
        num_in_boxes_anchor = bbox_preds.shape[0]

        pair_wise_ious = bbox_overlaps(gt_bboxes_per_image, bbox_preds)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        gt_cls_per_image = F.one_hot(gt_labels.to(torch.int64), self.num_classes).float() \
                           .unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)

        cls_preds = cls_preds.float().unsqueeze(0).repeat(num_gts, 1, 1).sigmoid_() * \
                    obj_preds.unsqueeze(0).repeat(num_gts, 1, 1).sigmoid_()

        pair_wise_cls_loss = F.binary_cross_entropy(
            cls_preds.sqrt_(), gt_cls_per_image, reduction="none"
        ).sum(-1)

        cost = (pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000 * (~is_in_boxes_and_regression))
        matching_matrix = self.dynamic_k_matching(cost, pair_wise_ious)

        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        gt_matched_classes = gt_labels[matched_gt_inds]
        return fg_mask_inboxes, pred_ious_this_matching, gt_matched_classes

    def dynamic_k_matching(self, cost, pair_wise_ious):
        matching_matrix = torch.zeros_like(cost)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = 10
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_index in range(matching_matrix.shape[0]):
            _, pos_idx = torch.topk(cost[gt_index], dynamic_ks[gt_index].item(), largest=False)
            matching_matrix[gt_index][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0

        return matching_matrix





@HEADS.register_module()
class NASFCOSHead(FCOSHead):
    """Anchor-free head used in `NASFCOS <https://arxiv.org/abs/1906.04423>`_.

    It is quite similar with FCOS head, except for the searched structure of
    classification branch and bbox regression branch, where a structure of
    "dconv3x3, conv3x3, dconv3x3, conv1x1" is utilized instead.
    """

    def __init__(self, *args, init_cfg=None, **kwargs):
        if init_cfg is None:
            init_cfg = [
                dict(type='Caffe2Xavier', layer=['ConvModule', 'Conv2d']),
                dict(
                    type='Normal',
                    std=0.01,
                    override=[
                        dict(name='conv_reg'),
                        dict(name='conv_centerness'),
                        dict(
                            name='conv_cls',
                            type='Normal',
                            std=0.01,
                            bias_prob=0.01)
                    ]),
            ]
        super(NASFCOSHead, self).__init__(*args, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        dconv3x3_config = dict(
            type='DCNv2',
            kernel_size=3,
            use_bias=True,
            deform_groups=2,
            padding=1)
        conv3x3_config = dict(type='Conv', kernel_size=3, padding=1)
        conv1x1_config = dict(type='Conv', kernel_size=1)

        self.arch_config = [
            dconv3x3_config, conv3x3_config, dconv3x3_config, conv1x1_config
        ]
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i, op_ in enumerate(self.arch_config):
            op = copy.deepcopy(op_)
            chn = self.in_channels if i == 0 else self.feat_channels
            assert isinstance(op, dict)
            use_bias = op.pop('use_bias', False)
            padding = op.pop('padding', 0)
            kernel_size = op.pop('kernel_size')
            module = ConvModule(
                chn,
                self.feat_channels,
                kernel_size,
                stride=1,
                padding=padding,
                norm_cfg=self.norm_cfg,
                bias=use_bias,
                conv_cfg=op)

            self.cls_convs.append(copy.deepcopy(module))
            self.reg_convs.append(copy.deepcopy(module))

        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

