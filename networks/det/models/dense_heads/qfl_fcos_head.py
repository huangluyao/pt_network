import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.bbox import distance2bbox, multiclass_nms
from networks.det.models.builder import HEADS
from .anchor_free_head import AnchorFreeHead
from ..utils import multi_apply, Scale, bbox_overlaps
INF = 1e8


@HEADS.register_module()
class QFCOSHead(AnchorFreeHead):

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
        >>> self = QFCOSHead(11, 7)
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
                 loss_cls = dict(type='QualityFocalLoss', beta=2.0, loss_weight=1.0),
                 loss_bbox = dict(type='CIoULoss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):

        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        super(QFCOSHead, self).__init__(num_classes, in_channels,
                                       loss_cls=loss_cls,
                                       loss_bbox=loss_bbox,
                                       norm_cfg=norm_cfg,
                                       **kwargs
                                       )

        if kwargs['train_cfg'] is not None:
            self.debug = kwargs['train_cfg'].get('debug', False)
            output_dir = kwargs["train_cfg"].get("output_dir", None)
            if output_dir is not None and self.debug:
                self.output_file = os.path.join(output_dir, "vis_point")
                os.makedirs(self.output_file)
            else:
                self.debug = False
            self.ota = kwargs["train_cfg"].get("use_ota", False)
        else:
            self.debug = False
            self.output_file = None
            self.ota = False

    def _init_layers(self):
        super(QFCOSHead, self)._init_layers()
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        super(QFCOSHead, self).init_weights()

    def forward(self, feats):

        return multi_apply(self.forward_single, feats, self.scales, self.strides,
                           self.cls_convs, self.reg_convs, self.conv_cls, self.conv_reg)

    def forward_single(self, x, scale, stride,
                       cls_convs, reg_convs, conv_cls, conv_reg
                       ):

        cls_feat = x
        reg_feat = x

        cls_feat = cls_convs(cls_feat)
        cls_score = conv_cls(cls_feat)

        reg_feat = reg_convs(reg_feat)
        bbox_pred = conv_reg(reg_feat)

        bbox_pred = scale(bbox_pred).float()

        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred

    def loss(self, preds, gt_labels, **kwargs):

        cls_scores, bbox_preds, = preds[0], preds[1]

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
                image_name = os.path.split(gt_labels["image_path"][i])[-1]
                img = (img * std + mean).astype(np.uint8)
                pred_iou = pred_ious[i]

                # draw boxes
                for bbox in gt_bboxes_list:
                    bbox_f = np.array(bbox[:4], np.int32)
                    img = cv2.rectangle(img, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), (255, 255, 255), 1)

                # for each feature
                start_pos = 0
                end_pos = 0
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

                    score_map = []
                    # get cls scores map
                    cls_scores_maps= cls_scores[j].sigmoid()[i]
                    for cls_scores_map in cls_scores_maps:

                        score = cls_scores_map.detach().cpu().numpy()
                        score = (cv2.resize(score, img.shape[:2]) * 255).astype(np.uint8)
                        heat_img = cv2.cv2.applyColorMap(score, cv2.COLORMAP_JET)
                        img_add = cv2.addWeighted(_img, 0.3, heat_img, 0.7, 0)
                        score_map.append(img_add)

                    # get iou_score map
                    num_pos = pos_mask.sum()
                    end_pos = start_pos +num_pos
                    iou_score = cls_scores_maps.new_zeros(cls_scores_maps[0].shape)
                    if start_pos != end_pos:
                        iou_score[pos_mask] = pred_iou[start_pos:end_pos]
                        start_pos = end_pos
                    iou_score = iou_score.detach().cpu().numpy()
                    score = (cv2.resize(iou_score, img.shape[:2]) * 255).astype(np.uint8)
                    heat_img = cv2.cv2.applyColorMap(score, cv2.COLORMAP_JET)
                    img_add = cv2.addWeighted(_img, 0.3, heat_img, 0.7, 0)
                    score_map.append(img_add)

                    _img = np.concatenate(score_map, axis=1)
                    # save image
                    save_name = "feature_size_%dx%d_%s" % (featmap_sizes[j][0], featmap_sizes[j][1], image_name)
                    save_path = os.path.join(self.output_file, save_name)
                    cv2.imwrite(save_path, _img)

        pred_ious = torch.cat(pred_ious)
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
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
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        # pos_centerness_targets = self.centerness_target(pos_bbox_targets)

        centerness_denorm = max(pred_ious.sum().detach(), 1e-6)

        # pos_cls_preds = flatten_cls_scores[pos_inds]
        # pos_cls_targets = flatten_labels[pos_inds]
        # loss_cls = self.loss_cls(pos_cls_preds, pos_cls_targets, avg_factor=num_pos)
        score = flatten_labels.new_zeros(flatten_labels.shape)
        score[pos_inds] = pred_ious
        loss_cls = self.loss_cls(flatten_cls_scores, (flatten_labels, score), avg_factor=num_pos)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pred_ious,
                avg_factor=centerness_denorm)

        else:
            loss_bbox = pos_bbox_preds.sum()

        return dict(
                    loss_cls=loss_cls,
                    loss_bbox=loss_bbox,
                    )

    def get_bboxes(self, preds):
        cls_scores, bbox_preds = preds[0], preds[1]

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.shape[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]

        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       mlvl_points)

        if len(result_list)==0:
            pass

        return torch.cat(result_list, dim=0)

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
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

        for cls_score, bbox_pred, points in zip(
                cls_scores, bbox_preds, mlvl_points):
            assert cls_score.shape[-2:] == bbox_pred.shape[-2:]

            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.num_classes).sigmoid()

            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

            # Always keep topk op for dynamic input in onnx
            if nms_pre_tensor > 0 and (torch.onnx.is_in_onnx_export()
                                       or scores.shape[-2] > nms_pre_tensor):
                from torch import _shape_as_tensor
                # keep shape as tensor and get k
                num_anchor = _shape_as_tensor(scores)[-2].to(device)
                nms_pre = torch.where(nms_pre_tensor < num_anchor,
                                      nms_pre_tensor, num_anchor)

                max_scores, _ = (scores).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)

        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        # Set max number of box to be feed into nms in deployment
        deploy_nms_pre = cfg.get('deploy_nms_pre', -1)
        if deploy_nms_pre > 0 and torch.onnx.is_in_onnx_export():
            batch_mlvl_scores, _ = (
                batch_mlvl_scores
            ).max(-1)
            _, topk_inds = batch_mlvl_scores.topk(deploy_nms_pre)
            batch_inds = torch.arange(batch_mlvl_scores.shape[0]).view(
                -1, 1).expand_as(topk_inds)
            batch_mlvl_scores = batch_mlvl_scores[batch_inds, topk_inds, :]
            batch_mlvl_bboxes = batch_mlvl_bboxes[batch_inds, topk_inds, :]

        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         ):
                result = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg["score_thr"],
                    cfg["nms"],
                    cfg["max_per_img"])

                det_results.append(result)
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   )
            ]
        return det_results

    def get_targets(self, points, gt_labels, preds=None):

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
            cls_scores, bbox_preds = preds[0], preds[1]
            batch_size = cls_scores[0].shape[0]
            batch_cls_scoures = [cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
                                 for cls_score in cls_scores]
            batch_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
                                for bbox_pred in bbox_preds]

            batch_cls_scoures = torch.cat(batch_cls_scoures, dim=1)
            batch_bbox_preds = torch.cat(batch_bbox_preds, dim=1)

            outputs = torch.cat([batch_bbox_preds, batch_cls_scoures], dim=-1)
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

        return concat_lvl_labels, concat_lvl_bbox_targets, pred_ious_list

    @torch.no_grad()
    def _get_target_single(self, gt_bboxes, gt_labels, preds, points,
                           regress_ranges, num_points_per_lvl):

        if not isinstance(gt_bboxes, torch.Tensor):
            gt_bboxes = regress_ranges.new_tensor(gt_bboxes)
        if not isinstance(gt_labels, torch.Tensor):
            gt_labels = regress_ranges.new_tensor(gt_labels)

        num_points = points.shape[0]
        num_gts = gt_labels.shape[0]

        gt_bboxes_per_image = gt_bboxes
        if num_gts == 0:
            return gt_labels.new_full([num_points], self.num_classes), \
                   gt_bboxes.new_zeros([num_points, 4]), gt_bboxes.new_zeros(0)

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
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] >= 0
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

        if self.ota:
            fg_mask_inboxes, pred_ious_this_matching, gt_matched_classes = \
                self.get_assignments(preds, points, gt_bboxes_per_image, gt_labels, pos_inds,
                                 is_in_boxes_and_regression, num_gts)
            pos_inds[pos_inds.clone()] = fg_mask_inboxes
            labels[~pos_inds] = self.num_classes

        else:
            bbox_preds = preds[:, :4]  # [batch, n_anchors_all, 4]

            bbox_preds = distance2bbox(points, bbox_preds)[pos_inds]
            pair_wise_ious = bbox_overlaps(gt_bboxes_per_image, bbox_preds)
            pred_ious_this_matching = pair_wise_ious.max(0)[0] if len(bbox_preds) > 0 else bbox_preds.new_zeros(0)

        return labels, bbox_targets, pred_ious_this_matching

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        y, x = super(QFCOSHead, self)._get_points_single(featmap_size, stride,dtype,
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
        cls_preds = preds[:, 4:]  # [batch, n_anchors_all, n_cls]

        bbox_preds = distance2bbox(points, bbox_preds)[pos_inds]
        pair_wise_ious = bbox_overlaps(gt_bboxes_per_image, bbox_preds)

        cls_preds = cls_preds[pos_inds]
        obj_preds = pair_wise_ious.max(0)[0].unsqueeze(-1)
        num_in_boxes_anchor = bbox_preds.shape[0]

        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        gt_cls_per_image = F.one_hot(gt_labels.to(torch.int64), self.num_classes).float() \
                           .unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)

        cls_preds = cls_preds.float().unsqueeze(0).repeat(num_gts, 1, 1).sigmoid_()

        pair_wise_cls_loss = F.binary_cross_entropy(
            cls_preds.sqrt_(), gt_cls_per_image, reduction="none"
        ).sum(-1)

        cost = (pair_wise_cls_loss + 3 * pair_wise_ious_loss + 100000 * (~is_in_boxes_and_regression))
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

