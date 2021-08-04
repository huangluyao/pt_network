# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/7/31 上午10:58
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.bbox import bbox_overlaps
from .anchor_free_head import AnchorFreeHead
from ..utils import multi_apply, multiclass_nms
from ..builder import HEADS, build_loss


@HEADS.register_module()
class YOLOXHead(AnchorFreeHead):
    def __init__(self,
                 num_classes,
                 in_channels=[256, 512, 1024],
                 feat_channels=256,
                 loss_cls=dict(type='FocalLoss', gamma=2.0, alpha=0.25, loss_weight=1.0),
                 loss_obj=dict(type='FocalLoss', gamma=2.0, alpha=0.25, loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 norm_cfg=dict(type="BN2d"),
                 **kwargs
                 ):

        super(YOLOXHead, self).__init__(num_classes, in_channels,
                                       feat_channels=feat_channels,
                                       loss_cls=loss_cls,
                                       loss_bbox=loss_bbox,
                                       norm_cfg=norm_cfg,
                                       **kwargs
                                       )
        self.conv_obj = nn.ModuleList([nn.Conv2d(feat_channels, 1, 1) for _ in range(len(self.in_channels))])

        self.grids = [torch.zeros(1)] * len(self.in_channels)
        self.loss_obj = build_loss(loss_obj)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.conv_obj,
                           self.cls_convs, self.reg_convs, self.conv_cls, self.conv_reg)

    def forward_single(self, x, conv_obj,
                       cls_convs, reg_convs, conv_cls, conv_reg
                       ):

        cls_x = x
        reg_x = x

        cls_feat = cls_convs(cls_x)
        cls_score = conv_cls(cls_feat)

        reg_feat = reg_convs(reg_x)
        bbox_pred = conv_reg(reg_feat)
        obj_pred =  conv_obj(reg_feat)

        return cls_score, bbox_pred, obj_pred

    def loss(self, preds, gt_labels, **kwargs):
        cls_scores, reg_preds, obj_preds= preds[0], preds[1], preds[2]
        batch_size = cls_scores[0].shape[0]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, reg_preds[0].dtype,
                                           reg_preds[0].device)
        bbox_preds = []
        expanded_strides = []
        type = reg_preds[0].dtype
        for i, stride in enumerate(self.strides):
            yv, xv = all_level_points[i]
            hsize, wsize = featmap_sizes[i]
            grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(type)
            grid = grid.view(1, -1, 2)
            self.grids[i] = grid

            reg_preds[i] = reg_preds[i].permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            reg_preds[i][..., :2] = (reg_preds[i][..., :2] + grid) * stride
            reg_preds[i][..., 2:4] = torch.exp(reg_preds[i][..., 2:4]) * stride

            bbox_preds_l = (reg_preds[i][..., 0] - reg_preds[i][..., 2] * 0.5).unsqueeze(-1)
            bbox_preds_t = (reg_preds[i][..., 1] - reg_preds[i][..., 3] * 0.5).unsqueeze(-1)
            bbox_preds_r = (reg_preds[i][..., 0] + reg_preds[i][..., 2] * 0.5).unsqueeze(-1)
            bbox_preds_b = (reg_preds[i][..., 1] + reg_preds[i][..., 3] * 0.5).unsqueeze(-1)

            bbox_preds.append(torch.cat([bbox_preds_l, bbox_preds_t, bbox_preds_r, bbox_preds_b], dim=-1))
            expanded_strides.append(
                torch.zeros(1, grid.shape[1]).fill_(self.strides[i]).type_as(reg_preds[0]))

        cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
                      for cls_score in cls_scores]

        obj_preds = [obj_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
                      for obj_pred in obj_preds]

        output = [torch.cat([bbox_preds[i], obj_preds[i], cls_scores[i]], dim=-1) for i in range(len(bbox_preds))]

        cls_targets, reg_targets, obj_targets, fg_masks, num_fg_img = \
                            self.get_targets(output, gt_labels, expanded_strides)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_fg = sum(num_fg_img)
        num_fg = max(num_fg, 1)

        bbox_preds = torch.cat(bbox_preds, dim=1).view(-1, 4)
        cls_scores = torch.cat(cls_scores, dim=1).view(-1, self.num_classes)
        obj_preds = torch.cat(obj_preds, dim=1).view(-1, 1)

        loss_cls = self.loss_cls(cls_scores[fg_masks], cls_targets.detach(), avg_factor=num_fg)
        loss_iou = self.loss_bbox(bbox_preds[fg_masks], reg_targets, avg_factor=num_fg)
        loss_obj = self.loss_obj(obj_preds, obj_targets, avg_factor=num_fg)

        return dict(
                    loss_cls=loss_cls,
                    loss_bbox=loss_iou,
                    loss_obj=loss_obj
                    )

    def get_bboxes(self, preds, with_nms=True):

        cls_scores, reg_preds, obj_preds = preds[0], preds[1], preds[2]
        batch_size = cls_scores[0].shape[0]
        self.hw = [x.shape[-2:] for x in cls_scores]

        cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes) for cls_score in cls_scores]
        reg_preds = [reg_pred.permute(0, 2, 3, 1). reshape(batch_size, -1, 4) for reg_pred in reg_preds]
        obj_preds = [obj_pred.permute(0, 2, 3, 1).reshape(batch_size, -1 , 1) for obj_pred in obj_preds]

        dtype = cls_scores[0].type()
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        reg_preds = torch.cat(reg_preds, dim=1)
        cls_scores = torch.cat(cls_scores, dim=1)
        obj_preds = torch.cat(obj_preds, dim=1)

        reg_preds[..., :2] = (reg_preds[..., :2] + grids) * strides
        reg_preds[..., 2:4] = torch.exp(reg_preds[..., 2:4]) * strides

        bbox_preds_l = (reg_preds[..., 0] - reg_preds[..., 2] * 0.5).unsqueeze(-1)
        bbox_preds_t = (reg_preds[..., 1] - reg_preds[..., 3] * 0.5).unsqueeze(-1)
        bbox_preds_r = (reg_preds[..., 0] + reg_preds[..., 2] * 0.5).unsqueeze(-1)
        bbox_preds_b = (reg_preds[..., 1] + reg_preds[..., 3] * 0.5).unsqueeze(-1)

        bbox_preds = torch.cat([bbox_preds_l, bbox_preds_t, bbox_preds_r, bbox_preds_b], dim=-1)

        if with_nms:
            det_results = []
            for bbox_pred,  cls_score, obj_pred in zip(bbox_preds, cls_scores, obj_preds):
                result = multiclass_nms(
                    bbox_pred,
                    cls_score,
                    self.train_cfg["score_thr"],
                    self.train_cfg["nms"],
                    self.train_cfg["max_per_img"],
                    score_factors=obj_pred)
                det_results.append(result)

            return torch.cat(det_results, dim=0)
        out_put = torch.cat([bbox_preds, obj_preds, cls_scores], dim=-1)
        return out_put

    def get_targets(self, output, gt_labels, expanded_strides):
        gt_bboxes_list = gt_labels['bboxes']
        gt_labels_list = gt_labels['label_index']

        flatten_output = torch.cat(output, dim=1)
        batch_size, total_anchor = flatten_output.shape[:2]
        flatten_expanded_strides = torch.cat(expanded_strides, dim=1).expand(batch_size, total_anchor)

        return multi_apply(self.get_targets_single, flatten_output, flatten_expanded_strides, gt_labels_list, gt_bboxes_list)

    def get_targets_single(self, outputs, expanded_strides, gt_labels, gt_bboxes):
        assert len(gt_labels) == len(gt_bboxes), "the number of gt_labels must equal the number og gt_bboxes"
        total_num_anchors = outputs.shape[0]
        if not isinstance(gt_bboxes, torch.Tensor):
            gt_bboxes = outputs.new_tensor(gt_bboxes)
        if not isinstance(gt_labels, torch.Tensor):
            gt_labels = outputs.new_tensor(gt_labels)

        num_gt = len(gt_labels)
        if num_gt == 0:
            cls_target = outputs.new_zeros((0, self.num_classes))
            reg_target = outputs.new_zeros((0, 4))
            obj_target = outputs.new_zeros((total_num_anchors, 1))
            is_in_boxes_anchor = outputs.new_zeros(total_num_anchors).bool()
            num_fg_img = 0
        else:
            is_in_boxes_anchor, is_in_boxes_and_center = self.get_in_boxes_info(expanded_strides,
                                                                                num_gt, gt_bboxes, total_num_anchors)

            bboxes_preds_per_image = outputs[..., :4][is_in_boxes_anchor]
            num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
            pair_wise_ious = bbox_overlaps(gt_bboxes, bboxes_preds_per_image)
            pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

            gt_cls_per_image = (
                F.one_hot(gt_labels.to(torch.int64), self.num_classes).float()
                    .unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
            )

            cls_preds_ = outputs[..., 5:][is_in_boxes_anchor].clone()
            obj_preds_ = outputs[..., 4][is_in_boxes_anchor].unsqueeze(-1)
            cls_preds_ = (
                    cls_preds_.float().sigmoid_().unsqueeze(0).repeat(num_gt, 1, 1)
                    * obj_preds_.sigmoid_().unsqueeze(0).repeat(num_gt, 1, 1)
            )

            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)

            cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center)
            num_fg_img, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = \
                self.dynamic_k_matching(cost, pair_wise_ious, gt_labels, num_gt, is_in_boxes_anchor)

            del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

            cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
            obj_target = is_in_boxes_anchor.unsqueeze(-1).float()
            reg_target = gt_bboxes[matched_gt_inds]
        return cls_target, reg_target, obj_target, is_in_boxes_anchor, num_fg_img

    def get_in_boxes_info(self,expanded_strides, num_gt, gt_bboxes, total_num_anchors):

        # get anchor point
        grid = torch.cat(self.grids, dim=1).squeeze()
        x_shifts_per_image = grid[:, 0] * expanded_strides
        y_shifts_per_image = grid[:, 1] * expanded_strides
        x_centers_per_image = (x_shifts_per_image + 0.5 * expanded_strides).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image = (y_shifts_per_image + 0.5 * expanded_strides).unsqueeze(0).repeat(num_gt, 1)

        # calc gt box in anchor
        gt_bboxes_vs_anchor = gt_bboxes.unsqueeze(1).repeat(1, total_num_anchors, 1)
        b_l = x_centers_per_image - gt_bboxes_vs_anchor[..., 0]
        b_r = gt_bboxes_vs_anchor[..., 2] - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_vs_anchor[..., 1]
        b_b = gt_bboxes_vs_anchor[..., 3] - y_centers_per_image

        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # in fixed center
        center_radius = 2.5

        gt_center_x = ((gt_bboxes[..., 2] - gt_bboxes[..., 0]) * 0.5 + gt_bboxes[..., 0])\
            .unsqueeze(1).repeat(1, total_num_anchors)
        gt_center_y = ((gt_bboxes[..., 3] - gt_bboxes[..., 1]) * 0.5 + gt_bboxes[..., 1])\
            .unsqueeze(1).repeat(1, total_num_anchors)

        gt_bboxes_per_image_l = gt_center_x - center_radius*expanded_strides.unsqueeze(0)
        gt_bboxes_per_image_r = gt_center_x + center_radius*expanded_strides.unsqueeze(0)
        gt_bboxes_per_image_t = gt_center_y - center_radius*expanded_strides.unsqueeze(0)
        gt_bboxes_per_image_b = gt_center_y + center_radius*expanded_strides.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image

        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_labels, num_gt, is_in_boxes_anchor):
        matching_matrix = torch.zeros_like(cost)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = 10
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k,dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)

        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0

        fg_mask_inboxes = matching_matrix.sum(0) > 0.

        num_fg = fg_mask_inboxes.sum().item()
        is_in_boxes_anchor[is_in_boxes_anchor.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_labels[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
