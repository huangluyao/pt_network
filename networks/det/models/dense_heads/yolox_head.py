# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/7/31 上午10:58
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.bbox import bbox_overlaps
from .anchor_free_head import AnchorFreeHead
from ..utils import multi_apply, multiclass_nms
from networks.det.models.builder import HEADS, build_loss


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

        if kwargs['train_cfg'] is not None:
            self.debug = kwargs['train_cfg'].get('debug', False)
            output_dir = kwargs["train_cfg"].get("output_dir", None)
            if output_dir is not None:
                self.output_file = os.path.join(output_dir, "vis_point")
                if os.path.exists(self.output_file):
                    os.makedirs(self.output_file)
            else:
                self.debug = False
        else:
            self.debug = False
            self.output_file = None

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
        obj_pred = conv_obj(reg_feat)

        return cls_score, bbox_pred, obj_pred

    def loss(self, preds, gt_labels, **kwargs):
        cls_scores, reg_preds, obj_preds= preds[0], preds[1], preds[2]
        batch_size = cls_scores[0].shape[0]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, reg_preds[0].dtype,
                                           reg_preds[0].device)
        outputs = []
        expanded_strides = []
        type = reg_preds[0].dtype
        for i, stride in enumerate(self.strides):
            output = torch.cat([reg_preds[i], obj_preds[i], cls_scores[i]], dim=1)
            grid = self.grids[i]
            hsize, wsize = featmap_sizes[i]
            if grid.shape[2:4] != output.shape[2:4]:
                yv, xv = all_level_points[i]
                grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(type)
                self.grids[i] = grid

            output = output.permute(0, 2, 3, 1).reshape(batch_size, hsize * wsize, -1)
            grid = grid.view(1, -1, 2)
            output[..., :2] = (output[..., :2] + grid) * stride
            output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
            expanded_strides.append(torch.zeros(hsize * wsize).fill_(stride).type_as(reg_preds[0]))
            outputs.append(output)

        expanded_strides = torch.cat(expanded_strides)
        outputs = torch.cat(outputs, 1)
        cls_targets, reg_targets, obj_targets, fg_masks, num_fg = self.get_targets(outputs, gt_labels, expanded_strides)

        if self.debug:
            import numpy as np
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
                feature_start = 0
                for j in range(len(self.strides)):
                    feature_end = feature_start + featmap_sizes[j][0] * featmap_sizes[j][1]
                    total_sample = len(fg_masks)
                    feature_labels = fg_masks[i * total_sample // batch_size:(i + 1) * total_sample // batch_size]
                    level_label = feature_labels[feature_start: feature_end]
                    feature_start = feature_end

                    pos_mask = level_label.view(featmap_sizes[j][0], featmap_sizes[j][1])
                    # get positive point mask
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

        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]

        # cx,cy,w,h => xyxy
        bbox_preds_l = bbox_preds[..., 0] - bbox_preds[..., 2] * 0.5
        bbox_preds_t = bbox_preds[..., 1] - bbox_preds[..., 3] * 0.5
        bbox_preds_r = bbox_preds[..., 0] + bbox_preds[..., 2] * 0.5
        bbox_preds_b = bbox_preds[..., 1] + bbox_preds[..., 3] * 0.5
        xyxy_bbox_preds = torch.stack([bbox_preds_l, bbox_preds_t, bbox_preds_r, bbox_preds_b], dim=-1)

        loss_iou = self.loss_bbox(xyxy_bbox_preds.view(-1, 4)[fg_masks], reg_targets, avg_factor=num_fg)
        loss_obj = self.loss_obj(obj_preds.view(-1, 1), obj_targets.to(type), avg_factor=num_fg)
        loss_cls = self.loss_cls(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets, avg_factor=num_fg)

        reg_weight = 5.0
        loss_iou = reg_weight * loss_iou

        return dict(
                    loss_cls=loss_cls,
                    loss_bbox=loss_iou,
                    loss_obj=loss_obj
                    )

    def get_targets(self,outputs, gt_labels, expanded_strides):
        gt_bboxes_list = gt_labels['bboxes']
        gt_labels_list = gt_labels['label_index']
        num_fg, cls_targets, obj_targets, reg_targets, fg_masks = \
        multi_apply(self.get_targets_single, outputs, gt_bboxes_list, gt_labels_list,
                    expanded_strides_per_image=expanded_strides
                    )
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_fg = max(sum(num_fg), 1)
        return cls_targets, reg_targets, obj_targets, fg_masks, num_fg

    @torch.no_grad()
    def get_targets_single(self,output, gt_bboxe, gt_label, expanded_strides_per_image):
        if not isinstance(gt_bboxe, torch.Tensor):
            gt_bboxe = output.new_tensor(gt_bboxe)
        if not isinstance(gt_label, torch.Tensor):
            gt_label = output.new_tensor(gt_label)
        total_num_anchors = output.shape[0]
        num_gt = len(gt_bboxe)
        if num_gt== 0:
            cls_target = output.new_zeros((0, self.num_classes))
            reg_target = output.new_zeros((0, 4))
            obj_target = output.new_zeros((total_num_anchors, 1))
            fg_mask = output.new_zeros(total_num_anchors).bool()
            num_fg = 0.
        else:
            bbox_preds = output[:, :4]  # [batch, n_anchors_all, 4]
            obj_preds = output[:, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
            cls_preds = output[:, 5:]
            fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxe, expanded_strides_per_image,
                                                                     total_num_anchors)
            bboxes_preds_per_image = bbox_preds[fg_mask]
            cls_preds_ = cls_preds[fg_mask]
            obj_preds_ = obj_preds[fg_mask]
            num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

            # cx,cy,w,h => xyxy
            bboxes_preds_per_image_l = bboxes_preds_per_image[..., 0] - bboxes_preds_per_image[..., 2] * 0.5
            bboxes_preds_per_image_t = bboxes_preds_per_image[..., 1] - bboxes_preds_per_image[..., 3] * 0.5
            bboxes_preds_per_image_r = bboxes_preds_per_image[..., 0] + bboxes_preds_per_image[..., 2] * 0.5
            bboxes_preds_per_image_b = bboxes_preds_per_image[..., 1] + bboxes_preds_per_image[..., 3] * 0.5
            xyxy_bboxes_preds_per_image = torch.stack([bboxes_preds_per_image_l, bboxes_preds_per_image_t, bboxes_preds_per_image_r, bboxes_preds_per_image_b], dim=-1)

            pair_wise_ious = bbox_overlaps(gt_bboxe, xyxy_bboxes_preds_per_image, False)
            pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

            gt_cls_per_image = (
                F.one_hot(gt_label.to(torch.int64), self.num_classes).float()
                .unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
            )

            cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)

            del cls_preds_

            cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_ious_loss
                    + 100000.0 * (~is_in_boxes_and_center)
            )
            num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = \
            self.dynamic_k_matching(cost, pair_wise_ious, gt_label, num_gt, fg_mask)

            cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
            obj_target = fg_mask.unsqueeze(-1)
            reg_target = gt_bboxe[matched_gt_inds]
        return num_fg, cls_target, obj_target, reg_target, fg_mask


    def get_in_boxes_info(self, gt_bboxe, expanded_strides_per_image, total_num_anchors):
        grids = torch.cat([grid.view(1, -1, 2) for grid in self.grids], dim=1)
        x_shifts_per_image = grids[..., 0][0] * expanded_strides_per_image
        y_shifts_per_image = grids[..., 1][0] * expanded_strides_per_image
        num_gt = len(gt_bboxe)
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = gt_bboxe[:, 0].unsqueeze(-1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_r = gt_bboxe[:, 2].unsqueeze(-1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_t = gt_bboxe[:, 1].unsqueeze(-1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_b = gt_bboxe[:, 3].unsqueeze(-1).repeat(1, total_num_anchors)

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image

        bbox_deltas = torch.stack([b_l, b_r, b_t, b_b], dim=2)
        is_in_bboxes = bbox_deltas.min(-1).values > 0
        is_in_bboxes_all = is_in_bboxes.sum(0) > 0

        center_strides = (expanded_strides_per_image * 2.5).unsqueeze(0)
        gt_bboxes_per_image_cx = gt_bboxes_per_image_l + (gt_bboxes_per_image_r - gt_bboxes_per_image_l)*0.5
        gt_bboxes_per_image_cy = gt_bboxes_per_image_t + (gt_bboxes_per_image_b - gt_bboxes_per_image_t)*0.5

        gt_bboxes_per_image_l = gt_bboxes_per_image_cx - center_strides
        gt_bboxes_per_image_r = gt_bboxes_per_image_cx + center_strides
        gt_bboxes_per_image_t = gt_bboxes_per_image_cy - center_strides
        gt_bboxes_per_image_b = gt_bboxes_per_image_cy + center_strides

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(-1).values > 0
        is_in_centers_all = is_in_centers.sum(0) > 0

        is_in_boxes_anchor = is_in_bboxes_all | is_in_centers_all
        is_in_boxes_and_center = is_in_bboxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):

        # Dynamic K
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = 10
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def get_bboxes(self, preds, with_nms=True):

        cls_scores, reg_preds, obj_preds = preds[0], preds[1], preds[2]
        batch_size = cls_scores[0].shape[0]
        self.hw = [x.shape[-2:] for x in cls_scores]

        cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes) for cls_score in cls_scores]
        reg_preds = [reg_pred.permute(0, 2, 3, 1). reshape(batch_size, -1, 4) for reg_pred in reg_preds]
        obj_preds = [obj_pred.permute(0, 2, 3, 1).reshape(batch_size, -1 , 1) for obj_pred in obj_preds]

        dtype = reg_preds[0].type()
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            strides.append(torch.full((1, hsize * wsize, 1), stride))
        grids = torch.cat([grid.view(1, -1, 2) for grid in self.grids], dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        reg_preds = torch.cat(reg_preds, dim=1)
        cls_scores = torch.cat(cls_scores, dim=1).sigmoid()
        obj_preds = torch.cat(obj_preds, dim=1).sigmoid()

        reg_preds[..., :2] = (reg_preds[..., :2] + grids) * strides
        reg_preds[..., 2:4] = torch.exp(reg_preds[..., 2:4]) * strides

        bbox_preds_l = (reg_preds[..., 0] - reg_preds[..., 2] * 0.5).unsqueeze(-1)
        bbox_preds_t = (reg_preds[..., 1] - reg_preds[..., 3] * 0.5).unsqueeze(-1)
        bbox_preds_r = (reg_preds[..., 0] + reg_preds[..., 2] * 0.5).unsqueeze(-1)
        bbox_preds_b = (reg_preds[..., 1] + reg_preds[..., 3] * 0.5).unsqueeze(-1)

        bbox_preds = torch.cat([bbox_preds_l, bbox_preds_t, bbox_preds_r, bbox_preds_b], dim=-1)
        padding = cls_scores.new_zeros(batch_size, cls_scores.shape[1], 1)
        cls_scores = torch.cat([cls_scores, padding], dim=-1)

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