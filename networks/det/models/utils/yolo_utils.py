import numpy as np

import torch
import torch.nn.functional as F


def _xywh_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / (union_area + 1e-6)


def generate_yolov3_targets(gt_bboxes,
                            gt_labels,
                            num_classes,
                            anchors,
                            featmap_sizes,
                            featmap_strides):
    num_gts = len(gt_bboxes)
    num_attrib = 5 + num_classes
    num_anchors = anchors[0].shape[0]
    num_levels = len(featmap_strides)

    targets = [np.zeros((num_anchors,
                         num_attrib,
                         featmap_sizes[i],
                         featmap_sizes[i])) for i in range(3)]

    for gt_idx in range(num_gts):
        bbox_coor = gt_bboxes[gt_idx]
        class_id = int(gt_labels[gt_idx])

        onehot_labels = np.zeros(num_classes, dtype=np.float)
        onehot_labels[class_id-1] = 1.0
        #uniform_distribution = np.full(num_classes, 1.0 / num_classes)
        #deta = 0.01
        #smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

        bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                    bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / featmap_strides[:, np.newaxis]

        ious = []
        exist_positive = False
        for lvl_idx in range(num_levels):
            anchors_xywh = np.zeros((num_anchors, 4))
            anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[lvl_idx, 0:2]).astype(np.int32) + 0.5
            anchors_xywh[:, 2:4] = anchors[lvl_idx] / featmap_strides[lvl_idx]

            iou_scale = _xywh_iou(bbox_xywh_scaled[lvl_idx][np.newaxis, :], anchors_xywh)
            ious.append(iou_scale)
            iou_mask = iou_scale > 0.3

            if np.any(iou_mask):
                indx, indy = np.floor(bbox_xywh_scaled[lvl_idx, 0:2]).astype(np.int32)
                indx = np.clip(indx, 0, featmap_sizes[lvl_idx] - 1)
                indy = np.clip(indy, 0, featmap_sizes[lvl_idx] - 1)

                bbox_coor = np.concatenate([bbox_xywh[:2] - bbox_xywh[2:] * 0.5,
                                            bbox_xywh[:2] + bbox_xywh[2:] * 0.5], axis=-1)
                targets[lvl_idx][iou_mask, 0:4, indy, indx] = bbox_coor
                targets[lvl_idx][iou_mask, 4:5, indy, indx] = 1.0
                targets[lvl_idx][iou_mask, 5:, indy, indx] = onehot_labels

                exist_positive = True

        #if not exist_positive:
        #    best_anchor_ind = np.argmax(np.array(ious).reshape(-1), axis=-1)
        #    best_level = int(best_anchor_ind / num_anchors)
        #    best_anchor = int(best_anchor_ind % num_anchors)
        #    indx, indy = np.floor(bbox_xywh_scaled[best_level, 0:2]).astype(np.int32)
        #    indx = np.clip(indx, 0, featmap_sizes[lvl_idx] - 1)
        #    indy = np.clip(indy, 0, featmap_sizes[lvl_idx] - 1)
        #    bbox_coor = np.concatenate([bbox_xywh[:2] - bbox_xywh[2:] * 0.5,
        #                                bbox_xywh[:2] + bbox_xywh[2:] * 0.5], axis=-1)
        #    targets[best_level][best_anchor, 0:4, indy, indx] = bbox_coor
        #    targets[best_level][best_anchor, 4:5, indy, indx] = 1.0
        #    targets[best_level][best_anchor, 5:, indy, indx] = onehot_labels

    return targets


def _meshgrid(y, x):
    H = y.shape[0]
    W = x.shape[0]
    xx = x.repeat(H).view(H, W)
    yy = y.view(-1, 1).repeat(1, W)

    return yy, xx


def yolov3_decoder(pred_map,
                   anchors,
                   num_anchors,
                   num_classes,
                   stride,
                   training=False):
    num_attrib = 5 + num_classes
    B, C, H, W = pred_map.shape
    device = pred_map.device
    pred_map = pred_map.view(B, num_anchors, num_attrib, H, W)

    dx = torch.sigmoid(pred_map[:, :, 0, ...])
    dy = torch.sigmoid(pred_map[:, :, 1, ...])
    dw = torch.exp(pred_map[:, :, 2, ...])
    dh = torch.exp(pred_map[:, :, 3, ...])
    conf = torch.sigmoid(pred_map[:, :, 4, ...])
    probs = torch.sigmoid(pred_map[:, :, 5:, ...])

    y_grid, x_grid = _meshgrid(torch.arange(H).to(device),
                               torch.arange(W).to(device))
    xy_grid = torch.stack([y_grid, x_grid], dim=2)
    xy_grid = xy_grid.view(1, 1, H, W, 2)
    cx = (xy_grid[..., 1].float() + dx) * stride
    cy = (xy_grid[..., 0].float() + dy) * stride

    anchors = torch.Tensor(anchors).view(1, num_anchors, 1, 1, 2).to(device)
    w = dw * anchors[..., 0]
    h = dh * anchors[..., 1]

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = x1 + w - 1
    y2 = y1 + h - 1

    if training:
        coor_and_conf = torch.stack([x1, y1, x2, y2, conf], dim=2)
        output = torch.cat([coor_and_conf, probs], dim=2)
        return output

    max_prob, max_index = torch.max(probs, dim=2)
    class_id = (max_index + 1).float()
    conf = max_prob * conf

    output = torch.stack([x1, y1, x2, y2, class_id, conf], dim=4)
    output = output.view(B, -1, 6)

    return output


def _get_boxes_area(boxes):
    return (boxes[:, :, 2, :, :] - boxes[:, :, 0, :, :]) * \
           (boxes[:, :, 3, :, :] - boxes[:, :, 1, :, :])


def _boxes_iou_1v1(boxes1, boxes2, epsilon=1e-8):
    device = boxes1.device
    b1_x1 = boxes1[:, :, 0, ...]
    b1_y1 = boxes1[:, :, 1, ...]
    b1_x2 = boxes1[:, :, 2, ...]
    b1_y2 = boxes1[:, :, 3, ...]
    b2_x1 = boxes2[:, :, 0, ...]
    b2_y1 = boxes2[:, :, 1, ...]
    b2_x2 = boxes2[:, :, 2, ...]
    b2_y2 = boxes2[:, :, 3, ...]

    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)

    intersection = torch.max(x2 - x1 + 1.0, torch.Tensor([0.0]).to(device)) * \
                   torch.max(y2 - y1 + 1.0, torch.Tensor([0.0]).to(device))
    boxes1_area = (b1_x2 - b1_x1 + 1.0) * (b1_y2 - b1_y1 + 1.0)
    boxes2_area = (b2_x2 - b2_x1 + 1.0) * (b2_y2 - b2_y1 + 1.0)
    union = boxes1_area + boxes2_area - intersection
    ious = intersection / (union + epsilon)
    return ious


def _boxes_giou_1v1(boxes1, boxes2, epsilon=1e-8):
    device = boxes1.device
    b1_x1 = boxes1[:, :, 0, ...]
    b1_y1 = boxes1[:, :, 1, ...]
    b1_x2 = boxes1[:, :, 2, ...]
    b1_y2 = boxes1[:, :, 3, ...]
    b2_x1 = boxes2[:, :, 0, ...]
    b2_y1 = boxes2[:, :, 1, ...]
    b2_x2 = boxes2[:, :, 2, ...]
    b2_y2 = boxes2[:, :, 3, ...]

    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)

    intersection = torch.max(x2 - x1 + 1.0, torch.Tensor([0.0]).to(device)) * \
                   torch.max(y2 - y1 + 1.0, torch.Tensor([0.0]).to(device))
    boxes1_area = (b1_x2 - b1_x1 + 1.0) * (b1_y2 - b1_y1 + 1.0)
    boxes2_area = (b2_x2 - b2_x1 + 1.0) * (b2_y2 - b2_y1 + 1.0)
    union = boxes1_area + boxes2_area - intersection
    ious = intersection / (union + epsilon)

    enclose_left_up = torch.min(boxes1[:, :, :2, ...], boxes2[:, :, :2, ...])
    enclose_right_down = torch.max(boxes1[:, :, 2:, ...], boxes2[:, :, 2:, ...])
    enclose = torch.max(enclose_right_down - enclose_left_up + 1.0, torch.Tensor([0.0]).to(device))
    enclose_area = enclose[:, :, 0, ...] * enclose[:, :, 1, ...]
    gious = ious - 1.0 * (enclose_area - union) / (enclose_area + epsilon)
    return gious


def _hard_example_weights(y_pred, y_true, alpha=1., gamma=2.):
    weights = alpha * torch.pow(torch.abs(y_pred - y_true), gamma)
    return weights


def yolov3_losses(pred_map,
                  target,
                  num_classes,
                  anchors,
                  stride,
                  gt_bboxes,
                  pos_iou_thr=0.5):
    B, C, H, W = pred_map.shape
    im_area = H * W * stride * stride
    num_attrib = 5 + num_classes
    num_anchors = anchors.shape[0]
    pred_bbox = yolov3_decoder(pred_map=pred_map,
                               anchors=anchors,
                               num_anchors=num_anchors,
                               num_classes=num_classes,
                               stride=stride,
                               training=True)
    pred_bbox_coor = pred_bbox[:, :, 0:4, :, :]
    pred_bbox_conf = pred_bbox[:, :, 4, :, :]
    pred_bbox_prob = pred_bbox[:, :, 5:, :, :]
    pred_map = pred_map.view(B, num_anchors, -1, H, W)
    pred_map_conf = pred_map[:, :, 4, :, :]
    pred_map_prob = pred_map[:, :, 5:, :, :]

    target = torch.Tensor(target).to(pred_map.device)
    target_coor = target[:, :, 0:4, :, :]
    target_conf  = target[:, :, 4, :, :]
    target_label = target[:, :, 5:, :, :]

    gious = _boxes_giou_1v1(pred_bbox_coor, target_coor)
    scale_weights = 2.0 - 1.0 * _get_boxes_area(target_coor) / im_area
    iou_loss = target_conf * scale_weights * (1- gious)

    loss_weights = _hard_example_weights(pred_bbox_conf, target_conf)
    gt_bboxes = torch.Tensor(np.transpose(gt_bboxes, (0, 2, 1))).to(pred_bbox_coor.device)
    ious = _boxes_iou_1v1(pred_bbox_coor[:, :, :, :, :, np.newaxis],
                          gt_bboxes[:, np.newaxis, :, np.newaxis, np.newaxis, :])
    max_iou, _ = torch.max(ious, dim=-1)
    is_background = (1.0 - target_conf) * torch.lt(max_iou,
                                                   torch.Tensor(
                                                       [pos_iou_thr]
                                                   ).to(max_iou.device)).float()
    sigmoid_ce_loss = F.binary_cross_entropy_with_logits(
        pred_map_conf, target_conf, reduction='none')
    conf_loss = loss_weights * (target_conf * sigmoid_ce_loss + is_background * sigmoid_ce_loss)

    uniform_distribution = np.full(num_classes, 1.0 / num_classes)
    deta = 0.01
    pos_smooth = 1 - deta
    neg_smooth = deta / num_classes
    target_prob = target_label * pos_smooth + (1 - target_label) * neg_smooth

    cls_loss = F.binary_cross_entropy_with_logits(
        pred_map_prob, target_prob, reduction='none') * \
                       target_conf[:, :, np.newaxis, :, :]

    iou_loss = torch.mean(torch.sum(iou_loss, dim=(1, 2, 3)))
    conf_loss = torch.mean(torch.sum(conf_loss, dim=(1, 2, 3)))
    cls_loss = torch.mean(torch.sum(cls_loss, dim=(1, 2, 3, 4)))

    #return (iou_loss, conf_loss, cls_loss), pred_bbox
    return iou_loss, conf_loss, cls_loss
