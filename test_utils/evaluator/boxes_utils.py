# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/21 上午11:01
import numpy as np


def compute_boxes_iou(boxes1, boxes2, epsilon=1e-8):
    b1 = np.tile(boxes1, [1, boxes2.shape[0]])
    b1 = b1.reshape([-1, 4])
    b2 = np.tile(boxes2, [boxes1.shape[0], 1])
    x1 = np.maximum(b1[:, 0], b2[:, 0])
    y1 = np.maximum(b1[:, 1], b2[:, 1])
    x2 = np.minimum(b1[:, 2], b2[:, 2])
    y2 = np.minimum(b1[:, 3], b2[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    b1_area = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    b2_area = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = b1_area + b2_area - intersection
    ious = intersection / (union + epsilon)
    ious = ious.reshape([boxes1.shape[0], boxes2.shape[0]])

    return ious
