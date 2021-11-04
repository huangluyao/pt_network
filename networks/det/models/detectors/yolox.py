# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/7/31 下午1:45
from networks.det.models.builder import DETECTORS
from .single_stage import SingleStageDetector
from .pruned_single_stage import PrunedSingleStageDetector


@DETECTORS.register_module()
class YOLOX(SingleStageDetector):
    """
    Implementation of 'FCOS' <https://arxiv.org/abs/1904.01355>
    """

    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None, pretrained=None):
        super(YOLOX, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)

@DETECTORS.register_module()
class PrunedYOLOX(PrunedSingleStageDetector):

    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None, pretrained=None):
        super(PrunedYOLOX, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)