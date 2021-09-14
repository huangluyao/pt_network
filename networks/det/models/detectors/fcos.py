from .single_stage import SingleStageDetector
from .pruned_single_stage import PrunedSingleStageDetector
from networks.det.models.builder import DETECTORS

@DETECTORS.register_module()
class FCOS(SingleStageDetector):
    """
    Implementation of 'FCOS' <https://arxiv.org/abs/1904.01355>
    """

    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None, pretrained=None, **kwargs):
        super(FCOS, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, **kwargs)


@DETECTORS.register_module()
class PrunedFCOS(PrunedSingleStageDetector):

    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None, pretrained=None, **kwargs):
        super(PrunedFCOS, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, **kwargs)
pass