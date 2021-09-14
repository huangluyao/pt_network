import torch
import torch.nn as nn

from .base import BaseDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck


class PrunedSingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None, **kwargs):
        super(PrunedSingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)

        is_pruned_model = "mask_bn_dict" in neck.keys()

        if is_pruned_model:
            input_ = torch.zeros((1, 3, 416, 416))
            out = self.backbone(input_)
            neck.input_channel_maps = self.backbone.out_level_channel_map
            self.former_to_map = dict(**self.backbone.former_to_map)

        if neck is not None:
            self.neck = build_neck(neck)

            if is_pruned_model:
                out = self.neck(out)
                bbox_head.in_channels = self.neck.output_channels
                self.former_to_map.update(**self.neck.former_to_map)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Parameters
        ----------
        pretrained : str, optional
            Path to pre-trained weights.
            Defaults to None.
        """
        super(PrunedSingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, inputs):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, inputs, ground_truth, **kwargs):
        """Forward function for training.

        Parameters
        ----------
        inputs : Tensor
            Input images.
        ground_truth : dict[str, list]
            Ground truth for detector.
            dict(
                gt_bboxes=list[Tensor],
                gt_labels=list[Tensor],
                gt_masks=list[Tensor]
            )
        Returns
        -------
        dict[str, Tensor]
            a dictionary of loss components
        """

        x = self.extract_feat(inputs)
        losses = self.bbox_head.forward_train(x, ground_truth, **kwargs)
        return losses

    def forward_infer(self, inputs, **kwargs):
        x = self.extract_feat(inputs)
        x = self.bbox_head(x)
        bboxes = self.bbox_head.get_bboxes(x)
        return bboxes
