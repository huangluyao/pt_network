import torch
import torch.nn as nn

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base_classifier import BaseClassifier


@CLASSIFIERS.register_module()
class ImageClassifier(BaseClassifier):

    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(ImageClassifier, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(ImageClassifier, self).init_weights(pretrained)
        try:
            self.backbone.init_weights(pretrained=pretrained)
        except:
            print("backbone init weights failed")
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, inputs, ground_truth):
        x = self.extract_feat(inputs)

        losses = dict()
        gt_labels = torch.from_numpy(ground_truth['gt_labels']).to(inputs.device)
        loss = self.head.forward_train(x, gt_labels)
        losses.update(loss)
        return losses

    def forward_infer(self, inputs):
        x = self.extract_feat(inputs)
        logits = self.head(x)
        probs = torch.softmax(logits, dim=1)
        return probs
