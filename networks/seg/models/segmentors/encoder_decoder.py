import torch
import torch.nn as nn
import torch.nn.functional as F

from base.cnn import resize
from .base import BaseSegmentor
from .. import builder
from ..builder import SEGMENTORS
from ...utils import add_prefix


@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 **kwargs):
        super(EncoderDecoder, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Parameters
        ----------
        pretrained : str, optional
            Path to pre-trained weights.
            Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, inputs):
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        out = self._decode_head_forward_infer(x)
        out = resize(
            input=out,
            size=inputs.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, ground_truth):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode, seg_logit = self.decode_head.forward_train(x, ground_truth)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses, seg_logit

    def _decode_head_forward_infer(self, x):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_infer(x)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, ground_truth):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux, _ = aux_head.forward_train(x, ground_truth)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux, _ = self.auxiliary_head.forward_train(
                x, ground_truth)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_train(self, inputs, ground_truth):
        """Forward function for training.

        Parameters
        ----------
        inputs : Tensor
            Input images.
        ground_truth : Tensor
            Semantic segmentation masks
            used if the architecture supports semantic segmentation task.

        Returns
        -------
        dict[str, Tensor]
            a dictionary of loss components
        """

        x = self.extract_feat(inputs)

        losses = dict()
        gt_masks = torch.from_numpy(ground_truth['gt_masks']).to(inputs.device, dtype=inputs.dtype)
        loss_decode, seg_logit = self._decode_head_forward_train(x, gt_masks)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux  = self._auxiliary_head_forward_train(
                x, gt_masks)
            losses.update(loss_aux)

        if self.num_classes > 1:
            seg_probs = torch.softmax(seg_logit, dim=1)
        else:
            seg_probs = torch.sigmoid(seg_logit)

        return dict(
                losses=losses,
                preds=seg_probs,
                seg_logit=seg_logit
                )

    def forward_infer(self, inputs, logits=True):
        seg_logit = self.encode_decode(inputs)
        if self.num_classes > 1:
            seg_probs = torch.softmax(seg_logit, dim=1)
        else:
            seg_probs = torch.sigmoid(seg_logit)
        if logits:
            if torch.onnx.is_in_onnx_export():
                return seg_probs
            else:
                return dict(
                        preds=seg_probs,
                        seg_logit=seg_logit
                        )
        else: 
            return dict(
                    preds=seg_probs
                    )


