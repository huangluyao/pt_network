import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import PIXEL_SAMPLERS
from .base_pixel_sampler import BasePixelSampler


@PIXEL_SAMPLERS.register_module()
class OHEMPixelSampler(BasePixelSampler):
    """Online Hard Example Mining Sampler for segmentation.

    Parameters
    ----------
    context : nn.Module
        The context of sampler, subclass of
        :obj:`BaseDecodeHead`.
    thresh : float, optional
        The threshold for hard example selection.
        Below which, are prediction with low confidence. If not
        specified, the hard examples will be pixels of top ``min_kept``
        loss. Default: None.
    min_kept : int, optional
        The minimum number of predictions to keep.
        Default: 100000.
    """

    def __init__(self, context, thresh=None, input_radio=0.5):
        super(OHEMPixelSampler, self).__init__()
        self.context = context
        self.thresh = thresh
        self.input_radio = input_radio

    def sample(self, seg_logit, seg_label):
        """Sample pixels that have high loss or with low prediction confidence.

        Parameters
        ----------
        seg_logit : torch.Tensor
            segmentation logits, shape (N, C, H, W)
        seg_label : torch.Tensor
            segmentation label, shape (N, 1, H, W)

        Returns
        -------
        torch.Tensor
            segmentation weight, shape (N, H, W)
        """

        with torch.no_grad():
            batch_kept = int(self.input_radio * np.prod(seg_logit.shape))
            if self.context.ignore_label is None:
                valid_mask = torch.ones_like(seg_label, dtype=torch.bool)
            else:
                valid_mask = seg_label != self.context.ignore_label
            seg_weight = seg_logit.new_zeros(size=seg_label.size())
            valid_seg_weight = seg_weight[valid_mask]
            if self.thresh is not None:
                seg_prob = F.softmax(seg_logit, dim=1)

                tmp_seg_label = seg_label.clone().unsqueeze(1)
                tmp_seg_label[tmp_seg_label == self.context.ignore_label] = 0
                seg_prob = seg_prob.gather(1, tmp_seg_label).squeeze(1)
                sort_prob, sort_indices = seg_prob[valid_mask].sort()

                if sort_prob.numel() > 0:
                    min_threshold = sort_prob[min(batch_kept,
                                                  sort_prob.numel() - 1)]
                else:
                    min_threshold = 0.0
                threshold = max(min_threshold, self.thresh)
                valid_seg_weight[seg_prob[valid_mask] < threshold] = 1.
            else:
                if not isinstance(self.context.loss_decode, nn.ModuleList):
                    losses_decode = [self.context.loss_decode]
                else:
                    losses_decode = self.context.loss_decode
                losses = 0.0
                for loss_module in losses_decode:
                    if loss_module._get_name() == "CrossEntropyLoss":
                        losses += loss_module(
                            seg_logit,
                            seg_label,
                            weight=None,
                            ignore_index=self.context.ignore_label,
                            reduction_override='none')

                _, sort_indices = losses[valid_mask].sort(descending=True)
                valid_seg_weight[sort_indices[:batch_kept]] = 1.

            seg_weight[valid_mask] = valid_seg_weight

            return seg_weight
