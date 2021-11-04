import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .cross_entropy_loss import _expand_onehot_labels

@LOSSES.register_module()
class StructureLoss(nn.Module):

    @staticmethod
    def forward(pred, label, weight=None, **kwargs):
        mask = F.one_hot(torch.clamp_min(label.long(), 0)).float().permute(0, 3, 1, 2)
        weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
        wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
        pred  = torch.sigmoid(pred)
        inter = ((pred*mask)*weit).sum(dim=(2,3))
        union = ((pred+mask)*weit).sum(dim=(2,3))
        wiou  = 1-(inter+1)/(union-inter+1)
        return (wbce+wiou).mean()
