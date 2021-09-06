# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/8/10 下午3:39
import torch
import torch.nn as nn
from ..builder import LOSSES


@LOSSES.register_module()
class OhemCELoss(nn.Module):
    def __init__(self, thresh, stride=16, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.stride = stride
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels.long()).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        n_min = len(loss) // self.stride
        if loss[n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:n_min]
        return torch.mean(loss)
