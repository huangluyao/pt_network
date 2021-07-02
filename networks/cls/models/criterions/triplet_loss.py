# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/15 下午3:50
import torch
import numpy as np
import torch.nn as nn


class TripletLoss(nn.Module):

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        super(TripletLoss, self).__init__()

    def forward(self, y_pred, batch_size):
        anchor, positive, negative = y_pred[:int(batch_size)], \
                                     y_pred[int(batch_size):int(2*batch_size)], \
                                     y_pred[int(2*batch_size):]

        pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive,2), axis=-1))
        neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative,2), axis=-1))
        keep_all = (neg_dist - pos_dist < self.alpha).cpu().numpy().flatten()
        hard_triplets = np.where(keep_all == 1)

        pos_dist = pos_dist[hard_triplets].cuda()
        neg_dist = neg_dist[hard_triplets].cuda()

        basic_loss = pos_dist - neg_dist + self.alpha
        loss = torch.sum(basic_loss)/torch.max(torch.tensor(1), torch.tensor(len(hard_triplets[0])))
        return loss

