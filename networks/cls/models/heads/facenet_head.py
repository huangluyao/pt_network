# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/15 下午2:14
import torch.nn as nn
import torch.nn.functional as F
from ..criterions import TripletLoss
from .base_head import BaseHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class FaceNetHead(BaseHead):

    def __init__(self, in_channel, embedding_size, loss, alpha=0.5, num_classes=2, **kwargs):
        super(FaceNetHead, self).__init__()
        self.embedding = nn.Linear(in_channel, embedding_size)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        self.classifier = nn.Linear(embedding_size, num_classes)

        self.compute_loss = build_loss(loss)
        self.triple_loss = TripletLoss(alpha=alpha)

        pass

    def forward(self, inputs):
        x = self.embedding(inputs)
        before_normalize = self.last_bn(x)
        # classification
        output1 = self.classifier(before_normalize)
        # feature
        output2 = F.normalize(before_normalize, p=2, dim=1)
        return output1, output2

    def forward_infer(self, inputs, **kwargs):
        x = self.embedding(inputs)
        before_normalize = self.last_bn(x)
        output = F.normalize(before_normalize, p=2, dim=1)
        return output

    def forward_train(self, inputs, gt_labels, **kwargs):

        x = self.embedding(inputs)
        before_normalize = self.last_bn(x)

        output1 = self.classifier(before_normalize)
        output2 = F.normalize(before_normalize, p=2, dim=1)

        losses = dict()
        losses['_triplet_loss'] = self.triple_loss(output2, x.shape[0]//3)
        losses['_ce_loss'] = self.compute_loss(output1, gt_labels, avg_factor=x.shape[0])
        losses['losses'] = losses['_triplet_loss'] + losses['_ce_loss']

        return losses


