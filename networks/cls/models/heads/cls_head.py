from ..builder import HEADS, build_loss
from .base_head import BaseHead


@HEADS.register_module()
class ClsHead(BaseHead):

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 **kwargs):
        super(ClsHead, self).__init__()
        assert isinstance(loss, dict)

        self.compute_loss = build_loss(loss)

    def losses(self, inputs, gt_labels):
        num_samples = len(inputs)
        losses = dict()
        loss = self.compute_loss(inputs, gt_labels, avg_factor=num_samples)
        losses['losses'] = loss
        return losses

    def forward(self, inputs):
        return inputs
