import torch.nn as nn
import torch.nn.functional as F
from ..builder import MODULES


@MODULES.register_module()
class HingeLoss(nn.Module):

    def __init__(self, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(HingeLoss, self).__init__()
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss = nn.ReLU(inplace=True)

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """

        if is_disc:
            # 如果是鉴别器
            input = -input if target_is_real else input
            loss = self.loss(1 + input).mean()
        else:
            # 如果是生成器 损失函数为 -y
            loss = -input.mean()

        return loss if is_disc else loss * self.loss_weight
