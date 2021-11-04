import torch.nn as nn
import torch.nn.functional as F
from ..builder import MODULES


@MODULES.register_module()
class WGANLoss(nn.Module):

    def __init__(self, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(WGANLoss, self).__init__()
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

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
        loss = -input.mean() if target_is_real else input.mean()
        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

@MODULES.register_module()
class WGANLossNS(nn.Module):
    """WGAN loss in logistically non-saturating mode.

    This loss is widely used in StyleGANv2.

    Args:
        input (Tensor): Input tensor.
        target (bool): Target label.

    Returns:
        Tensor: wgan loss.
    """
    def __init__(self, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(WGANLossNS, self).__init__()
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val


    def forward(self,input, target_is_real, is_disc=False):
        loss = F.softplus(-input).mean() if target_is_real else F.softplus(input).mean()

        return loss if is_disc else loss * self.loss_weight