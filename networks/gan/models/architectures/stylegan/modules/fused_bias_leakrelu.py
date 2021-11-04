import torch
import torch.nn as nn
import torch.nn.functional as F

def fused_bias_leakyrelu(x, bias, negative_slope=0.2, scale=2**0.5):
    """Fused bias leaky ReLU function.

    This function is introduced in the StyleGAN2:
    http://arxiv.org/abs/1912.04958

    The bias term comes from the convolution operation. In addition, to keep
    the variance of the feature map or gradients unchanged, they also adopt a
    scale similarly with Kaiming initialization. However, since the
    :math:`1 + \alpha^2` : is too small, we can just ignore it. Therefore, the
    final scale is just :math:`\sqrt{2}`:. Of course, you may change it with # noqa: W605, E501
    your own scale.

    Args:
        input (torch.Tensor): Input feature map.
        bias (nn.Parameter): The bias from convolution operation.
        negative_slope (float, optional): Same as nn.LeakyRelu.
            Defaults to 0.2.
        scale (float, optional): A scalar to adjust the variance of the feature
            map. Defaults to 2**0.5.

    Returns:
        torch.Tensor: Feature map after non-linear activation.
    """

    if bias is not None:
        assert bias.ndim == 1
        assert bias.shape[0] == x.shape[1]
        x = x + bias.reshape([-1 if i == 1 else 1 for i in range(x.ndim)])

    x = F.leaky_relu(x, negative_slope)
    if scale != 1:
        x = x * scale

    return x


class FusedBiasLeakyReLU(nn.Module):

    def __init__(self,num_channels, negative_slope=0.2, scale=2**0.5):
        super(FusedBiasLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.scale =scale
        self.bias = nn.Parameter(torch.zeros(num_channels))


    def forward(self, x):
        return fused_bias_leakyrelu(x,self.bias, self.negative_slope, self.scale)