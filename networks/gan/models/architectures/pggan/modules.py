import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Tuple
from torch.nn.init import _calculate_correct_fan
from networks.gan.models.builder import MODULES
from networks.base.cnn.utils import normal_init
from networks.base.cnn.components import (NORM_LAYERS, PLUGIN_LAYERS, CONV_LAYERS,
                                build_norm_layer, build_activation_layer,
                                build_upsample_layer, build_conv_layer, ConvModule)


def pixel_norm(x, eps=1e-6):
    """Pixel Normalization.

    This normalization is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Args:
        x (torch.Tensor): Tensor to be normalized.
        eps (float, optional): Epsilon to avoid dividing zero.
            Defaults to 1e-6.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    if torch.__version__ >= '1.7.0':
        norm = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)
    # support older pytorch version
    else:
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
    norm = norm / torch.sqrt(torch.tensor(x.shape[1]).to(x))

    return x / (norm + eps)


class PixelNorm(nn.Module):
    """Pixel Normalization.

    This module is proposed in:
    Progressive Growing of GANs for Improved Quality, Stability, and Variation

    Args:
        eps (float, optional): Epsilon value. Defaults to 1e-6.
    """

    _abbr_ = 'pn'

    def __init__(self, in_channels=None, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return pixel_norm(x, self.eps)


@CONV_LAYERS.register_module()
class EqualizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros", gain=np.sqrt(2)):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                         padding_mode, )
        # make sure that the self.weight and self.bias are initialized according to
        # random normal distribution
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # define the scale for the weights:
        fan_in = np.prod(self.kernel_size) * self.in_channels
        self.scale = gain / np.sqrt(fan_in)

    def forward(self, x):
        return torch.conv2d(input=x, weight=self.weight * self.scale, bias=self.bias, stride=self.stride,
                            padding=self.padding, dilation=self.dilation, groups=self.groups, )

@CONV_LAYERS.register_module()
class EqualizedConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode="zeros", ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias,
                         dilation, padding_mode, )
        # make sure that the self.weight and self.bias are initialized according to
        # random normal distribution
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # define the scale for the weights:
        fan_in = self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size
        )
        return torch.conv_transpose2d(input=x, weight=self.weight * self.scale, bias=self.bias, stride=self.stride,
                                      padding=self.padding, output_padding=output_padding, groups=self.groups,
                                      dilation=self.dilation, )


class EqualizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, gain=np.sqrt(2), lr_mul=1.0) -> None:
        super().__init__(in_features, out_features, bias)

        # make sure that the self.weight and self.bias are initialized according to
        # random normal distribution
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # define the scale for the weights:
        fan_in = self.in_features
        self.scale = (gain / np.sqrt(fan_in)) * lr_mul

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight * self.scale, self.bias)



class PGGANNoiseTo2DFeat(nn.Module):

    def __init__(self,
                 noise_size,
                 out_channels,
                 gain=np.sqrt(2),
                 conv_cfg=dict(type="EqualizedConv2d"),
                 act_cfg=dict(type="LeakyReLU", negative_slope=0.2),
                 norm_cfg=dict(type="PixelNorm"),
                 normalize_latent = True,
                 order = ('linear', 'act', 'norm')
    ):

        super(PGGANNoiseTo2DFeat, self).__init__()
        self.noise_size = noise_size
        self.out_channels = out_channels
        self.normalize_latent = normalize_latent
        self.order = order

        self.linear = EqualizedLinear(
            noise_size,
            out_channels * 16,
            gain=gain,
            bias=False)

        # add bias for reshaped 2D feature.
        self.register_parameter(
            'bias', nn.Parameter(torch.zeros(1, out_channels, 1, 1)))

        self.activation = build_activation_layer(act_cfg)
        _, self.norm = build_norm_layer(norm_cfg, out_channels)

        self.conv = ConvModule(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                               conv_cfg=conv_cfg,norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        if self.normalize_latent:
            x = pixel_norm(x)

        for order in self.order:
            if order=='linear':
                x = self.linear(x)
                x = torch.reshape(x, (-1, self.out_channels, 4, 4))
                x = x + self.bias
            elif order == 'act':
                x = self.activation(x)
            elif order == "norm":
                x = self.norm(x)

        return self.conv(x)


class GenGeneralConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, conv_cfg=None,
                 norm_cfg=None, act_cfg=None,
                 up_sample_cfg=dict(type='nearest', scale_factor=2),
                 **kwargs):
        super(GenGeneralConvBlock, self).__init__()
        self.conv_1 = ConvModule(in_channels, out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=act_cfg,
                                 **kwargs
                                 )

        self.conv_2 = ConvModule(out_channels, out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=act_cfg,
                                 **kwargs
                                 )

        self.upsample_layer = build_upsample_layer(up_sample_cfg)

    def forward(self, x):
        x = self.upsample_layer(x)
        x = self.conv_2(self.conv_1(x))

        return x


class DisGeneralConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, conv_cfg=None,
                 norm_cfg=None, act_cfg=None,
                 **kwargs):
        super(DisGeneralConvBlock, self).__init__()

        self.conv1 = ConvModule(in_channels, in_channels, kernel_size=3,
                                stride=1, padding=1, bias=True,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg,
                                **kwargs
                                )

        self.conv2 = ConvModule(in_channels, out_channels, kernel_size=3,
                                stride=2, padding=1, bias=True,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg,
                                **kwargs
                                )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class PGGANDecisionHead(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 bias=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 **kwargs
                 ):
        super(PGGANDecisionHead, self).__init__()

        self.conv_1 = ConvModule(
            in_channels,
            out_channels, (3, 3),
            stirde=1,
            padding=1, bias=bias,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.conv_2 = ConvModule(out_channels, out_channels, (4, 4), bias=bias,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=act_cfg,)
        self.conv_3 = ConvModule(out_channels, 1, (1, 1), bias=bias, conv_cfg=conv_cfg,
                                 norm_cfg=None, act_cfg=None)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))


class MinibatchStdDev(torch.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    Args:
        group_size: Size of each group into which the batch is split
    """

    def __init__(self, group_size: int = 4) -> None:
        """

        Args:
            group_size: Size of each group into which the batch is split
        """
        super(MinibatchStdDev, self).__init__()
        self.group_size = group_size

    def extra_repr(self) -> str:
        return "group_size={self.group_size}"

    def forward(self, x, alpha: float = 1e-8):
        """
        forward pass of the layer
        Args:
            x: input activation volume
            alpha: small number for numerical stability
        Returns: y => x appended with standard deviation constant map
        """
        batch_size, channels, height, width = x.shape
        if batch_size > self.group_size:
            assert batch_size % self.group_size == 0, (
                "batch_size {batch_size} should be "
                "perfectly divisible by group_size {self.group_size}"
            )
            group_size = self.group_size
        else:
            group_size = batch_size

        # reshape x into a more amenable sized tensor
        y = torch.reshape(x, [group_size, -1, channels, height, width])

        # indicated shapes are after performing the operation
        # [G x M x C x H x W] Subtract mean over groups
        y = y - y.mean(dim=0, keepdim=True)

        # [M x C x H x W] Calc standard deviation over the groups
        y = torch.sqrt((y * y).mean(dim=0, keepdim=False) + alpha)

        # [M x 1 x 1 x 1]  Take average over feature_maps and pixels.
        y = y.mean(dim=[1, 2, 3], keepdim=True)

        # [B x 1 x H x W]  Replicate over group and pixels
        y = y.repeat(group_size, 1, height, width)

        # [B x (C + 1) x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y

