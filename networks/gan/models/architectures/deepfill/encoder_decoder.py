import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from networks.base.cnn.components import ConvModule, build_activation_layer
from .modules import SimpleGatedConvModule
from ...builder import MODULES, build_module

@MODULES.register_module()
class DeepFillEncoder(nn.Module):
    """Encoder used in DeepFill model.

    This implementation follows:
    Generative Image Inpainting with Contextual Attention

    Args:
        in_channels (int): The number of input channels. Default: 5.
        conv_type (str): The type of conv module. In DeepFillv1 model, the
            `conv_type` should be 'conv'. In DeepFillv2 model, the `conv_type`
            should be 'gated_conv'.
        norm_cfg (dict): Config dict to build norm layer. Default: None.
        act_cfg (dict): Config dict for activation layer, "elu" by default.
        encoder_type (str): Type of the encoder. Should be one of ['stage1',
            'stage2_conv', 'stage2_attention']. Default: 'stage1'.
        channel_factor (float): The scale factor for channel size.
            Default: 1.
        kwargs (keyword arguments).
    """
    _conv_type = dict(conv=ConvModule, gated_conv=SimpleGatedConvModule)

    def __init__(self, in_channels=5, conv_type='conv',
                 norm_cfg=None, act_cfg=dict(type='ELU'),
                 encoder_type='stage1',
                 channel_factor=1.,
                 **kwargs):
        super().__init__()
        conv_module = self._conv_type[conv_type]
        channel_list_dict = dict(
            stage1=[32, 64, 64, 128, 128, 128],
            stage2_conv=[32, 32, 64, 64, 128, 128],
            stage2_attention=[32, 32, 64, 128, 128, 128])
        channel_list = channel_list_dict[encoder_type]
        channel_list = [int(x * channel_factor) for x in channel_list]
        kernel_size_list = [5, 3, 3, 3, 3, 3]
        stride_list = [1, 2, 1, 2, 1, 1]

        for i in range(6):
            ks = kernel_size_list[i]
            padding = (ks - 1) // 2
            self.add_module(
                f'enc{i + 1}',
                conv_module(
                    in_channels,
                    channel_list[i],
                    kernel_size=ks,
                    stride=stride_list[i],
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs))
            in_channels = channel_list[i]


    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        for i in range(6):
            x = getattr(self, f'enc{i + 1}')(x)
        outputs = dict(out=x)
        return outputs


@MODULES.register_module()
class DeepFillDecoder(nn.Module):

    _conv_type = dict(conv=ConvModule, gated_conv=SimpleGatedConvModule)

    def __init__(self,
                 in_channels,
                 conv_type='conv',
                 norm_cfg=None,
                 act_cfg=dict(type='ELU'),
                 out_act_cfg=dict(type='clip', min=-1., max=1.),
                 channel_factor=1.,
                 **kwargs):
        super().__init__()
        self.with_out_activation = out_act_cfg is not None

        conv_module = self._conv_type[conv_type]

        channel_list = [128, 128, 64, 64, 32, 16, 3]
        channel_list = [int(x * channel_factor) for x in channel_list]

        channel_list[-1] = 3
        for i in range(7):
            kwargs_ = copy.deepcopy(kwargs)
            if i == 6:
                act_cfg = None
                if conv_type == "gated_conv":
                    kwargs_["feat_act_cfg"] = None
            self.add_module(
                f'dec{i + 1}',
                conv_module(
                    in_channels,
                    channel_list[i],
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs_
                )
            )
            in_channels = channel_list[i]

        if self.with_out_activation:
            act_type = out_act_cfg['type']
            if act_type == 'clip':
                act_cfg_ = copy.deepcopy(out_act_cfg)
                act_cfg_.pop('type')
                self.out_act = partial(torch.clamp, **act_cfg_)
            else:
                self.out_act = build_activation_layer(out_act_cfg)


    def forward(self, input_dict):
        """Forward Function.

        Args:
            input_dict (dict | torch.Tensor): Input dict with middle features
                or torch.Tensor.

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h, w).
        """
        if isinstance(input_dict, dict):
            x = input_dict['out']
        else:
            x = input_dict
        for i in range(7):
            x = getattr(self, f'dec{i + 1}')(x)
            if i in (1, 3):
                x = F.interpolate(x, scale_factor=2)

        if self.with_out_activation:
            x = self.out_act(x)
        return x


@MODULES.register_module()
class GLEncoderDecoder(nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 dilation_neck,
                 **kwargs
                 ):

        super(GLEncoderDecoder, self).__init__()
        self.encoder = build_module(encoder)
        self.decoder = build_module(decoder)
        self.dilation_neck = build_module(dilation_neck)

    def forward(self, x):
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        x = self.encoder(x)
        if isinstance(x, dict):
            x = x['out']
        x = self.dilation_neck(x)
        x = self.decoder(x)

        return x