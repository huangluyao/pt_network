import warnings

import torch.nn as nn

from ..utils import constant_init, kaiming_init
from .activation import build_activation_layer
from .conv import build_conv_layer
from .norm import build_norm_layer
from .padding import build_padding_layer
from .registry import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Parameters
    ----------
    dtype : str, optional
        Whether to store matrix in C- or Fortran-contiguous order,
        default is 'C'.
    in_channels : int
        Same as nn.Conv2d.
    out_channels : int
        Same as nn.Conv2d.
    kernel_size : {int, tuple[int]}
        Same as nn.Conv2d.
    stride : {int, tuple[int]}
        Same as nn.Conv2d.
    padding : {int, tuple[int]}
        Same as nn.Conv2d.
    dilation : {int, tuple[int]}
        Same as nn.Conv2d.
    groups : int
        Same as nn.Conv2d.
    bias : {bool, str}
        If specified as `auto`, it will be decided by the
        norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
        False. Default: "auto".
    conv_cfg : dict
        Config dict for convolution layer. Default: None,
        which means using conv2d.
    norm_cfg : dict
        Config dict for normalization layer. Default: None.
    act_cfg : dict
        Config dict for activation layer.
        Default: dict(type='ReLU').
    inplace : bool
        Whether to use inplace mode for activation.
        Default: True.
    with_spectral_norm : bool
        Whether use spectral norm in conv module.
        Default: False.
    padding_mode : str
        If the `padding_mode` has not been supported by
        current `Conv2d` in PyTorch, we will use our own padding layer
        instead. Currently, we support ['zeros', 'circular'] with official
        implementation and ['reflect'] with our own implementation.
        Default: 'zeros'.
    order : tuple[str]
        The order of conv/norm/activation layers. It is a
        sequence of "conv", "norm" and "act". Common examples are
        ("conv", "norm", "act") and ("act", "conv", "norm").
        Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act'),
                 shortcut=False,
                 **kwargs):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        conv_padding = 0 if self.with_explicit_padding else padding
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups
        self.shortcut = shortcut
        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        if self.with_norm:
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', "SiLU", "HardSwish"
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        if self.shortcut:
            temp = x
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)

        return x+temp if self.shortcut else x


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.

    See https://arxiv.org/pdf/1704.04861.pdf for details.

    This module can replace a ConvModule with the conv block replaced by two
    conv block: depthwise conv block and pointwise conv block. The depthwise
    conv block contains depthwise-conv/norm/activation layers. The pointwise
    conv block contains pointwise-conv/norm/activation layers. It should be
    noted that there will be norm/activation layer in the depthwise conv block
    if `norm_cfg` and `act_cfg` are specified.

    Parameters
    ----------
    in_channels : int
        Same as nn.Conv2d.
    out_channels : int
        Same as nn.Conv2d.
    kernel_size : int or tuple[int]
        Same as nn.Conv2d.
    stride : int or tuple[int]
        Same as nn.Conv2d. Default: 1.
    padding : int or tuple[int]
        Same as nn.Conv2d. Default: 0.
    dilation : int or tuple[int]
        Same as nn.Conv2d. Default: 1.
    norm_cfg : dict
        Default norm config for both depthwise ConvModule and
        pointwise ConvModule. Default: None.
    act_cfg : dict
        Default activation config for both depthwise ConvModule
        and pointwise ConvModule. Default: dict(type='ReLU').
    dw_norm_cfg : dict
        Norm config of depthwise ConvModule. If it is
        'default', it will be the same as `norm_cfg`. Default: 'default'.
    dw_act_cfg : dict
        Activation config of depthwise ConvModule. If it is
        'default', it will be the same as `act_cfg`. Default: 'default'.
    pw_norm_cfg : dict
        Norm config of pointwise ConvModule. If it is
        'default', it will be the same as `norm_cfg`. Default: 'default'.
    pw_act_cfg : dict
        Activation config of pointwise ConvModule. If it is
        'default', it will be the same as `act_cfg`. Default: 'default'.
    kwargs : optional
        Other shared arguments for depthwise and pointwise
        ConvModule. See ConvModule for ref.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 dw_norm_cfg='default',
                 dw_act_cfg='default',
                 pw_norm_cfg='default',
                 pw_act_cfg='default',
                 **kwargs):
        super(DepthwiseSeparableConvModule, self).__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'

        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg

        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,
            act_cfg=dw_act_cfg,
            **kwargs)

        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
            **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
