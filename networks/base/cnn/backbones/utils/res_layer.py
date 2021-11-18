from collections import OrderedDict
from torch import nn as nn

from ...components import (build_activation_layer, build_conv_layer,
                           build_norm_layer, build_plugin_layer,
                           PLUGIN_LAYERS)


class BasicBlock(nn.Module):
    """BasicBlock for ResNet.

    Parameters
    ----------
    in_channels : int
        Input channels of this block.
    out_channels : int
        Output channels of this block.
    stride : int
        stride of the block. Default: 1
    dilation : int
        dilation of convolution. Default: 1
    downsample : nn.Module
        downsample operation on identity branch.
        Default: None.
    style : str
        `pytorch` or `caffe`. It is unused and reserved for
        unified API with Bottleneck.
    with_cp : bool
        Use checkpoint or not. Using checkpoint will save some
        memory while slowing down the training speed.
    conv_cfg : dict
        dictionary to construct and config conv layer.
        Default: None
    norm_cfg : dict
        dictionary to construct and config norm layer.
        Default: dict(type='BN')
    """

    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 mid_channels=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 dcn=None,
                 plugins=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        if mid_channels is None:
            mid_channels = out_channels

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        if isinstance(stride, (tuple, list)):
            stride = max(stride)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, mid_channels, out_channels, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = build_activation_layer(cfg=act_cfg)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@PLUGIN_LAYERS.register_module()
class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    If style is "pytorch", the stride-two layer is the 3x3 conv layer, if it is
    "caffe", the stride-two layer is the first 1x1 conv layer.
    """

    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 mid_channels=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 dcn=None,
                 plugins=None):
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        assert out_channels % self.expansion == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_channels is None:
            mid_channels = out_channels // self.expansion
        self.mid_channels = mid_channels
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if isinstance(stride, (tuple, list)):
            self.conv1_stride = stride[0]
            self.conv2_stride = stride[1]
        else:
            self.conv1_stride = 1
            self.conv2_stride = stride

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = build_activation_layer(cfg=act_cfg)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                mid_channels, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                mid_channels, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                out_channels, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Parameters
        ----------
        in_channels : int
            Input channels of plugin.
        plugins : list[dict]
            List of plugins cfg to build.

        Returns
        -------
        list[str]
            List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        """Forward function for plugins."""
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class BasicBlockV2(nn.Module):
    """BasicBlock for ResNetV2.
    """

    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 mid_channels=None,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(BasicBlockV2, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, in_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, mid_channels, postfix=2)

        if isinstance(stride, (tuple, list)):
            stride = max(stride)

        self.add_module(self.norm1_name, norm1)
        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv2 = build_conv_layer(
            conv_cfg, mid_channels, out_channels, 3, padding=1, bias=False)

        self.relu = build_activation_layer(cfg=act_cfg)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.norm1(x)
            out = self.relu(out)
            out = self.conv1(out)

            out = self.norm2(out)
            out = self.relu(out)
            out = self.conv2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)


class BottleneckV2(nn.Module):
    """Bottleneck block for ResNetV2.
    """

    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 mid_channels=None,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(_Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if mid_channels is None:
            mid_channels = out_channels // self.expansion
        self.mid_channels = mid_channels

        if isinstance(stride, (tuple, list)):
            self.conv1_stride = stride[0]
            self.conv2_stride = stride[1]
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, in_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, mid_channels, postfix=3)

        self.add_module(self.norm1_name, norm1)
        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv2 = build_conv_layer(
            conv_cfg,
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm3_name, norm3)
        self.conv3 = build_conv_layer(
            conv_cfg,
            mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)

        self.relu = build_activation_layer(cfg=act_cfg)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.norm1(x)
            out = self.relu(out)
            out = self.conv1(out)

            out = self.norm2(out)
            out = self.relu(out)
            out = self.conv2(out)

            out = self.norm3(out)
            out = self.relu(out)
            out = self.conv3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Parameters
    ----------
    block : nn.Module
        block used to build ResLayer.
    num_blocks : int
        number of blocks.
    in_channels : int
        in_channels of block.
    out_channels : int
        out_channels of block.
    stride : int
        stride of the first block. Default: 1
    avg_down : bool
        Use AvgPool instead of stride conv when
        downsampling in the bottleneck. Default: False
    conv_cfg : dict
        dictionary to construct and config conv layer.
        Default: None
    norm_cfg : dict
        dictionary to construct and config norm layer.
        Default: dict(type='BN')
    multi_grid : int | None
        Multi grid dilation rates of last
        stage. Default: None
    contract_dilation : bool
        Whether contract first dilation of each layer
        Default: False
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 downsample3x3=False,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 multi_grid=None,
                 contract_dilation=False,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False))
            downsample.append(build_conv_layer(
                conv_cfg,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=conv_stride,
                bias=False))
            norm_name, norm_module = build_norm_layer(norm_cfg, out_channels, postfix='1')
            downsample.append(norm_module)
            downsample = nn.Sequential(*downsample)

        layers = []
        if multi_grid is None:
            if dilation > 1 and contract_dilation:
                first_dilation = dilation // 2
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]

        if stride > 1 and downsample3x3:
            stride = (1, stride, 1)

        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dilation=first_dilation,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    dilation=dilation if multi_grid is None else multi_grid[i],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)
