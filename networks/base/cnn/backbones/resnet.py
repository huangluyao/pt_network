from collections import OrderedDict
import torch.nn as nn
import torch.utils.checkpoint as cp

from .builder import BACKBONES
from ..components import (build_activation_layer, build_conv_layer,
                          build_norm_layer, ConvModule, _BatchNorm)
from ...utils import load_checkpoint
from .utils import BasicBlock, Bottleneck, ResLayer
from ..utils import constant_init, kaiming_init
from ...utils import get_logger


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Parameters
    ----------
    block : class
        The block class.
    expansion : int | None
        The given expansion ratio.

    Returns
    -------
    int
        The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


@BACKBONES.register_module()
class ResNet(nn.Module):
    """ResNet backbone.

    Parameters
    ----------
    depth : int
        Network depth, from {18, 34, 50, 101, 152}.
    in_channels : int
        Number of input image channels. Default: 3.
    stem_width : int
        Output channels of the first and second stem layers. Default: 32.
    base_channels : int
        Number of base channels of res layer. Default: 64.
    num_stages : int
        Stages of the network. Default: 4.
    strides : Sequence[int]
        Strides of the first block of each stage.
        Default: ``(1, 2, 2, 2)``.
    dilations : Sequence[int]
        Dilation of each stage.
        Default: ``(1, 1, 1, 1)``.
    out_levels : Sequence[int]
        Output from which levels.
        If only one level is specified, a single tensor (feature map) is returned,
        otherwise multiple levels are specified, a tuple of tensors will be returned.
        Default: ``(5, )``, means return the 1/32x feature map.
    style : str
        `pytorch` or `caffe`. If set to "pytorch", the stride-two
        layer is the 3x3 conv layer, otherwise the stride-two layer is
        the first 1x1 conv layer.
    deep_stem : bool
        Replace 7x7 conv in input stem with 3 3x3 conv.
        Default: False.
    avg_down : bool
        Use AvgPool instead of stride conv when
        downsampling in the bottleneck. Default: False.
    frozen_stages : int
        Stages to be frozen (stop grad and set eval mode).
        -1 means not freezing any parameters. Default: -1.
    conv_cfg : dict | None
        The config dict for conv layers. Default: None.
    norm_cfg : dict
        The config dict for norm layers.
    norm_eval : bool
        Whether to set norm layers to eval mode, namely,
        freeze running stats (mean and var).
        Note: Effect on Batch Norm and its variants only. Default: False.
    plugins : list[dict]
        List of plugins for stages, each dict contains:

        - cfg (dict, required): Cfg dict to build plugin.

        - position (str, required): Position inside block to insert plugin,
        options: 'after_conv1', 'after_conv2', 'after_conv3'.

        - stages (tuple[bool], optional): Stages to apply plugin, length
        should be same as 'num_stages'
    multi_grid : Sequence[int]|None
        Multi grid dilation rates of last
        stage. Default: None
    contract_dilation : bool
        Whether contract first dilation of each layer
        Default: False
    with_cp : bool
        Use checkpoint or not. Using checkpoint will save some
        memory while slowing down the training speed.
    zero_init_residual : bool
        Whether to use zero init for last norm layer
        in resblocks to let them behave as identity.

    Examples
    --------
    >>> from base.cnn import ResNet
    >>> import torch
    >>> self = ResNet(depth=18)
    >>> self.eval()
    >>> inputs = torch.rand(1, 3, 32, 32)
    >>> level_outputs = self.forward(inputs)
    >>> for level_out in level_outputs:
    ...     print(tuple(level_out.shape))
    (1, 64, 8, 8)
    (1, 128, 4, 4)
    (1, 256, 2, 2)
    (1, 512, 1, 1)
    """

    stage_channels = {
        18:  (64, 128, 256, 512),
        34:  (64, 128, 256, 512),
        50:  (256, 512, 1024, 2048),
        101: (256, 512, 1024, 2048),
        152: (256, 512, 1024, 2048)
    }

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 downsample3x3=False,
                 deep_stem=False,
                 avg_down=False,
                 stem_width=32,
                 in_channels=3,
                 num_classes=None,
                 stage_channels=None,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_levels=(1, 2, 3, 4, 5),
                 expansion=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 style='pytorch',
                 num_stages=4,
                 frozen_stages=-1,
                 norm_eval=False,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 multi_grid=None,
                 contract_dilation=False,
                 with_cp=False,
                 zero_init_residual=True,
                 **kwargs):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_width = stem_width
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_levels = out_levels
        assert max(out_levels) < 6
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.multi_grid = multi_grid
        self.contract_dilation = contract_dilation
        self.zero_init_residual = zero_init_residual
        self.num_classes = num_classes
        if stage_channels is not None:
            self.stage_channels = stage_channels
        else:
            self.stage_channels = self.stage_channels[depth]
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels, stem_width)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            if i == 0:
                _in_channels = stem_width * 2
            else:
                _in_channels = self.stage_channels[i-1]
            _out_channels = self.stage_channels[i]
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            stage_multi_grid = multi_grid if i == len(
                self.stage_blocks) - 1 else None

            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                stride=stride,
                dilation=dilation,
                style=self.style,
                downsample3x3=downsample3x3,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation)
            layer_name = f'layer{i+1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.stage_channels[-1], num_classes)

        self._freeze_stages()

    def make_stage_plugins(self, plugins, stage_idx):
        """make plugins for ResNet 'stage_idx'th stage .

        Currently we support to insert 'context_block',
        'empirical_attention_block', 'nonlocal_block' into the backbone like
        ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be :
        >>> plugins=[
        ...     dict(cfg=dict(type='xxx', arg1='xxx'),
        ...          stages=(False, True, True, True),
        ...          position='after_conv2'),
        ...     dict(cfg=dict(type='yyy'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='1'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='2'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3')
        ... ]
        >>> self = ResNet(depth=18)
        >>> stage_plugins = self.make_stage_plugins(plugins, 0)
        >>> assert len(stage_plugins) == 3

        Suppose 'stage_idx=0', the structure of blocks in the stage would be:
        conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:
        conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Parameters
        ----------
        plugins : list[dict]
            List of plugins cfg to build. The postfix is
            required if multiple same type plugins are inserted.
        stage_idx : int
            Index of stage to build

        Returns
        -------
        list[dict]
            Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_width=32):
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.conv1 = nn.Sequential(OrderedDict([
                ('0', build_conv_layer(
                    self.conv_cfg,
                    in_channels=in_channels,
                    out_channels=stem_width,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False
                )),
                ('1', build_norm_layer(self.norm_cfg, stem_width, postfix='1')[1]),
                ('2', build_activation_layer(self.act_cfg)),
                ('3', build_conv_layer(
                    self.conv_cfg,
                    in_channels=stem_width,
                    out_channels=stem_width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )),
                ('4', build_norm_layer(self.norm_cfg, stem_width, postfix='2')[1]),
                ('5', build_activation_layer(self.act_cfg)),
                ('6', build_conv_layer(
                    self.conv_cfg,
                    in_channels=stem_width,
                    out_channels=stem_width * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ))
            ]))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_width * 2,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_width * 2, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = build_activation_layer(cfg=self.act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Parameters
        ----------
        pretrained : str, optional
            Path to pre-trained weights.
            Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_logger('deepcv_base')
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        outs = []
        if 0 in self.out_levels:
            outs.append(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if 1 in self.out_levels:
            outs.append(x)
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i+2 in self.out_levels:
                outs.append(x)

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = x.contiguous().view(-1, self.stage_channels[-1])
            x = self.fc(x)
            return x
        else:
            if len(outs) == 1:
                return outs[0]
            else:
                return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_module()
class ResNetV1b(ResNet):
    """ResNetV1b variant described in
    `Bag of Tricks <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet, ResNetV1b switches the strides size of the
    first two convolutions in main path.
    """

    def __init__(self, **kwargs):
        super(ResNetV1b, self).__init__(
            downsample3x3=True, deep_stem=False, avg_down=False, **kwargs)


@BACKBONES.register_module()
class ResNetV1c(ResNet):
    """ResNetV1c variant described in
    `Bag of Tricks <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with ResNetV1b, ResNetV1c replaces the 7x7 conv
    in the input stem with three 3x3 convs.
    """

    def __init__(self, **kwargs):
        super(ResNetV1c, self).__init__(
            downsample3x3=True, deep_stem=True, avg_down=False, **kwargs)


@BACKBONES.register_module()
class ResNetV1d(ResNet):
    """ResNetV1d variant described in
    `Bag of Tricks <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with ResNetV1c, in the downsampling block of ResNetV1d, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            downsample3x3=True, deep_stem=True, avg_down=True, **kwargs)


@BACKBONES.register_module()
class ResNetV1e(ResNet):

    def __init__(self, **kwargs):
        super(ResNetV1e, self).__init__(
            downsample3x3=True, deep_stem=True, avg_down=True, stem_width=64, **kwargs)


@BACKBONES.register_module()
class ResNetV1s(ResNet):

    def __init__(self, **kwargs):
        super(ResNetV1s, self).__init__(
            downsample3x3=True, deep_stem=True, avg_down=False, stem_width=64, **kwargs)