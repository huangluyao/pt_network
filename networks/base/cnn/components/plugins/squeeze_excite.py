import torch.nn as nn

from ..conv_module import ConvModule
from ..registry import PLUGIN_LAYERS
from ....utils import is_seq_of


@PLUGIN_LAYERS.register_module()
class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.

    Parameters
    ----------
    channels : int
        The input (and output) channels of the SE layer.
    ratio : int
        Squeeze ratio in SELayer, the intermediate channel will be
        ``int(channels/ratio)``. Default: 16.
    conv_cfg : None or dict
        Config dict for convolution layer.
        Default: None, which means using conv2d.
    norm_cfg : dict or Sequence[dict]
        Similar to act_cfg.
    act_cfg : dict or Sequence[dict]
        Config dict for activation layer.
        If act_cfg is a dict, two activation layers will be configurated
        by this dict. If act_cfg is a sequence of dicts, the first
        activation layer will be configurated by the first dict and the
        second activation layer will be configurated by the second dict.
        Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 norm_cfg=(None, None),
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')),
                 rd_round_fn=None
                 ):
        super(SELayer, self).__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_seq_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        if rd_round_fn is None:
            rd_round_fn = round
        mid_channels = rd_round_fn(channels / ratio)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg[0],
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=mid_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg[1],
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out
