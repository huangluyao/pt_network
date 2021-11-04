import torch
import torch.nn as nn
from ...builder import build_module, MODULES
from networks.base.cnn.components import ConvModule

@MODULES.register_module()
class MultiLayerDiscriminator(nn.Module):

    def __init__(self,
                 in_channels,
                 max_channels,
                 num_convs=5,
                 kernel_size=5,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 out_act_cfg=dict(type='ReLU'),
                 with_input_norm=True,
                 with_out_convs=False,
                 with_spectral_norm=False,
                 **kwargs
                 ):

        super(MultiLayerDiscriminator, self).__init__()

        self.max_channels = max_channels
        self.num_convs = num_convs
        self.with_out_act = out_act_cfg is not None
        self.with_out_convs = with_out_convs

        cur_channels = in_channels

        for i in range(num_convs):
            out_channel = min(64 * 2**i, max_channels)
            norm_cfg_ = norm_cfg
            act_cfg_ = act_cfg
            if i == 0 and not with_input_norm:
                norm_cfg_ = None
            if (i == num_convs - 1 and not self.with_out_convs):
                norm_cfg_ = None
                act_cfg_ = out_act_cfg

            self.add_module(
                f'conv{i+1}',
                ConvModule(
                    cur_channels,
                    out_channel,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size//2,
                    norm_cfg=norm_cfg_,
                    act_cfg=act_cfg_,
                    with_spectral_norm=with_spectral_norm,
                    **kwargs)
            )
            cur_channels = out_channel

    def forward(self, x):
        num_convs = self.num_convs

        for i in range(num_convs):
            x = getattr(self, f'conv{i + 1}')(x)

        return x


    def init_weights(self):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        for m in self.modules():
            # Here, we only initialize the module with fc layer since the
            # conv and norm layers has been intialized in `ConvModule`.
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
