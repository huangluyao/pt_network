import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from networks.gan.models.architectures.pggan.modules import PGGANNoiseTo2DFeat, GenGeneralConvBlock,\
                                                            DisGeneralConvBlock, MinibatchStdDev, PGGANDecisionHead
from networks.base.cnn.components import ConvModule, build_upsample_layer
from networks.gan.models.builder import MODULES


@MODULES.register_module()
class PGGANGenerator(nn.Module):
    """Generator for PGGAN.

    Args:
        noise_size (int): Size of the input noise vector.
        out_scale (int): Output scale for the generated image.
        label_size (int, optional): Size of the label vector.
            Defaults to 0.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this
            number. Defaults to 8192.
        channel_decay (float, optional): Decay for channels of feature maps.
            Defaults to 1.0.
        max_channels (int, optional): Maximum channels for the feature
            maps in the generator block. Defaults to 512.
        fused_upconv (bool, optional): Whether use fused upconv.
            Defaults to True.
        conv_module_cfg (dict, optional): Config for the convolution
            module used in this generator. Defaults to None.
        fused_upconv_cfg (dict, optional): Config for the fused upconv
            module used in this generator. Defaults to None.
        upsample_cfg (dict, optional): Config for the upsampling operation.
            Defaults to None.
    """

    def __init__(self,
                 noise_size,
                 out_scale,
                 label_size=0,
                 base_channels=8192,
                 channel_decay=1.,
                 max_channels=512,
                 fused_up_conv=True,
                 conv_cfg=dict(type="EqualizedConv2d"),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 norm_cfg=dict(type='PixelNorm'),
                 fused_up_conv_cfg=None,
                 up_sample_cfg=dict(type='nearest', scale_factor=2)
                 ):
        super(PGGANGenerator, self).__init__()
        self.noise_size = noise_size if noise_size else min(base_channels, max_channels)
        self.out_log2_scale = int(np.log2(out_scale))
        self.label_size = label_size
        self.base_channels = base_channels
        self.channel_decay = channel_decay
        self.max_channels = max_channels
        self.fused_up_conv = fused_up_conv
        self.up_sample_cfg =up_sample_cfg

        self.noise2feat = PGGANNoiseTo2DFeat(noise_size,
                                             self._stage_to_out_channel(1),
                                             conv_cfg=conv_cfg,
                                             norm_cfg=norm_cfg,
                                             act_cfg=act_cfg
                                             )

        self.to_rgb_layers = nn.ModuleList(
            [
                ConvModule(self._stage_to_out_channel(stage-1),
                           3,
                           kernel_size=1,
                           stride=1,
                           bias=True,
                           conv_cfg=conv_cfg,
                           norm_cfg=None,
                           act_cfg=None
                           )
                for stage in range(2, self.out_log2_scale + 1)
            ]
        )

        self.conv_blocks = nn.ModuleList(
            [
                GenGeneralConvBlock(self._stage_to_out_channel(stage),
                                    self._stage_to_out_channel(stage+1),
                                    conv_cfg=conv_cfg,
                                    act_cfg=act_cfg,
                                    norm_cfg=norm_cfg
                                    )
                for stage in range(1, self.out_log2_scale-1)
            ]
        )

        self.upsample_layer = build_upsample_layer(self.up_sample_cfg)


    def _stage_to_out_channel(self, stage):
        return min(int(self.base_channels / (2**(stage * self.channel_decay))), self.max_channels)

    def forward(self, noise,
                num_batches=0,
                transition_weight=1.,
                return_noise=False,
                curr_scale=-1
                ):

        if isinstance(noise, torch.Tensor):
            assert noise.shape[1] == self.noise_size
            assert noise.ndim == 2, ('The noise should be in shape of (n, c), '
                                     f'but got {noise.shape}')
            noise_batch = noise
        else:
            assert num_batches > 0
            # TODO: check pggan default noise type
            noise_batch = torch.randn((num_batches, self.noise_size))

        noise_batch = noise_batch.to(next(self.parameters()).device)

        # build current computational graph
        curr_log2_scale = self.out_log2_scale if curr_scale < 0 else int(np.log2(curr_scale))


        x = self.noise2feat(noise_batch)

        if curr_log2_scale < 3:
            out_img = self.to_rgb_layers[0](x)
        else:
            for layer_block in self.conv_blocks[:curr_log2_scale - 3]:
                x = layer_block(x)

            last_img = self.to_rgb_layers[curr_log2_scale-3](x)
            residual_img = self.upsample_layer(last_img)
            x = self.conv_blocks[curr_log2_scale-3](x)
            out_img = self.to_rgb_layers[curr_log2_scale-2](x)

            out_img = residual_img + transition_weight * (out_img - residual_img)

        if return_noise:
            return dict(fake_img=out_img, noise_batch=noise_batch)

        else:
            return out_img


@MODULES.register_module()
class PGGANDiscriminator(nn.Module):

    def __init__(self,
                 in_scale,
                 base_channels=8192,
                 max_channel=512,
                 in_channels=3,
                 channel_decay=1.,
                 min_batch_std=dict(group_size=4),
                 conv_cfg=dict(type="EqualizedConv2d"),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                 norm_cfg=dict(type='PixelNorm'),
                 ):
        super(PGGANDiscriminator, self).__init__()

        self.base_channels= base_channels
        self.max_channel = max_channel
        self.channel_decay = channel_decay
        self.in_log2_scale = int(np.log2(in_scale))

        self.from_rgb_layers = nn.ModuleList(
            [
                ConvModule(in_channels,
                           self._stage_to_out_channel(stage - 1),
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           conv_cfg=conv_cfg,
                           norm_cfg=None,
                           act_cfg=act_cfg
                           )
                for stage in range(2, self.in_log2_scale + 1)
            ]
        )


        self.layers = nn.ModuleList(
            [
                DisGeneralConvBlock(
                    self._stage_to_out_channel(stage+1),
                    self._stage_to_out_channel(stage),
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
                for stage in range(1, self.in_log2_scale -1)
            ]
        )

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.min_batch_std_layer = None
        if min_batch_std is not None:
            self.min_batch_std_layer = MinibatchStdDev(**min_batch_std)
            decision_in_channels = self._stage_to_out_channel(1) + 1
        else:
            decision_in_channels = self._stage_to_out_channel(1)

        self.decision = PGGANDecisionHead(decision_in_channels, self._stage_to_out_channel(0),
                                          conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, transition_weight=1., curr_scale=-1):

        curr_log2_scale = self.in_log2_scale if curr_scale < 4 else int(np.log2(curr_scale))

        original_img = x
        x = self.from_rgb_layers[curr_log2_scale - 2](x)
        if curr_log2_scale > 2:
            img_down = self.downsample(original_img)
            residual = self.from_rgb_layers[curr_log2_scale - 3](img_down)
            x = self.layers[curr_log2_scale - 3](x)
            x = residual + transition_weight * (x - residual)

            for layer in reversed(self.layers[:curr_log2_scale-3]):
                x = layer(x)

        if self.min_batch_std_layer:
            x = self.min_batch_std_layer(x)

        x = self.decision(x)

        return x.view(-1, 1)


    def _stage_to_out_channel(self, stage):
        return min(int(self.base_channels / (2**(stage * self.channel_decay))), self.max_channel)


if __name__=="__main__":
    curr_scales = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    generator = PGGANGenerator(512, 1024)
    discriminator = PGGANDiscriminator(in_scale=1024)

    for curr_scale in curr_scales:
        fake_imgs = generator(
            None,
            num_batches=4,
            curr_scale=curr_scale,
            transition_weight=0.5)

        out = discriminator(fake_imgs,
                      curr_scale=curr_scale,
                      transition_weight=0.5)

        print("fake_shape", fake_imgs.shape, "out_shape", out.shape)

