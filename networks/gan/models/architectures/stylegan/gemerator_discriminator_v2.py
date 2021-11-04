import random
import torch
import torch.nn as nn
import numpy as np
from ..pggan import PixelNorm
from ...builder import MODULES
from .utils import get_module_device
from .modules import (EqualLinearActModule, ConstantInput, ModulatedStyleConv, ModulatedToRGB,
                      ConvDownLayer, ResBlock, ModMBStddevLayer
                      )

@MODULES.register_module()
class StyleGANv2Generator(nn.Module):
    r"""StyleGAN2 Generator.

    In StyleGAN2, we use a static architecture composing of a style mapping
    module and number of convolutional style blocks. More details can be found
    in: Analyzing and Improving the Image Quality of StyleGAN CVPR2020.
    """
    def __init__(self,
                 out_size,
                 style_channels,
                 num_mlps=8,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 default_style_mode='mix',
                 mix_prob=0.9,
                 pretrained=None):
        super(StyleGANv2Generator, self).__init__()
        self.style_channels = style_channels
        if isinstance(out_size, int):
            self.out_size = [out_size, out_size]
        else:
            self.out_size = out_size
        self.default_style_mode = default_style_mode
        self.mix_prob = mix_prob

        # define style mapping layers
        mapping_layers = [PixelNorm()]
        for _ in range(num_mlps):
            mapping_layers.append(
                EqualLinearActModule(
                    style_channels,
                    style_channels,
                    equalized_lr_cfg=dict(lr_mul=lr_mlp, gain=1.),
                    act_cfg=dict(type='fused_bias')))

        self.style_mapping = nn.Sequential(*mapping_layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        # constant input layer
        self.constant_input = ConstantInput(self.channels[4])

        # 4x4 stage
        self.conv1 = ModulatedStyleConv(
            self.channels[4],
            self.channels[4],
            kernel_size=3,
            style_channels=style_channels,
            blur_kernel=blur_kernel)

        self.to_rgb1 = ModulatedToRGB(
            self.channels[4],
            style_channels,
            upsample=False)

        # generator backbone (8x8 --> higher resolutions)
        self.log_size = int(np.log2(self.out_size[0]))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channels_ = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channels_ = self.channels[2**i]

            self.convs.append(
                ModulatedStyleConv(
                    in_channels_,
                    out_channels_,
                    3,
                    style_channels,
                    upsample=True,
                    blur_kernel=blur_kernel))
            self.convs.append(
                ModulatedStyleConv(
                    out_channels_,
                    out_channels_,
                    3,
                    style_channels,
                    upsample=False,
                    blur_kernel=blur_kernel))
            self.to_rgbs.append(
                ModulatedToRGB(
                    out_channels_,
                    style_channels,
                    upsample=True))  # set to global fp16

            in_channels_ = out_channels_

        self.num_latents = self.log_size * 2 - 2
        self.num_injected_noises = self.num_latents - 1

        # register buffer for injected noises
        for layer_idx in range(self.num_injected_noises):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            self.register_buffer(f'injected_noise_{layer_idx}',
                                 torch.randn(*shape))


    def forward(self,
                styles,
                num_batches=-1,
                return_noise=False,
                return_latents=False,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                injected_noise=None,
                randomize_noise=True):

        device = get_module_device(self)
        assert num_batches > 0 and not input_is_latent
        if self.default_style_mode == 'mix' and random.random() < self.mix_prob:
            styles = [
                torch.randn((num_batches, self.style_channels))
                for _ in range(2)
            ]
        else:
            styles = [torch.randn((num_batches, self.style_channels))]
        styles = [s.to(device) for s in styles]

        noise_batch = styles
        styles = [self.style_mapping(s) for s in styles]

        if injected_noise is None:
            if randomize_noise:
                injected_noise = [None] * self.num_injected_noises
            else:
                injected_noise = [
                    getattr(self, f'injected_noise_{i}')
                    for i in range(self.num_injected_noises)
                ]

        # no style mixing
        if len(styles) < 2:
            inject_index = self.num_latents

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        # style mixing
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latents - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.num_latents - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        # 4x4 stage
        out = self.constant_input(latent)
        out = self.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        _index = 1

        # 8x8 ---> higher resolutions
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], injected_noise[1::2],
                injected_noise[2::2], self.to_rgbs):
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)
            _index += 2

        # make sure the output image is torch.float32 to avoid RunTime Error
        # in other modules
        img = skip.to(torch.float32)

        if return_latents or return_noise:
            output_dict = dict(
                fake_img=img,
                latent=latent,
                inject_index=inject_index,
                noise_batch=noise_batch)
            return output_dict

        return img


@MODULES.register_module()
class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator.

    The architecture of this discriminator is proposed in StyleGAN2. More
    details can be found in: Analyzing and Improving the Image Quality of
    StyleGAN CVPR2020.
    """
    def __init__(self,
                 in_size,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 mbstd_cfg=dict(group_size=4, channel_groups=1),
                 pretrained=None):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        log_size = int(np.log2(in_size))
        in_channels = channels[in_size]

        convs = [
            ConvDownLayer(3, channels[in_size], 1)
        ]

        for i in range(log_size, 2, -1):
            out_channel = channels[2**(i - 1)]

            convs.append(
                ResBlock(
                    in_channels,
                    out_channel,
                    blur_kernel))

            in_channels = out_channel

        self.convs = nn.Sequential(*convs)
        self.mbstd_layer = ModMBStddevLayer(**mbstd_cfg)

        self.final_conv = ConvDownLayer(in_channels + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinearActModule(
                channels[4] * 4 * 4,
                channels[4],
                act_cfg=dict(type='fused_bias')),
            EqualLinearActModule(channels[4], 1),
        )

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Predict score for the input image.
        """
        x = self.convs(x)

        x = self.mbstd_layer(x)

        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)

        return x