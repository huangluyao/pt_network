import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..common import get_module_device
from .modules import EqualLinearActModule, ConstantInput, ModulatedPEStyleConv, ModulatedToRGB
from ..pggan import PixelNorm
from ...builder import MODULES,build_module


@MODULES.register_module()
class MSStyleGANNv2Generator(nn.Module):
    """StyleGAN2 Generator.

    In StyleGAN2, we use a static architecture composing of a style mapping
    module and number of convolutional style blocks. More details can be found
    in: Analyzing and Improving the Image Quality of StyleGAN CVPR2020.

    Args:
        out_size (int): The output size of the StyleGAN2 generator.
        style_channels (int): The number of channels for style code.
        num_mlps (int, optional): The number of MLP layers. Defaults to 8.
        channel_multiplier (int, optional): The multiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        lr_mlp (float, optional): The learning rate for the style mapping
            layer. Defaults to 0.01.
        default_style_mode (str, optional): The default mode of style mixing.
            In training, we defaultly adopt mixing style mode. However, in the
            evaluation, we use 'single' style mode. `['mix', 'single']` are
            currently supported. Defaults to 'mix'.
        eval_style_mode (str, optional): The evaluation mode of style mixing.
            Defaults to 'single'.
        mix_prob (float, optional): Mixing probability. The value should be
            in range of [0, 1]. Defaults to 0.9.
    """
    def __init__(self,
                 out_size,
                 style_channels,
                 num_mlps=8,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 default_style_mode='mix',
                 eval_style_mode='single',
                 mix_prob=0.9,
                 no_pad=False,
                 deconv2conv=False,
                 interp_pad=None,
                 up_config=dict(scale_factor=2, mode='nearest'),
                 up_after_conv=False,
                 head_pos_encoding=None,
                 head_pos_size=(4, 4),
                 interp_head=False,
                 **kwargs):
        super(MSStyleGANNv2Generator, self).__init__()
        self.style_channels = style_channels
        self.no_pad = no_pad
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

        # define style mapping layers
        mapping_layers = [PixelNorm()]

        for _ in range(num_mlps):
            mapping_layers.append(
                EqualLinearActModule(
                    style_channels,
                    style_channels,
                    lr_mul=lr_mlp,
                    gain=1.,
                    act_cfg=dict(type='fused_bias')))

        self.style_mapping = nn.Sequential(*mapping_layers)

        size_ = head_pos_size
        if self.no_pad:
            size_ += 2
        self.constant_input = ConstantInput(self.channels[4], size=size_)

        # 4x4 stage
        self.conv1 = ModulatedPEStyleConv(
            self.channels[4],
            self.channels[4],
            kernel_size=3,
            style_channels=style_channels,
            blur_kernel=blur_kernel,
            deconv2conv=self.deconv2conv,
            no_pad=self.no_pad,
            up_config=self.up_config,
            interp_pad=self.interp_pad)

        self.to_rgb1 = ModulatedToRGB(
            self.channels[4], style_channels, upsample=False)

        # generator backbone (8x8 --> higher resolutions)
        self.log_size = int(np.log2(self.out_size))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channels_ = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channels_ = self.channels[2**i]

            self.convs.append(
                ModulatedPEStyleConv(
                    in_channels_,
                    out_channels_,
                    3,
                    style_channels,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    deconv2conv=self.deconv2conv,
                    no_pad=self.no_pad,
                    up_config=self.up_config,
                    interp_pad=self.interp_pad,
                    up_after_conv=self.up_after_conv))
            self.convs.append(
                ModulatedPEStyleConv(
                    out_channels_,
                    out_channels_,
                    3,
                    style_channels,
                    upsample=False,
                    blur_kernel=blur_kernel,
                    deconv2conv=self.deconv2conv,
                    no_pad=self.no_pad,
                    up_config=self.up_config,
                    interp_pad=self.interp_pad,
                    up_after_conv=self.up_after_conv))
            self.to_rgbs.append(
                ModulatedToRGB(out_channels_, style_channels, upsample=True))

            in_channels_ = out_channels_

        self.num_latents = self.log_size * 2 - 2
        self.num_injected_noises = self.num_latents - 1

        # register buffer for injected noises
        noises = self.make_injected_noise()
        for layer_idx in range(self.num_injected_noises):
            self.register_buffer(f'injected_noise_{layer_idx}',
                                 noises[layer_idx])


    def forward(self,
                styles=None,
                num_batches=-1,
                return_noise=False,
                return_latents=False,
                inject_index=None,
                chosen_scale=0):
        """Forward function.

        This function has been integrated with the truncation trick. Please
        refer to the usage of `truncation` and `truncation_latent`.

        Args:
            styles (torch.Tensor | list[torch.Tensor] | callable | None): In
                StyleGAN2, you can provide noise tensor or latent tensor. Given
                a list containing more than one noise or latent tensors, style
                mixing trick will be used in training. Of course, You can
                directly give a batch of noise through a ``torch.Tensor`` or
                offer a callable function to sample a batch of noise data.
                Otherwise, the ``None`` indicates to use the default noise
                sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            return_latents (bool, optional): If True, ``latent`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            inject_index (int | None, optional): The index number for mixing
                style codes. Defaults to None.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncation trick will be adopted. Defaults to 1.
            truncation_latent (torch.Tensor, optional): Mean truncation latent.
                Defaults to None.
            input_is_latent (bool, optional): If `True`, the input tensor is
                the latent tensor. Defaults to False.
            injected_noise (torch.Tensor | None, optional): Given a tensor, the
                random noise will be fixed as this input injected noise.
                Defaults to None.
            randomize_noise (bool, optional): If `False`, images are sampled
                with the buffered noise tensor injected to the style conv
                block. Defaults to True.

        Returns:
            torch.Tensor | dict: Generated image tensor or dictionary \
                containing more data.
        """
        # receive noise and conduct sanity check.
        if styles is None:
            device = get_module_device(self)
            assert num_batches > 0
            if self.default_style_mode == 'mix' and random.random() < self.mix_prob:
                styles = [torch.randn((num_batches, self.style_channels)) for _ in range(2)]
            else:
                styles = [torch.randn((num_batches, self.style_channels))]

            styles = [s.to(device) for s in styles]

        noise_batch = styles
        styles = [self.style_mapping(s) for s in styles]

        injected_noise = [None] * self.num_injected_noises

        # no style mixing
        if len(styles) < 2:
            inject_index = self.num_latents

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latents - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.num_latents - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        if isinstance(chosen_scale, int):
            chosen_scale = (chosen_scale, chosen_scale)

        out = self.constant_input(latent)
        if chosen_scale[0] != 0 or chosen_scale[1] != 0:
            out = F.interpolate(
                out,
                size=(out.shape[2] + chosen_scale[0],
                      out.shape[3] + chosen_scale[1]),
                mode='bilinear',
                align_corners=True)

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

        img = skip

        if return_latents or return_noise:
            output_dict = dict(
                fake_img=img,
                latent=latent,
                inject_index=inject_index,
                noise_batch=noise_batch,
                injected_noise=injected_noise)
            return output_dict

        return img


    def make_injected_noise(self, chosen_scale=0):
        device = get_module_device(self)

        base_scale = 2**2 + chosen_scale

        noises = [torch.randn(1, 1, base_scale, base_scale, device=device)]

        for i in range(3, self.log_size + 1):
            for n in range(2):
                _pad = 0
                if self.no_pad and not self.up_after_conv and n == 0:
                    _pad = 2
                noises.append(
                    torch.randn(
                        1,
                        1,
                        base_scale * 2**(i - 2) + _pad,
                        base_scale * 2**(i - 2) + _pad,
                        device=device))

        return noises


@MODULES.register_module()
class MSStyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator.

    The architecture of this discriminator is proposed in StyleGAN2. More
    details can be found in: Analyzing and Improving the Image Quality of
    StyleGAN CVPR2020.

    Args:
        in_size (int): The input size of images.
        channel_multiplier (int, optional): The multiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        mbstd_cfg (dict, optional): Configs for minibatch-stddev layer.
            Defaults to dict(group_size=4, channel_groups=1).
    """
    def __init__(self,
                 in_size,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 mbstd_cfg=dict(group_size=4, channel_groups=1),
                 with_adaptive_pool=False,
                 pool_size=(2, 2)):
        super().__init__()
        self.with_adaptive_pool = with_adaptive_pool
        self.pool_size = pool_size

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
        convs = [ConvDownLayer(3, channels[in_size], 1)]
