import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """UpFRIDn for 2d features.

    UpFIRDn is short for upsample, apply FIR filter and downsample. More
    details can be found in:
    https://www.mathworks.com/help/signal/ref/upfirdn.html

    Args:
        input (Tensor): Tensor with shape of (n, c, h, w).
        kernel (Tensor): Filter kernel.
        up (int, optional): Upsampling factor. Defaults to 1.
        down (int, optional): Downsampling factor. Defaults to 1.
        pad (tuple[int], optional): Padding for tensors, (x_pad, y_pad).
            Defaults to (0, 0).

    Returns:
        Tensor: Tensor after UpFIRDn.
    """
    out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0],
                           pad[1], pad[0], pad[1])
    return out


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1,
                     pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out,
        [0, 0,
         max(pad_x0, 0),
         max(pad_x1, 0),
         max(pad_y0, 0),
         max(pad_y1, 0)])
    out = out[:,
              max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0),
              max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :, ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)


class UpsampleUpFIRDn(nn.Module):
    """UpFIRDn for Upsampling.

    This module is used in the ``to_rgb`` layers in StyleGAN2 for upsampling
    the images.

    Args:
        kernel (Array): Blur kernel/filter used in UpFIRDn.
        factor (int, optional): Upsampling factor. Defaults to 2.
    """

    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = _make_kernel(kernel) * (factor**2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """
        out = upfirdn2d(
            x, self.kernel.to(x.dtype), up=self.factor, down=1, pad=self.pad)

        return out

class Blur(nn.Module):
    """Blur module.

    This module is adopted rightly after upsampling operation in StyleGAN2.

    Args:
        kernel (Array): Blur kernel/filter used in UpFIRDn.
        pad (list[int]): Padding for features.
        upsample_factor (int, optional): Upsampling factor. Defaults to 1.
    """

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = _make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """

        # In Tero's implementation, he uses fp32
        return upfirdn2d(x, self.kernel.to(x.dtype), pad=self.pad)