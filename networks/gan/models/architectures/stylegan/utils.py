import torch
from ..common import get_module_device

@torch.no_grad()
def get_mean_latent(generator, num_samples=4096, bs_per_repeat=1024):
    """Get mean latent of W space in Style-based GANs.

    Args:
        generator (nn.Module): Generator of a Style-based GAN.
        num_samples (int, optional): Number of sample times. Defaults to 4096.
        bs_per_repeat (int, optional): Batch size of noises per sample.
            Defaults to 1024.

    Returns:
        Tensor: Mean latent of this generator.
    """
    device = get_module_device(generator)
    mean_style = None
    n_repeat = num_samples // bs_per_repeat
    assert n_repeat * bs_per_repeat == num_samples

    for _ in range(n_repeat):
        style = generator.style_mapping(
            torch.randn(bs_per_repeat,
                        generator.style_channels).to(device)).mean(
                            0, keepdim=True)
        if mean_style is None:
            mean_style = style
        else:
            mean_style += style
    mean_style /= float(n_repeat)

    return mean_style

