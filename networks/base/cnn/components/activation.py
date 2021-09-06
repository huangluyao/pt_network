import torch
import torch.nn as nn

from ...utils import build_from_cfg
from .registry import ACTIVATION_LAYERS

for module in [
        nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.ReLU6, nn.ELU,
        nn.Sigmoid, nn.Tanh, nn.Identity
]:
    ACTIVATION_LAYERS.register_module(module=module)


@ACTIVATION_LAYERS.register_module()
class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x*torch.sigmoid(x)


@ACTIVATION_LAYERS.register_module()
class HardSwish(nn.Module):

	def __init__(self):
		super(HardSwish, self).__init__()

	def forward(self, inputs):
		clip = torch.clamp(inputs + 3, 0, 6) / 6
		return inputs * clip


@ACTIVATION_LAYERS.register_module()
class HardSigmoid(nn.Module):

	def __init__(self):
		super(HardSigmoid, self).__init__()

	def forward(self, inputs):
		return torch.clamp(inputs + 3, 0, 6) / 6


def build_activation_layer(cfg):
    """Build activation layer.

    Parameters
    ----------
    cfg : dict
        The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns
    -------
    act_layer : nn.Module
        Created activation layer.
    """
    return build_from_cfg(cfg, ACTIVATION_LAYERS)