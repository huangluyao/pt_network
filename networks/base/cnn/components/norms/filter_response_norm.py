import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..registry import NORM_LAYERS


class TLU(nn.Module):
    def __init__(self, num_features, ndim):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        shape = (1, num_features) + (1, ) * (ndim - 2)
        self.tau = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)


@NORM_LAYERS.register_module('FRN')
class FilterResponseNorm(_BatchNorm):
    """Filter Response Normalization

    See 'Filter Response Normalization Layer:
    Eliminating Batch Dependence in the Training of Deep Neural Networks'
    (https://arxiv.org/abs/1911.09737) for details.

    Parameters
    ----------
    num_features : int
        An integer indicating the number of input feature dimensions.
    ndim : int
        An integer indicating the number of dimensions of the expected input tensor.
    eps : float
        A scalar constant or learnable variable.
    is_eps_leanable : bool
        A bool value indicating whether the eps is learnable.
    """
    def __init__(self, num_features, ndim=4, eps=1e-6,
                 is_eps_leanable=True, leanable_eps_value=0.01):
        assert ndim in [3, 4, 5], \
            'FilterResponseNorm only supports 3d, 4d or 5d inputs.'
        super(FilterResponseNorm, self).__init__(num_features, eps=eps)
        self.shape = (1, num_features) + (1, ) * (ndim - 2)
        if is_eps_leanable:
            self.epsilon = nn.Parameter(torch.ones(*self.shape) * leanable_eps_value)
        else:
            self.register_buffer('epsilon', torch.ones(*self.shape) * eps)
        self.actvation = TLU(num_features, ndim)

    def forward(self, x):
        avg_dims = tuple(range(2, x.dim()))
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.epsilon + self.eps))
        x = self.weight.view(self.shape) * x + self.bias.view(self.shape)
        return self.actvation(x)
