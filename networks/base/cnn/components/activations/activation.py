import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import ACTIVATION_LAYERS

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


@ACTIVATION_LAYERS.register_module()
class Sine(nn.Module):
	def __init__(self, w0 = 1.):
		super().__init__()
		self.w0 = w0
	def forward(self, x):
		return torch.sin(self.w0 * x)


@ACTIVATION_LAYERS.register_module()
class Mish(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x *( torch.tanh(F.softplus(x)))

@ACTIVATION_LAYERS.register_module()
class FReLU(nn.Module):
	def __init__(self, c1, k=3):  # ch_in, kernel
		super().__init__()
		self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
		self.bn = nn.BatchNorm2d(c1)

	def forward(self, x):
		return torch.max(x, self.bn(self.conv(x)))

@ACTIVATION_LAYERS.register_module()
class AconC(nn.Module):
	def __init__(self, c1):
		super().__init__()
		self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
		self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
		self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

	def forward(self, x):
		dpx = (self.p1 - self.p2) * x
		return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x

@ACTIVATION_LAYERS.register_module()
class MetaAconC(nn.Module):
	def __init__(self, c1, k=1, s=1, r=16):  # ch_in, kernel, stride, r
		super().__init__()
		c2 = max(r, c1 // r)
		self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
		self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
		self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)
		self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)


	def forward(self, x):
		y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
		beta = torch.sigmoid(self.fc2(self.fc1(y)))  # bug patch BN layers removed
		dpx = (self.p1 - self.p2) * x
		return dpx * torch.sigmoid(beta * dpx) + self.p2 * x


@ACTIVATION_LAYERS.register_module()
class xUnitS(nn.Module):
	def __init__(self, c1=64, kernel_size=7, batch_norm=False):
		super(xUnitS, self).__init__()
		# slim xUnit
		self.features = nn.Sequential(
			nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=kernel_size, padding=(kernel_size // 2), groups=c1),
			nn.BatchNorm2d(num_features=c1) if batch_norm else nn.Identity(),
			nn.Sigmoid()
		)

	def forward(self, x):
		a = self.features(x)
		r = x * a
		return r


@ACTIVATION_LAYERS.register_module()
class xUnitD(nn.Module):
	def __init__(self, c1=64, kernel_size=7, batch_norm=False):
		super(xUnitD, self).__init__()
		# dense xUnit
		self.features = nn.Sequential(
			nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=1, padding=0),
			nn.BatchNorm2d(num_features=c1) if batch_norm else nn.Identity(),
			nn.ReLU(),
			nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=kernel_size, padding=(kernel_size // 2), groups=c1),
			nn.BatchNorm2d(num_features=c1) if batch_norm else nn.Identity(),
			nn.Sigmoid()
		)

	def forward(self, x):
		a = self.features(x)
		r = x * a
		return r


@ACTIVATION_LAYERS.register_module()
class SMU(nn.Module):
	"""
	Implementation of SMU activation.
	    Parameters:
        - alpha: hyper parameter
    References:
        - See related paper:
        https://arxiv.org/abs/2111.04682
	"""

	def __init__(self, alpha=0.25):
		super(SMU, self).__init__()
		self.alpha = alpha
		# initialize mu
		self.mu = torch.nn.Parameter(torch.tensor(1000000.0))

	def forward(self, x):
		return ((1 + self.alpha) * x + (1 - self.alpha) * x * torch.erf(self.mu * (1 - self.alpha) * x)) / 2


@ACTIVATION_LAYERS.register_module()
class SMU1(nn.Module):
	'''
    Implementation of SMU-1 activation.
    Parameters:
        - alpha: hyper parameter
    References:
        - See related paper:
        https://arxiv.org/abs/2111.04682
    '''

	def __init__(self, alpha=0.25):
		'''
        Initialization.
        INPUT:
            - alpha: hyper parameter
            aplha is initialized with zero value by default
        '''
		super(SMU1, self).__init__()
		self.alpha = alpha
		# initialize mu
		self.mu = torch.nn.Parameter(torch.tensor(4.352665993287951e-9))

	def forward(self, x):
		return ((1 + self.alpha) * x + torch.sqrt(torch.square(x - self.alpha * x) + torch.square(self.mu))) / 2