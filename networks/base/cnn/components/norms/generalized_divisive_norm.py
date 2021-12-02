import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Function
from ..registry import NORM_LAYERS


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size()) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


@NORM_LAYERS.register_module()
class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2 ** -18,
                 **kwargs):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.FloatTensor([reparam_offset])

        self.build(ch, torch.device(device))

    def build(self, ch, device):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** .5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta.to(device))

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma.to(device))
        self.pedestal = self.pedestal.to(device)

    def forward(self, inputs):
        device_id = inputs.device.index

        beta = self.beta.to(device_id)
        gamma = self.gamma.to(device_id)
        pedestal = self.pedestal.to(device_id)

        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(beta, self.beta_bound)
        beta = beta ** 2 - pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(gamma, self.gamma_bound)
        gamma = gamma ** 2 - pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs

