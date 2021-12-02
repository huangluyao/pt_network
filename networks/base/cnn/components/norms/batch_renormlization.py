import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchRenormalization(nn.Module):

    def __init__(self, num_features, eps=1e-6, momentum=0.1, affine=True,
                 rmax_inc_step=1000, dmax_inc_step=500):
        super(BatchRenormalization, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.weight = nn.Parameter(
            torch.tensor(num_features, dtype=torch.float), requires_grad=True
        )
        self.bias = nn.Parameter(
            torch.tensor(num_features, dtype=torch.float), requires_grad=True
        )

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_std", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.rmax_inc_step = rmax_inc_step
        self.dmax_inc_step = dmax_inc_step
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_std.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key not in state_dict:
            state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(BatchRenormalization, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


    @property
    def rmax(self):
        return (2 / self.rmax_inc_step * self.num_batches_tracked + 25 / 35).clamp_(1.0, 3.0)

    @property
    def dmax(self):
        return (5 / self.dmax_inc_step * self.num_batches_tracked - 25 / 20).clamp_(0.0, 5.0)

    def forward(self, x):

        if self.training:
            self.num_batches_tracked = self.num_batches_tracked + 1

            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exponential_average_factor = self.momentum

            dims = tuple([i for i in range(x.dim()) if i != 1])
            batch_mean = x.mean(dims, keepdim=True)
            batch_std = x.std(dims, unbiased=False, keepdim=True) + self.eps

            r = (batch_std.detach() / self.running_std.view_as(batch_std)).clamp_(1/self.rmax, self.rmax)
            d = ((batch_mean.detach() - self.running_mean.view_as(batch_std) ) / self.running_std.view_as(batch_std)
                 ).clamp_(-self.dmax, self.dmax)

            x = (x - batch_mean) / batch_std * r + d

            self.running_mean += exponential_average_factor * (batch_mean.detach().view_as(self.running_mean) - self.running_mean)
            self.running_std += exponential_average_factor * (batch_std.detach().view_as(self.running_std) - self.running_std)

        else:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_std,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )
        if self.affine:
            x = self.weight * x + self.bias
        return x