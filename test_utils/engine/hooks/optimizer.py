import os

import torch
import torch.nn as nn
from .hook import HOOKS, Hook
from torch.nn.utils import clip_grad
from test_utils.evaluator.draw_plot import static_bn
from test_utils.utils.pruned_utils import get_ignore_bn_list, get_bn_weights

@HOOKS.registry()
class OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        # 梯度裁切参数 判断是否使用梯度裁切, 若在训练的过程中，loss暴增可以使用
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad(set_to_none=True)
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()


@HOOKS.registry()
class OptimizerSparsityHook(OptimizerHook):

    def __init__(self, sr=1e-4, dynamic_sr=False,
                 save_bn=True,
                 interval=50,
                 stop_sparsity_epoch=0,
                 **kwargs):
        super(OptimizerSparsityHook, self).__init__(**kwargs)
        self.sr = sr
        self.dynamic_sr = dynamic_sr
        self.save_bn = save_bn
        self.interval = interval
        self.stop_sparsity_epoch = stop_sparsity_epoch

    def before_train(self, runner):
        self.ignore_bn_list = get_ignore_bn_list(runner.model)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])

        use_sparsity = False
        if self.stop_sparsity_epoch == 0:
            use_sparsity = True
        elif runner._epoch < self.stop_sparsity_epoch:
            use_sparsity = True

        if use_sparsity:
            if self.dynamic_sr:
                srtmp = self.sr * (1 - 0.9 * runner._epoch / runner.max_epochs)
            else:
                srtmp = self.sr

            for k, m in runner.model.named_modules():
                if isinstance(m, nn.BatchNorm2d) and (k not in self.ignore_bn_list):
                    if m.weight.grad is not None:
                        m.weight.grad.data.add_(srtmp * torch.sign(m.weight.data))  # L1
                        m.bias.grad.data.add_(srtmp * torch.sign(m.bias.data))  # L1

        runner.optimizer.step()

    def after_epoch(self, runner):

        if runner._epoch % self.interval == 0:
            bn_weights, bnb_weights = get_bn_weights(runner.model, self.ignore_bn_list)
            save_weights_path = os.path.join(runner.output_dir, f"bn_weights_at_epoch{runner._epoch}.png")
            save_bias_path = os.path.join(runner.output_dir, f"bn_bias_at_epoch{runner._epoch}.png")

            static_bn(bn_weights.numpy(), save_weights_path)
            static_bn(bnb_weights.numpy(), save_bias_path)

