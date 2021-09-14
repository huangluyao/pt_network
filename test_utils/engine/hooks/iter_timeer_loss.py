import time
import torch
from .hook import HOOKS, Hook
from test_utils.utils.distributed_utils import reduce_value

@HOOKS.registry()
class IterTimerHook(Hook):
    def before_train_epoch(self, runner):
        runner.logger.info('-' * 25 + 'epoch: %d/%d' % (runner._epoch, runner.max_epochs) + '-' * 25)

    def before_iter(self, runner):
        self.t = time.time()

    def after_iter(self, runner):
        step_time =  time.time() - self.t

        # 等待所有进程计算完毕
        if runner.device != torch.device("cpu"):
            torch.cuda.synchronize(runner.device)

        outputs = {key:reduce_value(loss, average=True) for key, loss in runner.outputs.items()}

        cur_lr = runner.optimizer.param_groups[0]["lr"]
        if runner.inner_iter % 10 == 0:
            step_status = '=> Step %6d \tTime %5.2f \tLr %2.6f \t[Loss]:' % (
                runner.inner_iter, step_time, cur_lr)
            for key in outputs:
                step_status += ' %s: %7.4f' % (key, outputs[key].detach().cpu().numpy())
            runner.logger.info(step_status)
