import time

from .hook import HOOKS, Hook


@HOOKS.registry()
class IterTimerHook(Hook):
    def before_train_epoch(self, runner):
        runner.logger.info('-' * 25 + 'epoch: %d/%d' % (runner._epoch, runner.max_epochs) + '-' * 25)

    def before_iter(self, runner):
        self.t = time.time()

    def after_iter(self, runner):
        step_time =  time.time() - self.t
        cur_lr = runner.optimizer.param_groups[0]["lr"]
        if runner.inner_iter % 10 == 0:
            step_status = '=> Step %6d \tTime %5.2f \tLr %2.6f \t[Loss]:' % (
                runner.inner_iter, step_time, cur_lr)
            for key in runner.outputs:
                step_status += ' %s: %7.4f' % (key, runner.outputs[key].detach().cpu().numpy())
            runner.logger.info(step_status)
