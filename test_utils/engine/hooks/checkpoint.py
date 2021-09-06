import os

from .hook import HOOKS, Hook

@HOOKS.registry()

class CheckpointHook(Hook):
    """保存网络模型及相关参数。
    Args:
        interval (int): 保存迭代次数。如果by_epoch为True代表保存的是epoch，
                        否则代表保存的是iterations.默认是-1不保存
        by_epoch (bool): 保存训练模型通过每个epoch或者是iterations。默认True
        save_optimizer (bool): 是否保存优化器参数信息
        out_dir (str, optional): 网络模型的保存路径
        max_keep_ckpts (int, optional):保存左右的模型。
    """
    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.args = kwargs


    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return

        runner.logger.info(f'Saving checkpoint at {runner.epoch + 1} epochs')
        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            filename_tmpl = self.args.get('filename_tmpl', 'epoch_{}.pth')
            current_epoch = runner.epoch + 1
            for epoch in range(current_epoch - self.max_keep_ckpts, 0, -1):
                ckpt_path = os.path.join(self.out_dir,
                                         filename_tmpl.format(epoch))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break


    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return

        runner.logger.info(
            f'Saving checkpoint at {runner.iter + 1} iterations')
        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            filename_tmpl = self.args.get('filename_tmpl', 'iter_{}.pth')
            current_iter = runner.iter + 1
            for _iter in range(
                    current_iter - self.max_keep_ckpts * self.interval, 0,
                    -self.interval):
                ckpt_path = os.path.join(self.out_dir,
                                         filename_tmpl.format(_iter))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break

