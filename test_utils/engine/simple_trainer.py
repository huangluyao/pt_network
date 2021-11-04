# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/11 上午11:10
import torch

from .train_base import TrainerBase
from .utils import get_host_info
from .builder import RUNNERS

@RUNNERS.register_module()
class SimplerTrainer(TrainerBase):
    def __init__(self, train_dataloader, **kwargs):
        super(SimplerTrainer, self).__init__(**kwargs)

        self.data_loader = train_dataloader


    def run_iter(self, img_batch, label_batch=None, return_metrics=True):
        outputs = self.model(img_batch, return_metrics=return_metrics,
                            ground_truth=label_batch)

        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        self.outputs = outputs

    def run(self, max_epochs=None, **kwargs):
        """
        训练函数
        Args:
            workflow (list[tuple]): 一个(phase, epochs)的列表去指定运行和验证的次数
                如 [('train', 2), ('val', 1)] 代表训练两次， 验证一次
            max_epochs (int): 总共训练的次数
        """
        if max_epochs is not None:
            self.max_epochs = max_epochs
        assert self.max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        self._max_iters = self.max_epochs * len(self.data_loader)
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)

        self.before_train()

        while self.epoch < self.max_epochs:
            self.train(**kwargs)

        # 等待其他进程完成
        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)

        self.after_train()

    def train(self, **kwargs):
        self.mode="train"
        self._max_iters = self.max_epochs * len(self.data_loader)
        self.before_train_epoch()
        self.model.train()

        # 等待其他进程完成
        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)

        for i, (img_batch, label_batch) in enumerate(self.data_loader):
            if not isinstance(img_batch, torch.Tensor):
                img_batch = torch.from_numpy(img_batch).to(self.device, dtype=torch.float32)
            if img_batch.device != self.device:
                img_batch = img_batch.to(self.device)
            self.inner_iter = i
            self.before_train_iter()
            self.run_iter(img_batch, label_batch, return_metrics=True)
            self.after_train_iter()

            self.iter += 1

        # 等待其他进程完成
        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)

        self.after_train_epoch()
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.before_val_epoch()
        for i, (img_batch, label_batch) in enumerate(data_loader):
            self._inner_iter = i
            self.before_val_iter()
            with torch.no_grad():
                self.run_iter(img_batch, return_metrics=False)
            self.after_val_iter()
        self.after_val_epoch()

