import torch
from .train_base import TrainerBase
from .utils import get_host_info

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    from torch.cuda.amp import GradScaler
except ImportError:
    pass


class IterLoader:

    def __init__(self, dataloader):

        self._data_loader = dataloader
        self.iter_loader = iter(self._data_loader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch


    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch +=1
            self.iter_loader = iter(self._data_loader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._data_loader)


class DynamicIterTrainer(TrainerBase):

    def __init__(self,
                 *args,
                 train_dataloader,
                 is_dynamic_ddp=False,
                 pass_training_status=False,
                 fp16_loss_scaler=None,
                 use_apex_amp=False,
                 **kwargs):
        super(DynamicIterTrainer, self).__init__(*args, **kwargs)

        self.data_loader = IterLoader(train_dataloader)
        self.meta = None
        self.is_dynamic_ddp = is_dynamic_ddp
        self.pass_training_status = pass_training_status

        # add a flag for checking if 'self.optimizer' comes from '_model'
        self.optimizer_from_model = False

        if hasattr(self.model, 'optimizer'):
            assert self.optimizer is None, (
                'Trainer and model cannot contain optimizer at the same time.'
            )

            self.optimizer_from_model = True
            self.optimizer = self.model.optimizer

        # add fp16 grad scaler, using pytorch official GradScaler
        self.with_fp16_grad_scaler = False
        if fp16_loss_scaler is not None:
            self.loss_scaler = GradScaler(**fp16_loss_scaler)
            self.with_fp16_grad_scaler = True
            self.logger.info('Use FP16 grad scaler in Training')

        # flag to use amp in apex (NVIDIA)
        self.use_apex_amp = use_apex_amp

    def train(self, **kwargs):

        self.model.train()
        self.mode = 'train'

        while self.iter < self.max_iters:
            data_batch = next(self.data_loader)

            for k, v in data_batch.items():
                try:
                    data_batch[k] = v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v).to(self.device)
                except:
                    pass

            self.before_train_iter()
            self.run_iter(data_batch, **kwargs)
            self.after_train_iter()

            self.inner_iter += 1
            self.iter += 1


    def run(self, **kwargs):

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)

        self.before_train()

        self.train(**kwargs)

        # 等待其他进程完成
        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)

        self.after_train()
        self.logger.info("finish train.")

    def run_iter(self, img_batch, **kwargs):
        outputs = self.model.train_step(img_batch, self.optimizer, **kwargs)

        if self.with_fp16_grad_scaler:
            self.loss_scaler.update()

        if self.optimizer_from_model:
            self.optimizer = self.model.optimizer

        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        self.outputs = outputs

    def val(self, data_loader, **kwargs):
        pass


