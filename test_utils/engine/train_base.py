# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/11 上午11:11
import os
from .hooks import Hook, HOOKS
from test_utils.engine.hooks.priority import get_priority
from ..utils.registry import build_from_cfg
from abc import ABCMeta, abstractmethod


class TrainerBase(metaclass=ABCMeta):

    def __init__(self,
                 model,
                 optimizer=None,
                 max_iters=None,
                 max_epochs=None,
                 work_dir=None,
                 logger=None,
                 **kwargs
                ):
        # check Args
        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')
        if os.path.exists(work_dir):
            self.work_dir = os.path.abspath(work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError(f'"work_dir" must be a str or None, or not find f{work_dir}"')

        self._hooks = []
        self._epoch = 0
        self.iter = 0
        self.inner_iter = 0
        self.max_epochs = max_epochs
        self.max_iters = max_iters
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self.mode = None


    def register_hook(self, cfg, priority="NORMAL"):

        hook = None
        if isinstance(cfg, dict):
            hook = build_from_cfg(cfg, HOOKS)

        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)


    def after_train_epoch(self):
        for hook in self._hooks:
            hook.after_train_epoch(self)

    def after_train_iter(self):
        for hook in self._hooks:
            hook.after_train_iter(self)

    def before_train_iter(self):
        for hook in self._hooks:
            hook.before_train_iter(self)

    def before_train_epoch(self):
        for hook in self._hooks:
            hook.before_train_epoch(self)

    def before_train(self):
        for hook in self._hooks:
            hook.before_train(self)

    def after_train(self):
        for hook in self._hooks:
            hook.after_train(self)

    def before_val_epoch(self):
        for hook in self._hooks:
            hook.before_val_epoch(self)

    def before_val_iter(self):
        for hook in self._hooks:
            hook.before_val_iter(self)

    def after_val_iter(self):
        for hook in self._hooks:
            hook.after_val_iter(self)

    def after_val_epoch(self):
        for hook in self._hooks:
            hook.after_val_iter(self)


    @abstractmethod
    def train(self, data_loader, **kwargs):
        pass

    @abstractmethod
    def val(self, data_loader, **kwargs):
        pass

    @abstractmethod
    def run(self, data_loaders, workflow, **kwargs):
        pass

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch
