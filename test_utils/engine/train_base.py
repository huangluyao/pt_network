# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/11 上午11:11
import time


class TrainerBase:
    def __init__(self, max_epochs, logger, **kwargs):

        #  Register hooks to the trainer.
        self.before_train_hooks = []
        self.after_train_hooks = []
        self.before_epoch_hooks = []
        self.after_epoch_hooks = []

        self.iter = 0
        self.start_iter = 0
        self.max_epochs = max_epochs
        self.logger = logger

    def register_before_train_hook(self, hook):
        assert callable(hook), 'hook must be callable object'
        self.before_train_hooks.append(hook)

    def register_after_train_hook(self, hook):
        assert callable(hook), 'hook must be callable object'
        self.after_train_hooks.append(hook)

    def register_before_epoch_hook(self, hook):
        assert callable(hook), 'hook must be callable object'
        self.before_epoch_hooks.append(hook)

    def register_after_epoch_hook(self, hook):
        assert callable(hook), 'hook must be callable object'
        self.after_epoch_hooks.append(hook)

    def train(self):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        self.logger.info("Starting training")
        self.before_train()
        for cur_epoch in range(1, self.max_epochs):

            self.logger.info('-' * 25 + 'epoch: %d/%d' % (cur_epoch, self.max_epochs) + '-' * 25)
            self.before_train_one_epoch()
            epoch_st = time.time()
            self.train_one_epoch()
            epoch_time = time.time() - epoch_st
            epoch_status = ("Epoch %6d \tTime %5.2f"
                            %(cur_epoch, epoch_time))

            self.logger.info(epoch_status)
            self.after_train_one_epoch()

        self.after_train()

    def train_one_epoch(self):
        return NotImplementedError


    def before_train(self):
        for before_train_hook in self.before_train_hooks:
            before_train_hook()

    def after_train(self):
        for after_train_hook in self.after_train_hooks:
            after_train_hook()

    def before_train_one_epoch(self):
        for before_epoch_hook in self.before_epoch_hooks:
            before_epoch_hook()

    def after_train_one_epoch(self):
        for after_epoch_hook in self.after_epoch_hooks:
            after_epoch_hook()
