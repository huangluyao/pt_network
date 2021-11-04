from test_utils.engine.hooks import Hook, HOOKS

class BaseEvalHook(Hook):
    """Classes eval hook.

    Attributes:
        dataloader: 验证数据集
        start: 从第几轮开始验证
        evaluation_first: 再训练没有开始之前，先做一次验证。默认False
        interval： 验证的间隔
        **kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """
    def __init__(self,
                 dataloader,
                 start=None,
                 evaluation_first=False,
                 interval=1,
                 **kwargs
                 ):
        self.interval = interval
        self.start = start
        self.evaluation_first = evaluation_first
        self.dataloader = dataloader
        self.total_epoch_losses = []


    def before_train_epoch(self, runner):
        if self.evaluation_first:
            self.evaluation_first=False
            self.after_train_epoch(runner)

    def after_train_iter(self, runner):
        self.total_epoch_losses.append(float(runner.outputs['loss']))

    def after_train_epoch(self, runner):
        if not self.evaluation_flag(runner):
            return

        epoch_loss = sum(self.total_epoch_losses)/(len(self.total_epoch_losses)+1e-6)
        param_group = runner.optimizer.param_groups[0]

        self.evaluate(param_group["lr"],
                epoch_loss,
                self.dataloader,
                runner.model,
                logger=runner.logger,
                )

        self.total_epoch_losses = []

    def evaluation_flag(self, runner):
        if self.start is None:
            if not self.every_n_epochs(runner, self.interval):
                # No evaluation during the interval epochs.
                return False
        elif (runner.epoch + 1) < self.start:
            # No evaluation if start is larger than the current epoch.
            return False
        else:
            # Evaluation only at epochs 3, 5, 7... if start==3 and interval==2
            if (runner.epoch + 1 - self.start) % self.interval:
                return False

        return True

    def evaluate(self,learning_rate, avg_losses, dataloader, model, runner=None, threshold=None, logger=None, **kwargs):
        raise InterruptedError
