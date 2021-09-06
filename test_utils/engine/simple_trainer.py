# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/11 上午11:10
import torch

from .train_base import TrainerBase
from .utils import get_host_info



class SimplerTrainer(TrainerBase):
    def __init__(self, train_dataloader, **kwargs):
        super(SimplerTrainer, self).__init__(**kwargs)

        self.data_loader = train_dataloader


    def run_iter(self, img_batch, label_batch=None, return_metrics=True):
        outputs = self.model(img_batch, return_metrics=True,
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

        # time.sleep(1)  # 等待其他的 hook 完成
        self.after_train()

    def train(self, **kwargs):
        self.mode="train"
        self._max_iters = self.max_epochs * len(self.data_loader)
        self.before_train_epoch()
        self.model.train()
        # time.sleep(2)
        for i, (img_batch, label_batch) in enumerate(self.data_loader):
            if not isinstance(img_batch, torch.Tensor):
                img_batch = torch.from_numpy(img_batch).to(self.model.device, dtype=torch.float32)
            if img_batch.device != self.model.device:
                img_batch = img_batch.to(self.model.device)
            self.inner_iter = i
            self.before_train_iter()
            self.run_iter(img_batch, label_batch, return_metrics=True)
            self.after_train_iter()
            self.iter += 1

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


#
# import time
# import os
# import tempfile
# import json
# import numpy as np
# import torch.utils.data
# from networks.cls import build_classifier
# from networks.det import build_detector
# from networks.seg import build_segmentor
# from .optimizer import build_optimizer, build_scheduler
# from .data_loader import build_data_loader
#
# from ..datasets import build_dataset, statistics_data
# from ..evaluator import Evaluator, Visualizer, model_info
# from ..utils.checkpoint import load_checkpoint
# from ..utils.ema import ModelEMA
# from ..utils.distributed_utils import init_distributed_mode, dist

# class SimpleTrainer(TrainerBase):
#     def __init__(self, cfg, logger):
#         super(SimpleTrainer, self).__init__(max_epochs=cfg["scheduler"]["max_epochs"],
#                                             logger=logger)
#         self.cfg = cfg
#         self.input_size = self.cfg['dataset'].pop('input_size')
#         self.max_epochs = self.cfg['scheduler']['max_epochs']
#
#         self._checkpoint_dir = os.path.join(cfg['output_dir'], 'checkpoints')
#         if not os.path.exists(self._checkpoint_dir):
#             os.makedirs(self._checkpoint_dir)
#         model_name = cfg['model']['type']
#         self.performance_dir = os.path.join(cfg['output_dir'], 'performance')
#
#         init_distributed_mode(self.cfg)
#
#         # statistics data info
#         self.logger.info('-' * 25 + 'statistics_data:' + '-' * 25)
#         if cfg["dataset"].get("statistics_data", None):
#             self.means = cfg["dataset"].get("statistics_data", None).get("means")
#             self.stds = cfg["dataset"].get("statistics_data", None).get("stds")
#         else:
#             self.means, self.stds = statistics_data([cfg['dataset']['train_data_path'], cfg['dataset']['val_data_path']],
#                                                     cfg['dataset']['type'])
#
#         self.logger.info(f'data means: {self.means}')
#         self.logger.info(f'data stds: {self.stds}')
#
#         # update means and stds
#         self.update_dateset_info(cfg['dataset'])
#
#         # build loader
#         self.train_data_loader, self.val_data_loader = build_data_loader(dataset_cfg=cfg.dataset,
#                                                                          loader_cfg=cfg['loader_cfg'],
#                                                                          is_dist=cfg.distributed)
#
#         logger.info('-' * 25 + 'mode info' + '-' * 25)
#
#         if cfg.distributed == False:
#             self.logger.info('Not using distributed mode')
#         else:
#             self.logger.info('distributed init (rank {}): {}'.format(
#                 cfg.rank, cfg.dist_url), flush=True)
#
#         # init model
#         self._device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = self.build_model(cfg, len(self.train_data_loader.dataset.class_names))
#         self.model = self.model.to(self._device)
#
#         self.optimizer = build_optimizer(self.model, cfg['optimizer'])
#         self.scheduler = build_scheduler(self.optimizer, cfg['scheduler'])
#         self.build_evaluator(model_name)
#
#         # 只有训练带有BN结构的网络时使用SyncBatchNorm
#         if self.cfg.distributed:
#             # 使用SyncBatchNorm后训练会更耗时
#             self.mode = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self._device)
#             # 转为DDP模型
#             self.mode = torch.nn.parallel.DistributedDataParallel(self.mode, device_ids=[cfg.gpu])
#
#         self.logger.info('model info:')
#         self.logger.info(model_info(self.model, self.input_size))
#
#         self.use_ema = self.cfg.get("use_ema", None)
#
#         if self.use_ema:
#             self.ema_model = ModelEMA(self.model, 0.9998)
#
#         # save config file
#         self.cfg["dataset"]["classes_name"] = self.train_data_loader.dataset.class_names
#         self.cfg['dataset']["input_size"] = self.input_size
#         json_path = os.path.join(self.cfg['output_dir'], "config.json")
#         with open(json_path, 'w') as f:
#             json.dump(self.cfg, f, indent=4)
#
#     def train_one_epoch(self, cur_epoch):
#         self._cur_epoch_losses = []
#         self._cur_epoch = cur_epoch
#         self.model.train()
#         for step, (img_batch, label_batch) in enumerate(self.train_data_loader):
#
#             if not isinstance(img_batch, torch.Tensor):
#                 img_batch = torch.from_numpy(img_batch).to(self._device, dtype=torch.float32)
#             if img_batch.device != self._device:
#                 img_batch = img_batch.to(self._device)
#             st = time.time()
#             losses = self.model(img_batch, return_metrics=True,
#                                 ground_truth=label_batch)
#             step_time = time.time() - st
#             self.model.zero_grad()
#             self.optimizer.zero_grad()
#             cur_loss = losses['loss']
#             cur_loss.backward()
#             self.optimizer.step()
#             if self.use_ema:
#                 self.ema_model.update(self.model)
#             cur_lr = self.scheduler.get_lr()[0]
#             self._cur_epoch_losses.append(cur_loss.detach().cpu().numpy())
#
#             if step % 10 == 0:
#                 step_status = '=> Step %6d \tTime %5.2f \tLr %2.6f \t[Loss]:' %(
#                     step, step_time, cur_lr)
#                 for key in losses:
#                     step_status += ' %s: %7.4f' %(key, losses[key].detach().cpu().numpy())
#                 self.logger.info(step_status)
#         self.mean_loss = np.mean(self._cur_epoch_losses)
#         epoch_status = ("Epoch %6d \t[loss]:\taverage loss:%7.8f \tdecreases:%7.4f"
#                         %(self._cur_epoch, self.mean_loss,
#                           self._cur_epoch_losses[0]-self._cur_epoch_losses[-1]))
#         self.logger.info(epoch_status)
#
#     def after_train_one_epoch(self):
#         super(SimpleTrainer, self).after_train_one_epoch()
#         self.scheduler.step()
#         if self.use_ema:
#             self.ema_model.update_attr(self.model)
#         self._val_feature = None
#         self.validate()
#         self._evaluator.record_cur_epoch(
#             learning_rate=self.scheduler.get_lr()[0],
#             avg_loss=self.mean_loss,
#             y_prob=self._val_pred,
#             y_true=self._val_true,
#             y_feature=self._val_feature,
#             confidence_threshold=self.vis_score_threshold
#         )
#
#         # save checkpoint
#         state_dict = self.ema_model.ema.state_dict() if self.use_ema else self.model.state_dict()
#         if self._evaluator.is_train_best_epoch():
#             print("Saving checkpoint of minimum training loss.")
#             state = {
#                 'arch': type(self.model).__name__,
#                 'epoch': self._cur_epoch,
#                 'state_dict': state_dict,
#                 'optimizer': self.optimizer.state_dict(),
#             }
#             torch.save(state, "%s/train_best.pth" % (self._checkpoint_dir))
#         if self._evaluator.is_val_best_epoch():
#             print("Saving checkpoint of best validation metrics.")
#             state = {
#                 'arch': type(self.model).__name__,
#                 'epoch': self._cur_epoch,
#                 'state_dict': state_dict,
#                 'optimizer': self.optimizer.state_dict(),
#             }
#             torch.save(state, "%s/val_best.pth" % (self._checkpoint_dir))
#
#             # visualize
#             if self._visualizer is not None:
#                 self._visualizer.visualize(
#                     self._val_img_paths, self._val_pred,
#                     vis_predictions_dir=self._vis_predictions_dir
#                 )
#
#     def after_train(self):
#         super(SimpleTrainer, self).after_train()
#         best_metric = self._evaluator.get_best_metric()
#         self.logger.info(f"the best metric is {best_metric}")
#         self.logger.info('epoch end ')
#
#
#     def update_dateset_info(self, pipelines):
#         if isinstance(pipelines, list):
#             for augmentation in pipelines:
#                 if isinstance(augmentation, dict):
#                     if "Normalize" == augmentation.get('type', None):
#                         augmentation.update(dict(mean=self.means, std=self.stds))
#                     elif "Resize" == augmentation.get('type', None) and self.input_size is not None:
#                         augmentation.update(dict(width=self.input_size[0], height=self.input_size[1]))
#                     elif "RandomCrop" == augmentation.get('type', None) and self.input_size is not None:
#                         augmentation.update(dict(width=self.input_size[0], height=self.input_size[1]))
#         elif isinstance(pipelines, dict):
#             for k, v in pipelines.items():
#                 self.update_dateset_info(v)
#         else:
#             pass
#
#     def build_model(self, cfg, num_classes):
#         model_cfg = cfg.get('model')
#         number_classes_model ={"classification":"backbone",
#                                "detection":"bbox_head",
#                                "segmentation":"decode_head"
#                                }
#         num_classes_cfg = model_cfg.get(number_classes_model[cfg.get("task")], None)
#         if num_classes_cfg:
#             num_classes_cfg.update(dict(num_classes=num_classes))
#         if cfg.get('task') == "classification":
#             model = build_classifier(model_cfg)
#         elif cfg.get('task') == "detection":
#             model_cfg["train_cfg"]["output_dir"] = cfg["output_dir"]
#             model = build_detector(model_cfg)
#         elif cfg.get('task') == "segmentation":
#             model = build_segmentor(model_cfg)
#         else:
#             raise TypeError(f"task must be classification, detection or segmentation, but go {cfg['task']}")
#
#         checkpoint_path = cfg.get("pretrained", "")
#         if os.path.exists(checkpoint_path):
#             checkpoint_path = os.path.expanduser(checkpoint_path)
#             load_checkpoint(model, checkpoint_path, map_location='cpu', strict=True)
#             self.logger.info(f"load checkpoint from {checkpoint_path}")
#
#         if self.cfg.distributed:
#             checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
#             if self.cfg.rank == 0:
#                 torch.save(model.state_dict(), checkpoint_path)
#             dist.barrier()
#             # 指定map_location参数，否则会导致第一块GPU占用更多资源
#             model.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
#             if self.cfg.rank == 0:
#                 if os.path.exists(checkpoint_path) is True:
#                     os.remove(checkpoint_path)
#         return model
#
#     def build_evaluator(self, model_name):
#         self._evaluator = Evaluator(self.cfg.task, model_name=model_name,
#                                     logger=self.logger,
#                                     num_classes=len(self.train_data_loader.dataset.class_names),
#                                     class_names=self.train_data_loader.dataset.class_names,
#                                     performance_dir=self.performance_dir
#                                     )
#
#         self._vis_predictions_dir = os.path.join(self.performance_dir, 'vis_predictions')
#
#         self._visualizer = Visualizer(
#                 task=self.cfg.task,
#                 class_names=self.train_data_loader.dataset.class_names,
#                 input_size=self.input_size
#             )
#         self.vis_score_threshold = None
#
#
#     def validate(self):
#         self.model.eval()
#         evalmodel = self.ema_model.ema if self.use_ema else self.model
#         evalmodel.eval()
#         img_paths = []
#         for step, (img_batch, label_batch) in enumerate(self.val_data_loader):
#             img_paths += [img_path for img_path in label_batch['image_path']]
#             if not isinstance(img_batch, torch.Tensor):
#                 img_batch = torch.from_numpy(img_batch).to(self._device, dtype=torch.float32)
#             pred = evalmodel(img_batch)
#             if step ==0:
#                 pred_array = pred.detach().cpu().numpy()
#                 gt_array = label_batch['gt_labels']
#             else:
#                 pred_array = np.concatenate([pred_array, pred.detach().cpu().numpy()], axis=0)
#                 gt_array = np.concatenate([gt_array, label_batch['gt_labels']], axis=0)
#
#         self._val_pred = pred_array
#         self._val_true = gt_array
#         self._val_img_paths = img_paths

