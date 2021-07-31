# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/11 上午11:10
import time
import os

import json
import numpy as np
import torch.utils.data
from networks.cls import build_classifier
from networks.det import build_detector
from networks.seg import build_segmentor
from .optimizer import build_optimizer, build_scheduler
from .data_loader import build_data_loader
from .train_base import TrainerBase
from ..datasets import build_dataset, statistics_data
from ..evaluator import Evaluator, Visualizer, model_info
from ..utils.checkpoint import load_checkpoint
from ..utils.ema import ModelEMA


class Trainer:

    def __init__(self, cfg, logger):
        # init parameters
        self.logger = logger
        self._task = cfg['task']
        self.input_size = cfg['dataset'].pop('input_size')
        self.max_epochs = cfg['scheduler']['max_epochs']

        # statistics data info
        self.logger.info('statistics_data:')
        if cfg["dataset"].get("statistics_data", None):
            self.means = cfg["dataset"].get("statistics_data", None).get("means")
            self.stds = cfg["dataset"].get("statistics_data", None).get("stds")
        else:
            self.means, self.stds = statistics_data([cfg['dataset']['train_data_path'], cfg['dataset']['val_data_path']],
                                                    cfg['dataset']['type'])
        self.logger.info(f'data means: {self.means}')
        self.logger.info(f'data stds: {self.stds}')
        # update means and stds
        self.update_dateset_info(cfg['dataset'])
        # build loader
        self.train_data_loader, self.val_data_loader = build_data_loader(self._task, loader_cfg=cfg['loader_cfg'], **cfg['dataset'])
        # self.logger.info(self.train_data_loader.dataset.class_names)
        # init model
        self.model = self.build_model(cfg, len(self.train_data_loader.dataset.class_names))
        self.optimizer = build_optimizer(self.model, cfg['optimizer'])
        self.scheduler = build_scheduler(self.optimizer, cfg['scheduler'])
        self._device, gpu_ids = self.get_device(gpu_id=cfg.get('gpu_id', 0))

        self.logger.info(f"use gpu ids {gpu_ids}")
        if isinstance(gpu_ids, list):
            if len(gpu_ids) > 0:
                # os.environ['CUDA_VISIBLE_DEVICES'] =  ",".join([str(x) for x in gpu_ids])
                # self.model = torch.nn.parallel.DistributedDataParallel(self.model)
                self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)

        self.model = self.model.to(self._device)
        self.logger.info('model info:')
        self.logger.info(model_info(self.model, self.input_size))

        self._checkpoint_dir = os.path.join(cfg['output_dir'], 'checkpoints')
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)
        model_name = cfg['model']['type']
        self.performance_dir = os.path.join(cfg['output_dir'], 'performance')
        self._evaluator = Evaluator(self._task, model_name=model_name,
                                    logger=logger,
                                    num_classes=len(self.train_data_loader.dataset.class_names),
                                    class_names=self.train_data_loader.dataset.class_names,
                                    performance_dir=self.performance_dir
                                    )
        self.is_few_shot = 'FS' in cfg['dataset']['type']
        self._vis_predictions_dir = os.path.join(self.performance_dir, 'vis_predictions')
        if self._task == "classification":
            if self.is_few_shot:
                self._validate = self._validate_fs
            else:
                self._validate = self._validate_cls
            self._visualizer = Visualizer(
                task=self._task,
                class_names=self.train_data_loader.dataset.class_names,
                input_size=self.input_size
            )
            self.vis_score_threshold = None
        elif self._task == "detection":
            self.max_number_gt = self.train_data_loader.dataset.max_number_object_per_img
            self._validate = self._validate_det

            # Visualizer
            self.vis_score_threshold = cfg["model"]["test_cfg"]["score_thr"]
            self._visualizer = Visualizer(
                task=self._task,
                class_names=self.train_data_loader.dataset.class_names,
                input_size=self.input_size,
                vis_score_threshold=self.vis_score_threshold
            )

        elif self._task == "segmentation":
            self._visualizer = Visualizer(
                task=self._task,
                class_names=self.train_data_loader.dataset.class_names,
                input_size=self.input_size
            )
            self.vis_score_threshold = None
        # save config file
        cfg["dataset"]["classes_name"] = self.train_data_loader.dataset.class_names
        cfg['dataset']["input_size"] = self.input_size
        json_path = os.path.join(cfg['output_dir'], "config.json")
        with open(json_path, 'w') as f:
            json.dump(cfg, f, indent=4)

        self.use_ema = cfg.get("use_ema", None)
        if self.use_ema:
            self.ema_model = ModelEMA(self.model, 0.9998)

    def build_model(self, cfg, num_classes):
        model_cfg = cfg.get('model')
        number_classes_model ={"classification":"backbone",
                               "detection":"bbox_head",
                               "segmentation":"decode_head"
                               }
        num_classes_cfg = model_cfg.get(number_classes_model[cfg.get("task")], None)
        if num_classes_cfg:
            num_classes_cfg.update(dict(num_classes=num_classes))
        if cfg.get('task') == "classification":
            model = build_classifier(model_cfg)
        elif cfg.get('task') == "detection":
            model = build_detector(model_cfg)
        elif cfg.get('task') == "segmentation":
            model = build_segmentor(model_cfg)
        else:
            raise TypeError(f"task must be classification, detection or segmentation, but go {cfg['task']}")

        checkpoint_path = cfg.get("pretrained", "")
        if os.path.exists(checkpoint_path):
            checkpoint_path = os.path.expanduser(checkpoint_path)
            load_checkpoint(model, checkpoint_path, map_location='cpu', strict=True)
            self.logger.info(f"load checkpoint from {checkpoint_path}")
        return model

    def train(self):
        for cur_epoch in range(1, self.max_epochs):
            self._cur_epoch_losses = []
            self._cur_epoch = cur_epoch
            cur_lr = self.scheduler.get_lr()[0]

            self.logger.info('-'*25 + 'epoch: %d/%d'%(cur_epoch, self.max_epochs) + '-'*25)
            epoch_st = time.time()
            self.train_one_epoch()
            epoch_time = time.time() -epoch_st
            self.scheduler.step()
            mean_loss = np.mean(self._cur_epoch_losses)
            epoch_status = ("Epoch %6d \tTime %5.2f \t[loss]:\taverage loss:%7.8f \tdecreases:%7.4f"
                            %(self._cur_epoch, epoch_time, mean_loss,
                              self._cur_epoch_losses[0]-self._cur_epoch_losses[-1]))
            self.logger.info(epoch_status)

            if self.use_ema:
                self.ema_model.update_attr(self.model)

            # validation
            self._val_feature = None
            if self._task == "classification":
                self._validate_cls()
            elif self._task == "detection":
                self._validate_det(self.max_number_gt)
            elif self._task == "segmentation":
                self._validate_seg()

            self._evaluator.record_cur_epoch(
                learning_rate=cur_lr,
                avg_loss=mean_loss,
                y_prob=self._val_pred,
                y_true=self._val_true,
                y_feature=self._val_feature,
                confidence_threshold=self.vis_score_threshold
            )

            # save checkpoint
            state_dict = self.ema_model.ema.state_dict() if self.use_ema else self.model.state_dict()
            if self._evaluator.is_train_best_epoch():
                print("Saving checkpoint of minimum training loss.")
                state = {
                    'arch': type(self.model).__name__,
                    'epoch': self._cur_epoch,
                    'state_dict': state_dict,
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(state, "%s/train_best.pth"%(self._checkpoint_dir))
            if self._evaluator.is_val_best_epoch():
                print("Saving checkpoint of best validation metrics.")
                state = {
                    'arch': type(self.model).__name__,
                    'epoch': self._cur_epoch,
                    'state_dict': state_dict,
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(state, "%s/val_best.pth"%(self._checkpoint_dir))

                # visualize
                if self._visualizer is not None:
                    self._visualizer.visualize(
                        self._val_img_paths, self._val_pred,
                        vis_predictions_dir=self._vis_predictions_dir
                    )
        best_metric = self._evaluator.get_best_metric()
        self.logger.info(f"the best metric is {best_metric}")
        self.logger.info('epoch end ')

    def _validate_fs(self):
        self.model.eval()
        for step, (img_batch, label_batch) in enumerate(self.val_data_loader):
            if not isinstance(img_batch, torch.Tensor):
                img_batch = torch.from_numpy(img_batch).to(self._device, dtype=torch.float32)
            pred, feature = self.model(img_batch)
            if step ==0:
                pred_array = pred.detach().cpu().numpy()
                feature_array = feature.detach().cpu().numpy()
                gt_array = label_batch['gt_labels']
            else:
                pred_array = np.concatenate([pred_array, pred.detach().cpu().numpy()], axis=0)
                feature_array = np.concatenate([feature_array, feature.detach().cpu().numpy()], axis=0)
                gt_array = np.concatenate([gt_array, label_batch['gt_labels']], axis=0)

        self._val_pred = pred_array
        self._val_feature = feature_array
        self._val_true = gt_array

    def _validate_cls(self):
        self.model.eval()
        evalmodel = self.ema_model.ema if self.use_ema else self.model
        evalmodel.eval()
        img_paths = []
        for step, (img_batch, label_batch) in enumerate(self.val_data_loader):
            img_paths += [img_path for img_path in label_batch['image_path']]
            if not isinstance(img_batch, torch.Tensor):
                img_batch = torch.from_numpy(img_batch).to(self._device, dtype=torch.float32)
            pred = evalmodel(img_batch)
            if step ==0:
                pred_array = pred.detach().cpu().numpy()
                gt_array = label_batch['gt_labels']
            else:
                pred_array = np.concatenate([pred_array, pred.detach().cpu().numpy()], axis=0)
                gt_array = np.concatenate([gt_array, label_batch['gt_labels']], axis=0)

        self._val_pred = pred_array
        self._val_true = gt_array
        self._val_img_paths = img_paths

    def _validate_det(self, max_number_gt):
        evalmodel = self.ema_model.ema if self.use_ema else self.model
        evalmodel.eval()
        img_paths = []
        for step, (img_batch, label_batch) in enumerate(self.val_data_loader):
            img_paths += label_batch['image_path']
            if not isinstance(img_batch, torch.Tensor):
                img_batch = torch.from_numpy(img_batch).to(self._device, dtype=torch.float32)
            pred = evalmodel(img_batch)

            batch_label = []
            for boxes, index in zip(label_batch['bboxes'], label_batch['label_index']):
                index = np.expand_dims(index, axis=-1)
                if len(boxes) ==0 :
                    boxes = np.empty(shape=(0, 4))
                gt_label = np.concatenate([boxes, index], axis=-1)
                if gt_label.shape[0] < max_number_gt:
                    padding = np.zeros([max_number_gt - gt_label.shape[0], gt_label.shape[1]])
                    padding[...,-1] =-1
                    label = np.concatenate([gt_label, padding], axis=0)
                batch_label.append(label)

            if step ==0:
                pred_array = pred.detach().cpu().numpy()
                gt_array = np.array(batch_label)
            else:
                pred_array = np.concatenate([pred_array, pred.detach().cpu().numpy()], axis=0)
                gt_array = np.concatenate([gt_array, np.array(batch_label)], axis=0)

        self._val_pred = pred_array
        self._val_true = gt_array
        self._val_img_paths = img_paths

    def _validate_seg(self):
        evalmodel = self.ema_model.ema if self.use_ema else self.model
        evalmodel.eval()
        img_paths = []
        for step, (img_batch, label_batch) in enumerate(self.val_data_loader):
            img_paths += [img_path for img_path in label_batch['image_path']]
            if not isinstance(img_batch, torch.Tensor):
                img_batch = torch.from_numpy(img_batch).to(self._device, dtype=torch.float32)
            pred = evalmodel(img_batch)
            if step == 0:
                pred_array = pred.detach().cpu().numpy()
                pred_array = np.transpose(pred_array, [0, 2, 3, 1])
                gt_array = label_batch['gt_masks']
            else:
                pre_array1 = np.transpose(pred.detach().cpu().numpy(), [0, 2, 3, 1])
                pred_array = np.concatenate([pred_array, pre_array1], axis=0)
                gt_array = np.concatenate([gt_array, label_batch['gt_masks']], axis=0)

        self._val_pred = pred_array
        self._val_true = gt_array
        self._val_img_paths = img_paths

    def train_one_epoch(self):

        self.model.train()
        for step, (img_batch, label_batch) in enumerate(self.train_data_loader):

            # debug
            # for img, bboxes in zip(img_batch, label_batch['bboxes']):
            #     img = np.transpose(img, [1, 2, 0]).copy()
            #     img = img*np.array(self.stds) + np.array(self.means)
            #     img = img.astype(np.uint8)
            #     for bboxe in bboxes:
            #         x1, y1, x2, y2 = bboxe
            #         cv2.rectangle(img,(int(x1), int(y1)), (int(x2), int(y2)), (255,255,0))
            #     cv2.imshow('result', img)
            #     cv2.waitKey()

            if not isinstance(img_batch, torch.Tensor):
                img_batch = torch.from_numpy(img_batch).to(self._device, dtype=torch.float32)
            if img_batch.device != self._device:
                img_batch = img_batch.to(self._device)
            st = time.time()
            losses = self.model(img_batch, return_metrics=True,
                                ground_truth=label_batch)
            step_time = time.time() - st
            self.model.zero_grad()
            self.optimizer.zero_grad()
            cur_loss = losses['loss']
            cur_loss.backward()
            self.optimizer.step()
            if self.use_ema:
                self.ema_model.update(self.model)
            cur_lr = self.scheduler.get_lr()[0]
            self._cur_epoch_losses.append(cur_loss.detach().cpu().numpy())
            if step % 10 == 0:
                step_status = '=> Step %6d \tTime %5.2f \tLr %2.6f \t[Loss]:' %(
                    step, step_time, cur_lr)
                for key in losses:
                    step_status += ' %s: %7.4f' %(key, losses[key].detach().cpu().numpy())
                self.logger.info(step_status)

    def get_device(self, gpu_id):

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            self.logger.info("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")

        if isinstance(gpu_id, int):
            if gpu_id > gpu_count:
                print("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                      "on this machine.".format(gpu_id, gpu_count))
                gpu_id = gpu_count
            device = torch.device('cuda:%d'%(gpu_id) if torch.cuda.is_available() else 'cpu')
            return device, gpu_id

        if isinstance(gpu_id, list):
            if len(gpu_id) > 1:
                # torch.distributed.init_process_group(backend="nccl", world_size=1,init_method='tcp://localhost:23456', rank=0)
                pass
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        return device, gpu_id

    def update_dateset_info(self, pipelines):
        if isinstance(pipelines, list):
            for augmentation in pipelines:
                if isinstance(augmentation, dict):
                    if "Normalize" == augmentation.get('type', None):
                        augmentation.update(dict(mean=self.means, std=self.stds))
                    elif "Resize" == augmentation.get('type', None) and self.input_size is not None:
                        augmentation.update(dict(width=self.input_size[0], height=self.input_size[1]))
                    elif "RandomCrop" == augmentation.get('type', None) and self.input_size is not None:
                        augmentation.update(dict(width=self.input_size[0], height=self.input_size[1]))
        elif isinstance(pipelines, dict):
            for k, v in pipelines.items():
                self.update_dateset_info(v)
        else:
            pass


class SimpleTrainer(TrainerBase):

    def __init__(self, cfg, logger):
        super(SimpleTrainer, self).__init__(cfg['scheduler']['max_epochs'], logger)
        self.model = self.build_model(cfg)
        self.optimizer = build_optimizer(self.model, cfg['optimizer'])
        self.scheduler = build_scheduler(self.optimizer, cfg['scheduler'])
        self._device, gpu_ids = self.get_device(num_gpus=cfg['num_gpus'])
        self.model = self.model.to(self._device)
        self.train_data_loader, self.val_data_loader = self.build_train_loader(cfg['dataset'], cfg['loader_cfg'])

    def build_model(self, cfg):
        model_cfg = cfg.get('model')
        head_cfg = model_cfg.get('head', None)
        if head_cfg:
            head_cfg.update(dict(num_classes=cfg['dataset']['num_classes']))

        if cfg.get('task') == "classification":
            model = build_classifier(model_cfg)
        elif cfg.get('task') == "detection":
            model = build_detector(model_cfg)
        elif cfg.get('task') == "segmentation":
            model = build_segmentor(model_cfg)
        else:
            raise TypeError(f"task must be classification, detection or segmentation, but go {cfg['task']}")

        checkpoint_path = cfg.get("pretrained", "")
        if os.path.exists(checkpoint_path):
            checkpoint_path = os.path.expanduser(checkpoint_path)
            load_checkpoint(model, checkpoint_path, map_location='cpu', strict=True)
        return model

    def get_device(self, num_gpus):
        gpu_count = torch.cuda.device_count()
        if num_gpus > 0 and gpu_count == 0:
            print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            num_gpus = 0
        if num_gpus > gpu_count:
            print("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                  "on this machine.".format(num_gpus, gpu_count))
            num_gpus = gpu_count
        device = torch.device('cuda:0' if num_gpus > 0 else 'cpu')
        gpu_ids = list(range(num_gpus))
        return device, gpu_ids

    def train_one_epoch(self):
        self.model.train()

        for step, (img_batch, label_batch) in enumerate(self.train_data_loader):

            if not isinstance(img_batch, torch.Tensor):
                img_batch = torch.from_numpy(img_batch).to(self._device, dtype=torch.float32)

            st = time.time()
            losses = self.model(img_batch, return_metrics=True,
                                ground_truth=label_batch)
            step_time = time.time() - st
            self.model.zero_grad()
            self.optimizer.zero_grad()
            cur_loss = losses['losses']
            cur_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            cur_lr = self.scheduler.get_lr()[0]
            self._cur_epoch_losses.append(cur_loss.detach().cpu().numpy())
            if step % 10 == 0:
                step_status = '=> Step %6d \tTime %5.2f \tLr %2.5f \t[Loss]:' % (
                    step, step_time, cur_lr)
                for key in losses:
                    step_status += ' %s: %7.4f' % (key, losses[key].detach().cpu().numpy())
                self.logger.info(step_status)

        pass

    def build_train_loader(self, cfg, loader_cfg):
        dataset_type = cfg.pop('type')
        train_data_path = cfg.pop('train_data_path')
        val_data_path = cfg.pop('val_data_path')
        train_cfg_dict = dict(type=dataset_type, data_path=train_data_path, **cfg)
        val_cfg_dict = dict(type=dataset_type, data_path=val_data_path, **cfg)

        train_dataset = build_dataset(train_cfg_dict)
        val_dataset = build_dataset(val_cfg_dict)

        train_data_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=self.dataset_collate, **loader_cfg)
        val_data_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=self.dataset_collate, **loader_cfg)

        return train_data_loader, val_data_loader

    def dataset_collate(self, batch):
        images = []
        labels = []
        for img, label in batch:
            images.append(img)
            labels.append(label)

        images1 = np.array(images)[:, 0, :, :, :]
        images2 = np.array(images)[:, 1, :, :, :]
        images3 = np.array(images)[:, 2, :, :, :]
        images = np.concatenate([images1, images2, images3], 0)

        labels1 = np.array(labels)[:, 0]
        labels2 = np.array(labels)[:, 1]
        labels3 = np.array(labels)[:, 2]
        labels = np.concatenate([labels1, labels2, labels3], 0)
        gt = dict(gt_labels=labels)
        return images, gt

