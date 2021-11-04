import os

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from test_utils.evaluator.draw_plot import static_bn
from test_utils.utils.pruned_utils import get_bn_weights, get_model_list, pruned_model_load_state
from test_utils.utils.utils import build_model
from .hook import Hook, HOOKS


@HOOKS.registry()
class PrunedHook(Hook):

    def __init__(self,
                 interval,
                 start_finetune_epoch,
                 percent=None,
                 ):
        super(PrunedHook, self).__init__()
        self.percent = percent
        self.interval = interval
        self.start_finetune_epoch = start_finetune_epoch
        # self.current_times = 1

    def before_train(self, runner):          # 统计剪枝信息

        assert self.start_finetune_epoch < runner.max_epochs, f"start fine tune epoch must less than max_epoch:{runner.max_epochs}"

        # 获取要剪枝的BN层
        model_list, self.ignore_bn_list = get_model_list(runner.model)

        model_list = {k: v for k, v in model_list.items() if k not in self.ignore_bn_list}

        # 统计并计算bn的权重
        size_list = [idx.weight.data.shape[0] for idx in model_list.values()]
        bn_weights = torch.zeros(sum(size_list))
        index = 0
        for i, idx in enumerate(model_list.values()):
            size = size_list[i]
            bn_weights[index:(index + size)] = idx.weight.data.abs().clone()
            index += size

        sorted_bn = torch.sort(bn_weights)[0]
        # print("model_list:",model_list)
        # print("bn_weights:",bn_weights)
        # 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
        highest_thre = []
        for bnlayer in model_list.values():
            highest_thre.append(bnlayer.weight.data.abs().max().item())
        # print("highest_thre:",highest_thre)
        highest_thre = min(highest_thre)
        # 找到highest_thre对应的下标对应的百分比
        len_weights = len(bn_weights)
        percent_limit = (sorted_bn == highest_thre).nonzero()[0, 0].item() / len_weights

        bn_numpy = bn_weights.numpy()

        percent_suggested = (sorted_bn > 5e-5).nonzero()[0, 0].item() / len_weights

        runner.logger.info('-' * 25 + 'pruned info' + '-' * 25)
        runner.logger.info(f'The Suggested prune ratio is {percent_limit:.3f}, but you can set higher.')

        if self.percent is None:
            # 给出相关裁切建议
            runner.logger.info(f'The recommended pruning ratio is {percent_suggested:.3f} set pruned percent at {percent_suggested:.3f}')
            self.percent = percent_suggested
        else:
            runner.logger.info(f'The recommended pruning ratio is {percent_suggested:.3f} set pruned percent at {self.percent:.3f}')

        assert self.percent < percent_limit, f"Prune ratio should less than {percent_limit:.3f}, otherwise it may cause error!!!"

        save_weights_path = os.path.join(runner.output_dir, f"bn_weights.png")
        static_bn(bn_numpy, save_weights_path)

        self.pruned_times = int(self.start_finetune_epoch / self.interval)

        each_percent = self.percent / self.pruned_times
        self.each_index = int(each_percent * len_weights)

        runner.logger.info(f'The pruned {self.pruned_times} times, {each_percent*100}% for each pruned, {self.percent*100}% for total')
        runner.logger.info(f"Fine tune {self.interval} epoch after pruned")
        runner.logger.info(f"The final fine-tuning starts with the {self.start_finetune_epoch}th epoch")

    def before_epoch(self, runner):
        # 对网络进行剪枝

        if runner._epoch % self.interval == 0 and runner._epoch < self.start_finetune_epoch:
            # percent = self.each_percent * self.current_times
            bn_weights, bnb_weights = get_bn_weights(runner.model, self.ignore_bn_list)
            sorted_bn = torch.sort(bn_weights)[0]
            # thre_index = int(len(sorted_bn) * percent)
            thre = sorted_bn[self.each_index]
            # 剪枝
            maskbndict = {}
            runner.logger.info("=" * 34 + "pruned info" + "=" * 34)
            for bnname, bnlayer in runner.model.named_modules():
                if isinstance(bnlayer, nn.BatchNorm2d):
                    bn_module = bnlayer
                    mask = bn_module.weight.data.abs().ge(thre).float()
                    if bnname in self.ignore_bn_list:
                        mask = torch.ones(bnlayer.weight.data.size()).cuda()
                    maskbndict[bnname] = mask
                    # print("mask:",mask)
                    bn_module.weight.data.mul_(mask)
                    bn_module.bias.data.mul_(mask)
                    # print("bn_module:", bn_module.bias)
                    runner.logger.info(
                        f"|\t{bnname:<35}{'|':<10}{bn_module.weight.data.size()[0]:<10}{'|':<10}{int(mask.sum()):<10}|")
                elif "Head" in bnlayer._get_name():
                    break
            runner.logger.info("=" * 79)

            # 剪枝结束
            self.mask_bn_dict = maskbndict

            cfg = self.update_cfg(runner.cfg)
            pruned_model = build_model(cfg, len(cfg.class_names), runner.logger).to(runner.device)
            missing = pruned_model_load_state(self.mask_bn_dict, pruned_model, runner.model)
            runner.model = pruned_model

            if len(missing) > 0:
                runner.logger.info("warning:")
                runner.logger.info("missing key : %s" % ",".join(missing))
                runner.logger.info("=" * 79)


    def after_epoch(self, runner):

        #
        # cfg = self.update_cfg(runner.cfg)
        # pruned_model = build_model(cfg, len(cfg.class_names), runner.logger).to(runner.device)
        # pruned_model_load_state(self.mask_bn_dict, pruned_model, runner.model)
        # runner.pruned_model = pruned_model
        pass


    def after_train(self, runner):
        # cfg = self.update_cfg(runner.cfg)
        # pruned_model = build_model(cfg, len(cfg.class_names), runner.logger).to(runner.device)
        # pruned_model_load_state(self.mask_bn_dict, pruned_model, runner.model)
        pass

    def update_cfg(self, cfg):
        new_cfg = deepcopy(cfg)
        model = new_cfg.model
        model.type = "Pruned"+model.type
        model.backbone.type = "Pruned"+model.backbone.type
        model.backbone.mask_bn_dict = self.mask_bn_dict
        if hasattr(model, "neck"):
            model.neck.type = "Pruned"+model.neck.type
            model.neck.mask_bn_dict = self.mask_bn_dict
        new_cfg.pretrained = ""
        return new_cfg