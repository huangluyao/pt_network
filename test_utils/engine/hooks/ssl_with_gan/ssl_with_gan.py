import torch
import torch.nn.functional as F
import numpy as np
from .discrimitor import GAN_Discriminator
from .remain_dataset import get_remain_loader
from test_utils.engine.hooks.hook import HOOKS, Hook
from test_utils.engine.optimizer import build_optimizer
from networks.seg.models import build_loss


@HOOKS.registry()
class SSLWithGANHook(Hook):
    """
    携带gan的半监督插件，只适用于分割网络
    """
    def __init__(self,
                 train_gt_loader,
                 remain_image_folder,
                 loader_cfg,
                 threshold_st=0.6,
                 augmentations=None,
                 fm_loss_weight=0.1,
                 optimizer=dict(type="Adam", lr=1e-4, betas=(0.9,0.99)),
                 seg_loss=dict(type="CrossEntropyLoss"),
                 pretrained=None):
        super(SSLWithGANHook, self).__init__()
        self.pretrained = pretrained
        self.optimizer = optimizer
        self.base_lr = self.optimizer["lr"]
        self.train_gt_loader = train_gt_loader
        self.train_gt_loader_iter = iter(self.train_gt_loader)
        self.remain_loader = get_remain_loader(remain_image_folder, augmentations, loader_cfg)
        self.remain_loader_iter = iter(self.remain_loader)
        self.threshold_st = threshold_st
        self.seg_loss = build_loss(seg_loss)
        self.fm_loss_weight = fm_loss_weight

    def before_train(self, runner):
        num_classes = len(runner.data_loader.dataset.class_names)
        self.model_D = GAN_Discriminator(num_classes)

        if self.pretrained is not None:
            self.model_D.load_state_dict(torch.load(self.pretrained))

        self.model_D = self.model_D.to(runner.device)
        self.model_D.train()
        self.optimizer = build_optimizer(self.model_D, self.optimizer)
        self.optimizer.zero_grad()


    def after_train_iter(self, runner):

        self.optimizer.zero_grad()
        runner.optimizer.zero_grad()
        self._adjust_learning_rate_D(runner)

        for param in self.model_D.parameters():
            param.requires_grad = False

        # 获取无标签的图像数据
        try:
            image_batch = next(self.remain_loader_iter).to(runner.device)
        except:
            self.remain_loader_iter = iter(self.remain_loader)
            image_batch = next(self.remain_loader_iter).to(runner.device)
        # 对无标签的图像进行网络推理
        infer_result = runner.model(image_batch)
        pred_remain = infer_result["preds"]
        # 组合数据并放入鉴别器判断
        image_batch = (image_batch-torch.min(image_batch))/(torch.max(image_batch)- torch.min(image_batch))  # image_batch的数值设置为大于0的数
        pred_cat = torch.cat((pred_remain, image_batch), dim=1)
        D_out, D_out_y_pred = self.model_D(pred_cat)
        # 寻找比较好的标签并参与训练
        pred_sel, labels_sel, count = find_good_maps(D_out, infer_result["seg_logit"], self.threshold_st)
        if count > 0 and runner.iter >0:
            loss_st = self.seg_loss(pred_sel, labels_sel)
        else:
            loss_st = 0.0

        # 获取真实存在的数据图像
        try:
            img_gt_batch, label_gt_batch = next(self.train_gt_loader_iter)
        except:
            self.train_gt_loader_iter = iter(self.train_gt_loader)
            img_gt_batch, label_gt_batch = next(self.train_gt_loader_iter)

        label_gt_batch = label_gt_batch["gt_masks"]
        if not isinstance(img_gt_batch, torch.Tensor):
            img_gt_batch = torch.from_numpy(img_gt_batch).to(runner.device, dtype=torch.float32)
        if img_gt_batch.device != runner.device:
            img_gt_batch = img_gt_batch.to(runner.device)
        label_gt_batch = one_hot(label_gt_batch, num_classes=len(runner.data_loader.dataset.class_names)).to(runner.device)
        img_gt_batch = (img_gt_batch - torch.min(img_gt_batch))/(torch.max(img_gt_batch)-torch.min(img_gt_batch))

        # 整合数据送入鉴别器判别
        gt_cat = torch.cat([label_gt_batch, img_gt_batch], dim=1)
        D_gt_out, D_gt_out_pred = self.model_D(gt_cat)

        # 对两个结果产生的分布求 L1距离
        runner.outputs["loss_fm"] = torch.mean(torch.abs(torch.mean(D_gt_out_pred, 0) - torch.mean(D_out_y_pred, 0)))
        runner.outputs["loss"] += runner.outputs["loss_fm"] * self.fm_loss_weight
        if count > 0 and runner.iter >0:
            runner.outputs["loss_st"] = loss_st
            runner.outputs["loss"] += runner.outputs["loss_st"]

        # 对分割网络反向求导
        runner.outputs["loss"].backward()

        # 训练鉴别器
        for param in self.model_D.parameters():
            param.requires_grad = True

        pred_cat = pred_cat.detach()
        D_out_z, _ = self.model_D(pred_cat)
        y_fake_ = torch.zeros_like(D_out_z)
        loss_D_fake = F.binary_cross_entropy(D_out_z, y_fake_, reduction="mean")

        # train with gt
        D_out_z_gt, _ = self.model_D(gt_cat)
        y_real_ = torch.ones_like(D_out_z_gt)
        loss_D_real = F.binary_cross_entropy(D_out_z_gt, y_real_, reduction="mean")
        loss_D = (loss_D_fake + loss_D_real) / 2.0

        loss_D.backward()

        runner.optimizer.step()
        self.optimizer.step()

    def _get_gt_D_result(self, runner):
        # 获取真实存在的数据图像
        try:
            img_gt_batch, label_gt_batch = next(self.train_gt_loader_iter)
        except:
            self.train_gt_loader_iter = iter(self.train_gt_loader)
            img_gt_batch, label_gt_batch = next(self.train_gt_loader_iter)

        label_gt_batch = label_gt_batch["gt_masks"]
        if not isinstance(img_gt_batch, torch.Tensor):
            img_gt_batch = torch.from_numpy(img_gt_batch).to(runner.device, dtype=torch.float32)
        if img_gt_batch.device != runner.device:
            img_gt_batch = img_gt_batch.to(runner.device)
        label_gt_batch = one_hot(label_gt_batch, num_classes=len(runner.data_loader.dataset.class_names)).to(runner.device)
        img_gt_batch = (img_gt_batch - torch.min(img_gt_batch))/(torch.max(img_gt_batch)-torch.min(img_gt_batch))
        gt_cat = torch.cat([label_gt_batch, img_gt_batch], dim=1)
        # 送入鉴别器判别
        D_gt_out, D_gt_out_pred = self.model_D(gt_cat)
        return D_gt_out, D_gt_out_pred

    """
    def _get_fake_D_result(self, runner, train_D):
        # 获取无标签的图像数据
        try:
            image_batch = next(self.remain_loader_iter).to(runner.device)
        except:
            self.remain_loader_iter = iter(self.remain_loader)
            image_batch = next(self.remain_loader_iter).to(runner.device)
        # 对无标签的图像进行网络推理
        pred_remain = runner.model(image_batch)
        # 组合数据并放入鉴别器判断
        image_batch = (image_batch-torch.min(image_batch))/(torch.max(image_batch)- torch.min(image_batch))  # image_batch的数值设置为大于0的数
        pred_cat = torch.cat((F.softmax(pred_remain, dim=1), image_batch), dim=1)
        if train_D:
            pred_cat = pred_cat.detach()
        D_out, D_out_y_pred = self.model_D(pred_cat)

        return D_out, D_out_y_pred, pred_remain



    def before_train_iter(self, runner):
        self.optimizer.zero_grad()
        self._adjust_learning_rate_D(runner)
        # 训练鉴别器
        for param in self.model_D.parameters():
            param.requires_grad = True

        # 对于 网络推理的结果做训练
        D_out_z = self._get_fake_D_result(runner, train_D=True)[0]
        y_fake_ = torch.zeros_like(D_out_z)
        loss_D_fake = F.binary_cross_entropy(D_out_z, y_fake_, reduction="mean")

        # 对于 真实的分布做训练
        D_out_z_gt = self._get_gt_D_result(runner)[0]
        y_real_ = torch.ones_like(D_out_z_gt)
        loss_D_real = F.binary_cross_entropy(D_out_z_gt, y_real_, reduction="mean")
        loss_D = (loss_D_fake + loss_D_real) / 2.0
        loss_D.backward()
        self.optimizer.step()
        pass
    """

    def _adjust_learning_rate_D(self, runner):
        lr = self.base_lr * ((1 - float(runner.epoch) / runner.max_epochs) ** (0.9))
        self.optimizer.param_groups[0]['lr'] = lr
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]['lr'] = lr * 10


def compute_argmax_map(output):
    output = output.detach().cpu().numpy()
    output = output.transpose((1,2,0))
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
    output = torch.from_numpy(output).float()
    return output


def find_good_maps(D_outs, pred_all, threshold_st):
    count = 0
    for i in range(D_outs.size(0)):
        if D_outs[i] > threshold_st:
            count +=1

    if count > 0:
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3))
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3))
        num_sel = 0
        for j in range(D_outs.size(0)):
            if D_outs[j] > threshold_st:
                pred_sel[num_sel] = pred_all[j]
                label_sel[num_sel] = compute_argmax_map(pred_all[j])
                num_sel +=1
        return  pred_sel.cuda(), label_sel.cuda(), count
    else:
        return 0, 0, count


def one_hot(label, num_classes):
    one_hot = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)