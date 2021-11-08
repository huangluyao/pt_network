import torch
from ..builder import MODELS, build_module
from .base_gan import Base_GAN
from ..common import set_requires_grad

@MODELS.register_module()
class TwoStageInpaintor(Base_GAN):

    def __init__(self,
                 encdec,
                 disc=None,
                 loss_gan=None,
                 loss_l1_hole=None,
                 loss_l1_valid=None,
                 train_cfg=None,
                 test_cfg=None,
                 stage1_loss_type=('loss_l1_hole',),
                 stage2_loss_type=('loss_l1_hole', 'loss_gan'),
                 input_with_ones=True,
                 disc_input_with_mask=False,
                 **kwargs
                 ):
        super().__init__()
        self.input_with_ones = input_with_ones
        self.generator = build_module(encdec)
        self.disc_input_with_mask = disc_input_with_mask

        # build loss modules
        self.disc = build_module(disc)
        self.loss_gan = build_module(loss_gan)
        self.loss_l1_valid = build_module(loss_l1_valid)
        self.loss_l1_hole = build_module(loss_l1_hole)

        self.stage1_loss_type = stage1_loss_type
        self.stage2_loss_type = stage2_loss_type

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    def train_step(self, data_batch, optimizer, ddp_reducer=None):

        log_vars = {}
        gt_img = data_batch['image']           # 原图
        mask = data_batch['mask']               # 掩码图
        masked_img = data_batch['masked_img']   # 原图+掩码图


        # get common output from encdec
        if self.input_with_ones:
            tmp_ones = torch.ones_like(mask)
            input_x = torch.cat([masked_img, tmp_ones, mask], dim=1)
        else:
            input_x = torch.cat([masked_img, mask], dim=1)

        stage1_fake_res, stage2_fake_res = self.generator(input_x)

        stage1_fake_img = masked_img * (1. - mask) + stage1_fake_res * mask
        stage2_fake_img = masked_img * (1. - mask) + stage2_fake_res * mask

        # 训练判别器
        set_requires_grad(self.disc, True)
        if self.disc_input_with_mask:
            disc_input_x = torch.cat([stage2_fake_img.detach(), mask],
                                     dim=1)
        else:
            disc_input_x = stage2_fake_img.detach()

        disc_losses = self.forward_train_disc(disc_input_x, False, is_disc=True)
        loss_disc, log_vars_d = self.parse_losses(disc_losses)

        log_vars.update(log_vars_d)
        optimizer['disc'].zero_grad()
        loss_disc.backward()

        if self.disc_input_with_mask:
            disc_input_x = torch.cat([gt_img, mask], dim=1)
        else:
            disc_input_x = gt_img

        disc_losses = self.forward_train_disc(disc_input_x, True, is_disc=True)
        loss_disc, log_vars_d = self.parse_losses(disc_losses)
        log_vars.update(log_vars_d)
        loss_disc.backward()

        optimizer['disc'].step()

        # 生成器训练
        stage1_results = dict(
            fake_res=stage1_fake_res, fake_img=stage1_fake_img)
        stage2_results = dict(
            fake_res=stage2_fake_res, fake_img=stage2_fake_img)
        set_requires_grad(self.disc, False)

        results, two_stage_losses = self.two_stage_loss(
            stage1_results, stage2_results, data_batch)

        loss_two_stage, log_vars_two_stage = self.parse_losses(
            two_stage_losses)

        log_vars.update(log_vars_two_stage)
        optimizer['generator'].zero_grad()
        loss_two_stage.backward()
        optimizer['generator'].step()

        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['image'].data),
            results=results)

        return outputs


    def two_stage_loss(self, stage1_data, stage2_data, data_batch):
        """Calculate two-stage loss.

        Args:
            stage1_data (dict): Contain stage1 results.
            stage2_data (dict): Contain stage2 results.
            data_batch (dict): Contain data needed to calculate loss.

        Returns:
            dict: Contain losses with name.
        """
        gt = data_batch['image']
        mask = data_batch['mask']
        masked_img = data_batch['masked_img']
        loss = dict()
        results = dict(
            image=gt.cpu(), mask=mask.cpu(), masked_img=masked_img.cpu())

        # 计算stage1的损失函数
        if self.stage1_loss_type is not None:
            fake_res = stage1_data['fake_res']
            fake_img = stage1_data['fake_img']
            for type_key in self.stage1_loss_type:
                tmp_loss = self.calculate_loss_with_type(
                    type_key, fake_res, fake_img, gt, mask, prefix='stage1_')
                loss.update(tmp_loss)

        results.update(
            dict(
                stage1_fake_res=stage1_data['fake_res'].cpu(),
                stage1_fake_img=stage1_data['fake_img'].cpu()))

        # 计算阶段二的损失函数
        if self.stage2_loss_type is not None:
            fake_res = stage2_data['fake_res']
            fake_img = stage2_data['fake_img']
            for type_key in self.stage2_loss_type:
                tmp_loss = self.calculate_loss_with_type(
                    type_key, fake_res, fake_img, gt, mask, prefix='stage2_')
                loss.update(tmp_loss)
        results.update(
            dict(
                stage2_fake_res=stage2_data['fake_res'].cpu(),
                stage2_fake_img=stage2_data['fake_img'].cpu()))

        return results, loss


    def forward_train_disc(self, disc_input_x, is_real, is_disc):
        pred = self.disc(disc_input_x)
        loss_ = self.loss_gan(pred, is_real, is_disc)
        loss = dict(real_loss=loss_) if is_real else dict(fake_loss=loss_)
        return loss


    def calculate_loss_with_type(self,
                                 loss_type,
                                 fake_res,
                                 fake_img,
                                 gt,
                                 mask,
                                 prefix='stage1_'
                                 ):
        loss_dict = dict()

        if 'l1' in loss_type:
            weight = 1. - mask if 'valid' in loss_type else mask
            loss_l1 = getattr(self, loss_type)(fake_res, gt, weight=weight)
            loss_dict[prefix + loss_type] = loss_l1
        elif loss_type == 'loss_gan':
            if self.disc_input_with_mask:
                disc_input_x = torch.cat([fake_img, mask], dim=1)
            else:
                disc_input_x = fake_img
            g_fake_pred = self.disc(disc_input_x)
            loss_g_fake = self.loss_gan(g_fake_pred, True, is_disc=False)
            loss_dict[prefix + 'loss_g_fake'] = loss_g_fake

        return loss_dict

    def forward_test(self,
                     data,
                     **kwargs):

        masked_img = data["masked_img"]
        mask = data["mask"]

        if self.input_with_ones:
            tmp_ones = torch.ones_like(mask)
            input_x = torch.cat([masked_img, tmp_ones, mask], dim=1)
        else:
            input_x = torch.cat([masked_img, mask], dim=1)


        stage1_fake_res, stage2_fake_res = self.generator(input_x)
        fake_img = stage2_fake_res * mask + masked_img * (1. - mask)

        output = dict()
        output['stage1_fake_res'] = stage1_fake_res
        output['stage2_fake_res'] = stage2_fake_res
        output['fake_img'] = fake_img
        return output
