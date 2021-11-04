import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
from copy import deepcopy
from .base_gan import  Base_GAN
from ..builder import build_module, MODELS
from ..common import set_requires_grad
from ...specific.optimizer import build_dict_optimizer


@MODELS.register_module('StyleGANV1')
@MODELS.register_module('PGGAN')
@MODELS.register_module()
class ProgressiveGrowingGAN(Base_GAN):

    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 shift_loss=None,
                 gp_loss=None,
                 gen_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None
                 ):
        super(ProgressiveGrowingGAN, self).__init__()
        self.generator = build_module(generator)
        # support no discriminator in testing
        if discriminator is not None:
            self.discriminator = build_module(discriminator)
        else:
            self.discriminator = None

        # support no gan_loss in testing
        if gan_loss is not None:
            self.gan_loss = build_module(gan_loss)
        else:
            self.gan_loss = None

        if shift_loss is not None:
            self.shift_loss = build_module(shift_loss)

        if gp_loss is not None:
            self.gp_loss = build_module(gp_loss)

        if gen_auxiliary_loss:
            self.gen_auxiliary_losses = build_module(gen_auxiliary_loss)
            if not isinstance(self.gen_auxiliary_losses, nn.ModuleList):
                self.gen_auxiliary_losses = nn.ModuleList(
                    [self.gen_auxiliary_losses])
        else:
            self.gen_auxiliary_losses = None


        # register necessary training status
        self.register_buffer('shown_nkimg', torch.tensor(0.))
        self.register_buffer('_curr_transition_weight', torch.tensor(1.))

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        self._parse_test_cfg()

        # this buffer is used to resume model easily
        self.register_buffer(
            '_next_scale_int',
            torch.tensor(self.scales[0][0], dtype=torch.int32))

        self.register_buffer(
            '_curr_scale_int',
            torch.tensor(self.scales[-1][0], dtype=torch.int32))

    def _parse_train_cfg(self):
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)
        self.use_ema = self.train_cfg.get('use_ema', False)
        if self.use_ema:
            # use deepcopy to guarantee the consistency
            self.generator_ema = deepcopy(self.generator)

        # setup interpolation operation at the beginning of training iter
        interp_real_cfg = deepcopy(self.train_cfg.get('interp_real', None))
        if interp_real_cfg is None:
            interp_real_cfg = dict(mode='bilinear', align_corners=True)
        self.interp_real_to = partial(F.interpolate, **interp_real_cfg)

        # parsing the training schedule: scales : kimg
        assert isinstance(self.train_cfg['nkimgs_per_scale'],
                          dict), ('Please provide "nkimgs_per_'
                                  'scale" to schedule the training procedure.')

        nkimgs_per_scale = deepcopy(self.train_cfg['nkimgs_per_scale'])

        self.scales = []
        self.nkimgs = []

        for k, v in nkimgs_per_scale.items():
            if isinstance(k, str):
                k = (int(k), int(k))
            elif isinstance(k, int):
                k = (k, k)
            else:
                assert isinstance(k, tuple) or isinstance(k, list)

            assert len(self.scales) == 0 or k[0]> self.scales[-1][0]
            self.scales.append(k)
            self.nkimgs.append(v)

        self.cum_nkimgs = np.cumsum(self.nkimgs)
        self.curr_stage = 0
        self.prev_stage = 0

        # In each scale, transit from previous to rgb layer to newer to rgb layer
        # with `transition_kimgs` imgs
        self.transition_kimgs = self.train_cfg.get('transition_kimgs', 600)
        self.optimizer = build_dict_optimizer(self, self.train_cfg['optimizer_cfg'])
        self.g_lr_base = self.train_cfg['g_lr_base']
        self.d_lr_base = self.train_cfg['d_lr_base']

        # example for lr schedule: {'32': 0.001, '64': 0.0001}
        self.g_lr_schedule = self.train_cfg.get('g_lr_schedule', dict())
        self.d_lr_schedule = self.train_cfg.get('d_lr_schedule', dict())
        # reset the states for optimizers, e.g. momentum in Adam
        self.reset_optim_for_new_scale = self.train_cfg.get(
            'reset_optim_for_new_scale', True)
        # dirty walkround for avoiding optimizer bug in resuming
        self.prev_stage = self.train_cfg.get('prev_stage', self.prev_stage)

    def _parse_test_cfg(self):
        """Parsing train config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

        # whether to use exponential moving average for testing
        self.use_ema = self.test_cfg.get('use_ema', False)
        # TODO: finish ema part

    def train_step(self, data,
                   optimizer,
                   ddp_reducer=None,
                   running_status=None):
        # get data from data_batch
        real_imgs = data['real_img']

        # If you adopt ddp, this batch size is local batch size for each GPU.
        batch_size = real_imgs.shape[0]

        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        # check if optimizer from model
        if hasattr(self, 'optimizer'):
            optimizer = self.optimizer

        # update current stage
        self.curr_stage = int(
            min(
                sum(self.cum_nkimgs <= self.shown_nkimg.item()),
                len(self.scales) - 1))

        self.curr_scale = self.scales[self.curr_stage]
        self._curr_scale_int = self._next_scale_int.clone()

        if self.curr_stage == 0:
            transition_weight =1.
        else:
            transition_weight = (self.shown_nkimg.item() - self._actual_nkimgs[-1]) / self.transition_kimgs

        self._curr_transition_weight = torch.tensor(transition_weight).to(
            self._curr_transition_weight)

        # resize real image to target scale
        if real_imgs.shape[2:] == self.curr_scale:
            pass
        elif real_imgs.shape[2] >= self.curr_scale[0] and real_imgs.shape[
                3] >= self.curr_scale[1]:
            real_imgs = self.interp_real_to(real_imgs, size=self.curr_scale)
        else:
            raise RuntimeError(
                f'The scale of real image {real_imgs.shape[2:]} is smaller '
                f'than current scale {self.curr_scale}.')

        # discriminator training
        set_requires_grad(self.discriminator, True)
        optimizer['discriminator'].zero_grad()

        with torch.no_grad():
            fake_imgs = self.generator(None,
                                       num_batches=batch_size,
                                       curr_scale=self.curr_scale[0],
                                       transition_weight=transition_weight
                                       )

        # disc pred for fake images and real images
        disc_pred_fake = self.discriminator(
            fake_imgs,
            curr_scale=self.curr_scale[0],
            transition_weight=transition_weight
        )

        disc_pred_real = self.discriminator(
            real_imgs,
            curr_scale=self.curr_scale[0],
            transition_weight=transition_weight
        )
        # get data dict to compute losses for disc

        losses_dict = {}
        losses_dict['loss_disc_fake'] = self.gan_loss(disc_pred_fake, target_is_real=False, is_disc=True)
        losses_dict['loss_disc_real'] = self.gan_loss(disc_pred_real, target_is_real=True, is_disc=True)

        if hasattr(self, "shift_loss"):
            losses_dict['shift_loss'] = self.shift_loss(disc_pred_real) + self.shift_loss(disc_pred_fake)
        if hasattr(self, "gp_loss"):
            losses_dict['gp_loss'] = self.gp_loss(partial(self.discriminator,
                                                          curr_scale=self.curr_scale[0],
                                                          transition_weight=transition_weight
                                                          ),
                                                  real_data=real_imgs,
                                                  fake_data=fake_imgs
                                                  )

        loss_disc, log_vars_disc = self.parse_losses(losses_dict)

        loss_disc.backward()
        optimizer['discriminator'].step()

        if dist.is_initialized():
            _batch_size = batch_size * dist.get_world_size()
        else:
            _batch_size = batch_size

        self.shown_nkimg += (_batch_size / 1000.)

        log_vars_disc.update(
                dict(
                shown_nkimg=self.shown_nkimg.item(),
                curr_scale=self.curr_scale[0],
                transition_weight=transition_weight))

        # skip generator training if only train discriminator for current
        # iteration
        if (curr_iter + 1) % self.disc_steps != 0:
            results = dict(
                fake_imgs=fake_imgs.detach().cpu(), real_imgs=real_imgs.detach().cpu())
            outputs = dict(
                log_vars=log_vars_disc,
                num_samples=batch_size,
                results=results)
            if hasattr(self, 'iteration'):
                self.iteration += 1
            return outputs

        # generator training
        set_requires_grad(self.discriminator, False)
        optimizer['generator'].zero_grad()

        fake_imgs = self.generator(
            None,
            num_batches=batch_size,
            curr_scale=self.curr_scale[0],
            transition_weight=transition_weight
        )

        disc_pred_fake_g = self.discriminator(
            fake_imgs,
            curr_scale=self.curr_scale[0],
            transition_weight=transition_weight
        )

        losses_dict = {}
        # gan loss
        losses_dict['loss_disc_fake_g'] = self.gan_loss(
            disc_pred_fake_g,
            target_is_real=True,
            is_disc=False)

        loss_gen, log_vars_g = self.parse_losses(losses_dict)
        loss_gen.backward()
        optimizer['generator'].step()

        log_vars = {}
        log_vars.update(log_vars_g)
        log_vars.update(log_vars_disc)
        log_vars.update({'batch_size': batch_size})

        results = dict(fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1

        # check if a new scale will be added in the next iteration
        _curr_stage = int(
            min(
                sum(self.cum_nkimgs <= self.shown_nkimg.item()),
                len(self.scales) - 1))
        # in the next iteration, we will switch to a new scale
        if _curr_stage != self.curr_stage:
            # `self._next_scale_int` is updated at the end of `train_step`
            self._next_scale_int = self._next_scale_int * 2
        return outputs
