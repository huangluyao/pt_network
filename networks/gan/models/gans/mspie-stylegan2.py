import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from copy import deepcopy
from functools import partial
from ..builder import MODULES, build_module
from .base_gan import Base_GAN
from ..common import set_requires_grad


@MODULES.register_modules()
class MSPIEStyleGAN2(Base_GAN):
    """Unconditional GANs with static architecture in training.

    This is the standard GAN model containing standard adversarial training
    schedule. To fulfill the requirements of various GAN algorithms,
    ``disc_auxiliary_loss`` and ``gen_auxiliary_loss`` are provided to
    customize auxiliary losses, e.g., gradient penalty loss, and discriminator
    shift loss. In addition, ``train_cfg`` and ``test_cfg`` aims at setuping
    training schedule.

    Args:
        generator (dict): Config for generator.
        discriminator (dict): Config for discriminator.
        gan_loss (dict): Config for generative adversarial loss.
        disc_auxiliary_loss (dict): Config for auxiliary loss to
            discriminator.
        gen_auxiliary_loss (dict | None, optional): Config for auxiliary loss
            to generator. Defaults to None.
        train_cfg (dict | None, optional): Config for training schedule.
            Defaults to None.
        test_cfg (dict | None, optional): Config for testing schedule. Defaults
            to None.
    """

    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs
                 ):
        super(MSPIEStyleGAN2, self).__init__()
        self._gen_cfg=deepcopy(generator)
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

        if disc_auxiliary_loss:
            self.disc_auxiliary_loss = build_module(disc_auxiliary_loss)
            if not isinstance(self.disc_auxiliary_loss, nn.ModuleList):
                self.disc_auxiliary_loss = nn.ModuleList([self.disc_auxiliary_loss])

        else:
            self.disc_auxiliary_loss = None

        if gen_auxiliary_loss:
            self.gen_auxiliary_losses = build_module(gen_auxiliary_loss)
            if not isinstance(self.gen_auxiliary_losses, nn.ModuleList):
                self.gen_auxiliary_losses = nn.ModuleList(
                    [self.gen_auxiliary_losses])
        else:
            self.gen_auxiliary_losses = None

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parse_test_cfg()

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        # whether to use exponential moving average for training
        self.use_ema = self.train_cfg.get('use_ema', False)
        if self.use_ema:
            # use deepcopy to guarantee the consistency
            self.generator_ema = deepcopy(self.generator)

        self.real_img_key = self.train_cfg.get('real_img_key', 'real_img')

        # calculate the current result size according to the size of the input
        # feature map, e.g., positional encoding map
        # [h, w]
        self.generated_size = self.train_cfg.get('generated_size', [256, 256])
        # multiple input scales (a list of int) that will be added to the
        # original starting scale.
        self.multi_input_scales = self.train_cfg.get('multi_input_scales')
        self.multi_scale_probability = self.train_cfg.get(
            'multi_scale_probability')

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

        # whether to use exponential moving average for testing
        self.use_ema = self.test_cfg.get('use_ema', False)
        # TODO: finish ema part

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   running_status=None):
        """Train step function.

        This function implements the standard training iteration for
        asynchronous adversarial training. Namely, in each iteration, we first
        update discriminator and then compute loss for generator with the newly
        updated discriminator.

        As for distributed training, we use the ``reducer`` from ddp to
        synchronize the necessary params in current computational graph.

        Args:
            data_batch (dict): Input data from dataloader.
            optimizer (dict): Dict contains optimizer for generator and
                discriminator.
            ddp_reducer (:obj:`Reducer` | None, optional): Reducer from ddp.
                It is used to prepare for ``backward()`` in ddp. Defaults to
                None.
            running_status (dict | None, optional): Contains necessary basic
                information for training, e.g., iteration number. Defaults to
                None.

        Returns:
            dict: Contains 'log_vars', 'num_samples', and 'results'.
        """
        # get data from data_batch
        real_imgs = data_batch['real_img']
        # If you adopt ddp, this batch size is local batch size for each GPU.
        # If you adopt dp, this batch size is the global batch size as usual.
        batch_size = real_imgs.shape[0]

        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        if dist.is_initialized():
            # randomly sample a scale for current training iteration
            chosen_scale = np.random.choice(self.multi_input_scales, 1,
                                            self.multi_scale_probability)[0]

            chosen_scale = torch.tensor(chosen_scale, dtype=torch.int).cuda()
            dist.broadcast(chosen_scale, 0)
            chosen_scale = int(chosen_scale.item())

        else:
            chosen_scale = 0

        if real_imgs.shape[-2:] != self.generated_size:
            real_imgs = F.interpolate(
                real_imgs,
                size=self.generated_size,
                mode='bilinear',
                align_corners=True)

        # disc training
        set_requires_grad(self.discriminator, True)
        optimizer['discriminator'].zero_grad()
        with torch.no_grad():
            fake_imgs = self.generator(None, num_batches=batch_size, chosen_scale=0)

        # disc pred for fake imgs and real_imgs
        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)
        # get data dict to compute losses for disc
        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            iteration=curr_iter,
            batch_size=batch_size,
            gen_partial= partial(self.generator, chosen_scale=chosen_scale)
        )

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

        loss_disc.backward()
        optimizer['discriminator'].step()

        # skip generator training if only train discriminator for current
        # iteration
        if (curr_iter + 1) % self.disc_steps != 0:
            results = dict(
                fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
            log_vars_disc['curr_size'] = self.generated_size
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

        # TODO: add noise sampler to customize noise sampling
        fake_imgs = self.generator(
            None, num_batches=batch_size, chosen_scale=chosen_scale)
        disc_pred_fake_g = self.discriminator(fake_imgs)

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=fake_imgs,
            disc_pred_fake_g=disc_pred_fake_g,
            iteration=curr_iter,
            batch_size=batch_size,
            gen_partial=partial(self.generator, chosen_scale=chosen_scale))

        loss_gen, log_vars_g = self._get_gen_loss(data_dict_)

        loss_gen.backward()
        optimizer['generator'].step()

        log_vars = {}
        log_vars.update(log_vars_g)
        log_vars.update(log_vars_disc)
        log_vars['curr_size'] = self.generated_size

        results = dict(fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs
