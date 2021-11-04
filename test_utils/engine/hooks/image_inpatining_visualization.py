import cv2
import os
import torch
from .hook import Hook, HOOKS
from test_utils.utils.file_io import mkdir
from .visualize_training_samples import make_grid

@HOOKS.registry()
class ImageInpaintingVisualizationHook(Hook):

    def __init__(self,
                 res_name_list,
                 output_dir='training_sample',
                 interval=-1,
                 padding=0,
                 nrow = 2,
               **kwargs):
        super(ImageInpaintingVisualizationHook, self).__init__()
        self.interval = interval
        self.output_dir = output_dir
        self.padding = padding
        self.res_name_list = res_name_list
        self.nrow = nrow

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        results = runner.outputs['results']

        file_name = 'iter_{}.png'.format(runner.iter + 1)
        img_list = [x for k, x in results.items() if k in self.res_name_list]
        img_cat = torch.cat(img_list, dim=3).detach()
        img_cat = ((img_cat + 1) / 2)
        img_cat = img_cat.clamp_(0, 1)

        grid = make_grid(img_cat, nrow=self.nrow, padding=self.padding)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        save_folder = os.path.join(runner.work_dir, self.output_dir)
        mkdir(save_folder)

        cv2.imwrite(os.path.join(save_folder, file_name), ndarr)