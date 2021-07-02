import cv2
from functools import partial
import numpy as np
from six.moves import map, zip
import warnings

import torch


def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)
    cv2.add(img, mean, img)
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
    return img


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    """Convert tensor to images.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor that contains multiple images
    mean : tuple[float], optional
        Mean of images. Defaults to (0, 0, 0).
    std : tuple[float], optional
        Standard deviation of images.
        Defaults to (1, 1, 1).
    to_rgb : bool, optional
        Whether convert the images to RGB format.
        Defaults to True.

    Returns
    -------
    list[np.ndarray]
        A list that contains multiple images.
    """
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Notes
    -----
    This function applies the ``func`` to multiple inputs and
    map the multiple outputs of the ``func`` into different
    list. Each list contains the same type of outputs corresponding
    to different inputs.

    Parameters
    ----------
    func : Function
        A function that will be applied to a list of arguments.

    Returns
    -------
    tuple : list
        A tuple containing multiple list, each list contains
        a kind of returned results by the function.
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret
