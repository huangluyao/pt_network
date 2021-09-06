import numpy as np

def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Parameters
    ----------
    inputs : dict
        The input dict with str keys.
    prefix : str
        The prefix to add.

    Returns
    -------
    dict
        The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
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
