# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/10 下午5:45
import cv2
import numpy as np
from ..utils import preserve_channel_dim, preserve_shape
from ..utils import is_grayscale_image, is_rgb_image, clip

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

@preserve_channel_dim
def resize(img, height, width, interpolation=cv2.INTER_LINEAR):
    img_height, img_width = img.shape[:2]
    if height == img_height and width == img_width:
        return img
    img = cv2.resize(img, dsize=(width, height), interpolation=interpolation)
    return img


@preserve_shape
def random_flip(img, code=-1):
    return cv2.flip(img, code)


def bbox_vflip(bbox, rows, cols):
    x_min, y_min, x_max, y_max = bbox[:4]
    return x_min, rows - y_max, x_max, rows - y_min


def bbox_hflip(bbox, rows, cols):  # skipcq: PYL-W0613
    x_min, y_min, x_max, y_max = bbox[:4]
    return cols - x_max, y_min, cols - x_min, y_max


def bbox_flip(bbox, d, rows, cols):
    if d == 0:
        bbox = bbox_vflip(bbox, rows, cols)
    elif d == 1:
        bbox = bbox_hflip(bbox, rows, cols)
    elif d == -1:
        bbox = bbox_hflip(bbox, rows, cols)
        bbox = bbox_vflip(bbox, rows, cols)
    else:
        raise ValueError("Invalid d value {}. Valid values are -1, 0 and 1".format(d))
    return bbox



@preserve_channel_dim
def rotate(img, angle, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None):
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(img, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value)


def bbox_rotate(bbox, angle, rows, cols):
    """Rotates a bounding box by angle degrees.

    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        angle (int): Angle of rotation in degrees.
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    x_min, y_min, x_max, y_max = bbox[:4]
    boxes_points = np.array([[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]])

    new_bboxes_points = np.concatenate([boxes_points, np.ones([4, 1])], axis=-1)
    res = np.matmul(new_bboxes_points, matrix.transpose())
    x_min, y_min, w, h = cv2.boundingRect(res.astype(np.int32))

    # x_min, y_min, x_max, y_max = bbox[:4]
    # scale = cols / float(rows)
    # x = np.array([x_min, x_max, x_max, x_min]) - 0.5
    # y = np.array([y_min, y_min, y_max, y_max]) - 0.5
    # angle = np.deg2rad(angle)
    # x_t = (np.cos(angle) * x * scale + np.sin(angle) * y) / scale
    # y_t = -np.sin(angle) * x * scale + np.cos(angle) * y
    # x_t = x_t + 0.5
    # y_t = y_t + 0.5
    #
    # x_min, x_max = min(x_t), max(x_t)
    # y_min, y_max = min(y_t), max(y_t)

    return x_min, y_min, x_min+w, y_min+h


def normalize(img, mean, std, scale=1.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= scale

    std = np.array(std, dtype=np.float32)
    std *= scale

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def adjust_brightness(img, factor):
    if factor == 0:
        return np.zeros_like(img)
    elif factor == 1:
        return img

    if img.dtype == np.uint8:
        lut = np.arange(0, 256) * factor
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        return cv2.LUT(img, lut)

    return clip(img * factor, img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])


def adjust_contrast(img, factor):
    if factor == 1:
        return img

    if is_grayscale_image(img):
        mean = img.mean()
    else:
        mean = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()

    if factor == 0:
        return np.full_like(img, int(mean + 0.5), dtype=img.dtype)

    if img.dtype == np.uint8:
        lut = np.arange(0, 256) * factor
        lut = lut + mean * (1 - factor)
        lut = clip(lut, img.dtype, 255)

        return cv2.LUT(img, lut)

    return clip(
        img.astype(np.float32) * factor + mean * (1 - factor),
        img.dtype,
        MAX_VALUES_BY_DTYPE[img.dtype],
    )


def adjust_saturation(img, factor, gamma=0):
    if factor == 1:
        return img

    if is_grayscale_image(img):
        gray = img
        return gray
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if factor == 0:
        return gray

    result = cv2.addWeighted(img, factor, gray, 1 - factor, gamma=gamma)
    if img.dtype == np.uint8:
        return result

    # OpenCV does not clip values for float dtype
    return clip(result, img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])


def adjust_hue(img, factor):
    if is_grayscale_image(img):
        return img

    if factor == 0:
        return img

    if img.dtype == np.uint8:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lut = np.arange(0, 256, dtype=np.int16)
        lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
        img[..., 0] = cv2.LUT(img[..., 0], lut)
        return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[..., 0] = np.mod(img[..., 0] + factor * 360, 360)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def multiply(img, multiplier):
    """
    Args:
        img (numpy.ndarray): Image.
        multiplier (numpy.ndarray): Multiplier coefficient.

    Returns:
        numpy.ndarray: Image multiplied by `multiplier` coefficient.

    """
    if img.dtype == np.uint8:
        if len(multiplier.shape) == 1:
            return _multiply_uint8_optimized(img, multiplier)
        return _multiply_uint8(img, multiplier)
    return _multiply_non_uint8(img, multiplier)


def _multiply_non_uint8(img, multiplier):
    dtype = img.dtype
    maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
    return  clip(img * multiplier,  dtype, maxval)


def _multiply_uint8(img, multiplier):
    dtype = img.dtype
    maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
    img = img.astype(np.float32)
    return clip(np.multiply(img, multiplier), dtype, maxval)


def _multiply_uint8_optimized(img, multiplier):
    if is_grayscale_image(img) or len(multiplier) == 1:
        multiplier = multiplier[0]
        lut = np.arange(0, 256, dtype=np.float32)
        lut *= multiplier
        lut = clip(lut, np.uint8, MAX_VALUES_BY_DTYPE[img.dtype])
        func = _maybe_process_in_chunks(cv2.LUT, lut=lut)
        return func(img)

    channels = img.shape[-1]
    lut = [np.arange(0, 256, dtype=np.float32)] * channels
    lut = np.stack(lut, axis=-1)

    lut *= multiplier
    lut = clip(lut, np.uint8, MAX_VALUES_BY_DTYPE[img.dtype])

    images = []
    for i in range(channels):
        func = _maybe_process_in_chunks(cv2.LUT, lut=lut[:, i])
        images.append(func(img[:, :, i]))
    return np.stack(images, axis=-1)


def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.

    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.

    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.

    Returns:
        numpy.ndarray: Transformed image.

    """

    def __process_fn(img):
        num_channels = img.shape[2] if len(img.shape) == 3 else 1
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


def random_crop(img, crop_height, crop_width, h_start, w_start):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )

    x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    img = img[y1:y2, x1:x2]
    return img


def get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start):
    y1 = int((height-crop_height)*h_start)
    y2 = y1+ crop_height
    x1 = int((width-crop_width)* w_start)
    x2 = x1+ crop_width
    return x1, y1, x2, y2


def bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols):
    crop_coords = get_random_crop_coords(rows, cols, crop_height, crop_width, h_start, w_start)

    return crop_bbox_by_coords(bbox, crop_coords)

def crop_bbox_by_coords(bbox,  crop_coords):
    x_min, y_min, x_max, y_max = bbox[:4]
    x1, y1, _, _ = crop_coords
    cropped_bbox = x_min - x1, y_min - y1, x_max - x1, y_max - y1
    return cropped_bbox