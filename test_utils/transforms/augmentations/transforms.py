import numbers

import cv2
import random

from .. import TRANSFORM
from .transforms_interface import BasicTransform
from .functional import *


@TRANSFORM.registry()
class Resize(BasicTransform):
    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, prob=1, always_apply=False):
        super(Resize, self).__init__(always_apply, prob)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, **params):
        return resize(img, self.height, self.width, self.interpolation)

    def apply_to_bbox(self, bbox, **params):
        height = params["rows"]
        width = params["cols"]
        scale_x = self.width / width
        scale_y = self.height / height
        (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])

        x_min, x_max = x_min * scale_x, x_max * scale_x
        y_min, y_max = y_min * scale_y, y_max * scale_y
        return (x_min, y_min, x_max, y_max) + tail

    def apply_to_mask(self, img, **params):
        pass

    def get_params(self, **params):
        return {"cols": params["image"].shape[1], "rows": params["image"].shape[0]}


@TRANSFORM.registry()
class RandomFlip(BasicTransform):
    def __init__(self, direction=None, **kwargs):
        if direction is None:
            self.direction = -1
        elif direction == "horizontal":
            self.direction = 1
        elif direction == "vertical":
            self.direction = 0
        else:
            raise TypeError('direction must be horizontal, vertical or None,'
                             f'but got {direction}')
        super(RandomFlip, self).__init__(**kwargs)

    def apply(self, img, **params):
        return random_flip(img, self.direction)

    def apply_to_bbox(self, bbox, **params):
        return bbox_flip(bbox, self.direction, params["rows"], params['cols'])

    def apply_to_mask(self, img, **params):
        pass

    def get_params(self, **params):
        return {"cols": params["image"].shape[1], "rows": params["image"].shape[0]}


@TRANSFORM.registry()
class Rotate(BasicTransform):
    def __init__(
        self,
        limit,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        prob=0.5,
    ):
        super(Rotate, self).__init__(always_apply, prob)
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def get_params(self, **kwargs):
        return {"angle": random.uniform(self.limit[0], self.limit[1]),
                "cols": kwargs["image"].shape[1], "rows": kwargs["image"].shape[0]
                }

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return rotate(img, angle, interpolation, self.border_mode, self.value)

    def apply_to_bbox(self, bbox, angle=0, **params):
        return bbox_rotate(bbox, angle, params["rows"], params["cols"])


@TRANSFORM.registry()
class Normalize(BasicTransform):
    def __init__(self,mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 scale=1.0, **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.scale = scale

    def apply(self, img, **params):
        return normalize(img, self.mean, self.std, self.scale)

    @property
    def targets(self):
        return {"image": self.apply} # image only transform


@TRANSFORM.registry()
class ColorJitter(BasicTransform):
    def __init__(self,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.2, **kwargs):
        super(ColorJitter, self).__init__(**kwargs)
        self.brightness = self.__check_values(brightness, "brightness")
        self.contrast = self.__check_values(contrast, "contrast")
        self.saturation = self.__check_values(saturation, "saturation")
        self.hue = self.__check_values(hue, "hue", offset=0, bounds=[-0.5, 0.5], clip=False)

    def __check_values(self, value, name, offset=1, bounds=(0, float("inf")), clip=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be positive")

            value = [offset - value, offset + value]

            if clip:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
                raise ValueError("{} values should be between {}".format(name, bounds))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))
        return value

    def get_params(self, **kwargs):
        brightness = random.uniform(self.brightness[0], self.brightness[1])
        contrast = random.uniform(self.contrast[0], self.contrast[1])
        saturation = random.uniform(self.saturation[0], self.saturation[1])
        hue = random.uniform(self.hue[0], self.hue[1])

        transforms = [
            lambda x: adjust_brightness(x, brightness),
            lambda x: adjust_contrast(x, contrast),
            lambda x: adjust_saturation(x, saturation),
            lambda x: adjust_hue(x, hue),
        ]
        random.shuffle(transforms)

        return {"transforms": transforms}

    def apply(self, img, transforms=(), **params):
        for transform in transforms:
            img = transform(img)
        return img

    @property
    def targets(self):
        return {"image": self.apply} # image only transform
