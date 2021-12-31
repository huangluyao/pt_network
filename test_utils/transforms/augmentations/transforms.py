import numbers
import os
import torch
from .. import TRANSFORM
from .transforms_interface import BasicTransform
from .functional import *


@TRANSFORM.registry()
class Resize(BasicTransform):
    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, padding=True, prob=1, always_apply=False):
        super(Resize, self).__init__(always_apply, prob)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.padding = padding

    def apply(self, img, **params):
        return padding_resize(img, (self.height, self.width)) if self.padding else \
            resize(img, self.height, self.width, self.interpolation)

    def apply_to_bbox(self, bbox, **params):
        org_size = (params["rows"], params["cols"])
        return padding_resize_box(bbox, org_size, (self.height, self.width)) if self.padding else \
            resize_box(bbox, org_size, (self.height, self.width))

    def apply_to_mask(self, img, **params):
        return padding_resize_mask(img, (self.height, self.width)) if self.padding else \
            resize(img, self.height, self.width, cv2.INTER_NEAREST)

    def get_params(self, **params):

        if params.get("image", None) is not None:
            return {"cols": params["image"].shape[1], "rows": params["image"].shape[0]}
        elif params.get("images", None) is not None:
            return {"cols": params["images"][0].shape[1], "rows": params["images"][0].shape[0]}
        else:
            raise ValueError("not image in resize")


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
        return random_flip(img, self.direction)

    def get_params(self, **params):
        if params.get("image", None) is not None:
            return {"cols": params["image"].shape[1], "rows": params["image"].shape[0]}
        elif params.get("images", None) is not None:
            return {"cols": params["images"][0].shape[1], "rows": params["images"][0].shape[0]}
        else:
            raise ValueError("not image in resize")


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

        cols, rows = (kwargs["image"].shape[1], kwargs["image"].shape[0]) if kwargs.get("image", None) is not None else \
            (kwargs["images"][0].shape[1], kwargs["images"][0].shape[0])

        return {"angle": random.uniform(self.limit[0], self.limit[1]),
                "cols": cols, "rows": rows
                }

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return rotate(img, angle, interpolation, self.border_mode, self.value)

    def apply_to_bbox(self, bbox, angle=0, **params):
        return bbox_rotate(bbox, angle, params["rows"], params["cols"])

    def apply_to_mask(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return rotate(img, angle, interpolation, self.border_mode, self.value)

@TRANSFORM.registry()
class Normalize(BasicTransform):
    def __init__(self,mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 scale=1.0, **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.scale = scale

    def apply(self, img, **params):
        return normalize(img, self.mean, self.std, self.scale)

    @property
    def targets(self):
        return {"image": self.apply, "images": self.apply_to_images} # image only transform


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
        return {"image": self.apply, "images": self.apply_to_images} # image only transform


@TRANSFORM.registry()
class MultiplicativeNoise(BasicTransform):

    def __init__(self, multiplier=(0.9, 1.1), per_channel=False, **kwargs):
        super(MultiplicativeNoise, self).__init__(**kwargs)

        if isinstance(multiplier, (int, float)):
            self.multiplier = -multiplier, +multiplier
        else:
            self.multiplier = multiplier
        self.per_channel = per_channel

    def apply(self, img, multiplier=np.array([1]), **params):

        return multiply(img, multiplier)

    def get_params(self, **params):
        if self.multiplier[0] == self.multiplier[1]:
            return {"multiplier": np.array([self.multiplier[0]])}

        img = params.get("image", None)
        if img is None:
            img = params.get("images")[0]

        h, w = img.shape[:2]

        if self.per_channel:
            c = 1 if is_grayscale_image(img) else img.shape[-1]
        else:
            c = 1

        multiplier = np.random.uniform(self.multiplier[0], self.multiplier[1], [c])
        if is_grayscale_image(img):
            multiplier = np.squeeze(multiplier)

        return {"multiplier": multiplier}


    @property
    def targets(self):
        return {"image": self.apply, "images": self.apply_to_images} # image only transform

@TRANSFORM.registry()
class RandomCrop(BasicTransform):
    def __init__(self, height, width, **kwargs):
        super(RandomCrop, self).__init__(**kwargs)
        self.height = height
        self.width = width

    def apply(self, img, h_start=0, w_start=0, **params):
        return random_crop(img, self.height, self.width, h_start, w_start)

    def get_params(self, **params):
        height, width = params["image"].shape[:2] if params.get("image", None) is not None else params.get("images")[0].shape[:2]
        return {"h_start": random.random(), "w_start":random.random(), "rows":width, "cols":height}

    def apply_to_mask(self, img, h_start=0, w_start=0, **params):
        return random_crop(img, self.height, self.width, h_start, w_start)

    def apply_to_bbox(self,bbox,rows, cols, h_start=0, w_start=0, **params):
        return bbox_random_crop(bbox, self.height, self.width, h_start, w_start, rows, cols)

@TRANSFORM.registry()
class GaussNoise(BasicTransform):
    def __init__(self,var_limit=(10.0, 50.0), mean=0, **kwargs):
        super(GaussNoise, self).__init__(**kwargs)
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(type(var_limit))
            )

        self.mean = mean

    def apply(self, img, gauss=None, **params):
        return gauss_noise(img, gauss=gauss)

    def get_params(self, **params):
        image = params["image"] if params.get("image", None) is not None else params.get("images")[0]

        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

        gauss = random_state.normal(self.mean, sigma, image.shape)
        return {"gauss": gauss}

    @property
    def targets(self):
        return {"image": self.apply, "images": self.apply_to_images} # image only transform


@TRANSFORM.registry()
class ToTensor(BasicTransform):
    def __init__(self, transpose_mask=False, **kwargs):
        super(ToTensor, self).__init__( **kwargs)
        self.transpose_mask = transpose_mask

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [2, 3]:
            raise ValueError("ToTensor only supports images in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask, **params):
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask)


@TRANSFORM.registry()
class CopyPaste(BasicTransform):
    def __init__(self, image_infos, min_scale=0.5, max_scale=1.5, **kwargs):
        super(CopyPaste, self).__init__(**kwargs)
        self.img_infos = self.check_image_infos(image_infos)
        self.min_scale = min_scale
        self.max_scale = max_scale


    def get_params(self, **params):

        image_info = np.random.choice(self.img_infos)
        image = cv2.imread(image_info["image_path"])
        mask = np.zeros(image.shape[:2])
        for label_point, index in zip(image_info["label_points"], image_info["label_index"]):
            mask = cv2.fillPoly(mask, [np.int0(label_point)], (int(index)))
        rescale_ratio = np.random.uniform(self.min_scale, self.max_scale)

        return {
            "img_src":image,
            "mask_src":mask,
            "rescale_ratio":rescale_ratio
        }

        pass

    def apply(self, img, **params):
        return copy_paste_img(img, params["mask_src"], params["img_src"], params["rescale_ratio"])

    def apply_to_mask(self, img, **params):
        return copy_paste_mask(img, params["mask_scr"], params["rescale_ratio"], 0)

    def check_image_infos(self, image_infos):
        # check image_infos have object
        return [image_info for image_info in image_infos if len(image_info["label_points"]) > 0]

