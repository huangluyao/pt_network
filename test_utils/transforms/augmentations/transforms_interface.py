import random
import cv2


class BasicTransform:
    def __init__(self, always_apply=False, prob=0.5):
        self.p = prob
        self.always_apply = always_apply

    def __call__(self, *args, force_apply=False, **kwargs):
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params(**kwargs)
            res = {}
            for key, arg in kwargs.items():
                if arg is not None:
                    target_function = self._get_target_function(key)
                    res[key] = target_function(arg, **params)
            return res
        return kwargs

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
        }

    def _get_target_function(self, key):
        return self.targets.get(key, lambda x, **p: x)

    def apply(self, img, **params):
        raise NotImplementedError

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_bboxes(self, bboxes, **params):
        return [self.apply_to_bbox(tuple(bbox[:4]), **params) + tuple(bbox[4:]) for bbox in bboxes]

    def apply_to_mask(self, img, **params):
        return self.apply(img, **params)

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def get_params(self, **params):
        return {}