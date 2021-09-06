from transforms.registry import build_from_cfg
from . import TRANSFORM


class Compose:
    def __init__(self, transforms):
        """Compose multiple transforms sequentially.

        Args:
            transforms (Sequence[dict | callable]): Sequence of transform object or
                config dict to be composed.
        """
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, TRANSFORM)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or dict')

    def __call__(self, **data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(**data)
            if data is None:
                return None
        return data

