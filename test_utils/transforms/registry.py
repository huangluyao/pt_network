
class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def registry(self, name=None):
        def decorator(cls):
            if name is None:
                self._module_dict[cls.__name__] = cls
            else:
                self._module_dict[name] = cls
            return cls
        return decorator

    def get(self, key):
        return self._module_dict.get(key, None)

    @property
    def name(self):
        return self._name


def build_from_cfg(cfg, registry):

    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
            raise KeyError('`cfg` or `default_args` must contain the key "type", ')
    args = cfg.copy()
    obj_type = args.pop('type')

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
        return obj_cls(**args)
