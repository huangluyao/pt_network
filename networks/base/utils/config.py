import ast
from addict import Dict
from importlib import import_module
import json
import os.path as osp
import shutil
import sys
import tempfile

from .misc import update_value_of_dict, repalce_kwargs_in_dict

BASE_KEY = '_base_'
OVERWRITE_KEY = '_overwrite_'
RESERVED_KEYS = ['filename', 'text']


def set_default(obj):
    """Set default json values for non-serializable values.

    It helps convert ``set``, ``range`` and ``np.ndarray`` data types to list,
    convert ``torch.Tensor`` to ``np.ndarray``, convert list of length 1 to float or int.
    """
    if isinstance(obj, (set, range)):
        return set_default(list(obj))
    elif isinstance(obj, np.ndarray):
        return set_default(obj.tolist())
    elif isinstance(obj, torch.Tensor):
        return set_default(obj.cpu().numpy())
    elif isinstance(obj, list):
        return obj[0] if len(obj) == 1 else obj
    elif isinstance(obj, (float, int)):
        return obj
    else:
        raise TypeError


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            excep = AttributeError(f"'{self.__class__.__name__}' object has no "
                                   f"attribute '{name}'")
        except Exception as e:
            excep = e
        else:
            return value
        raise excep


class Config:
    """Used to get the configuration in the configuration files (python/json/yaml).

    Examples
    --------
    >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
    >>> cfg.b
    {'b1': [0, 1]}
    >>> cfg = Config.fromfile('configs/imagenet/resnet34_bs32.json')
    >>> cfg.filename
    "/home/xxx/deepcv_server/configs/imagenet/resnet34_bs32.json"
    """

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)

    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename, 'r') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    @staticmethod
    def fromfile(filename):
        cfg_dict, cfg_text = Config._file2dict(filename)
        keys = cfg_dict.keys()
        for key in keys:
            if key.startswith('_') and key.endswith('_'):
                update_value_of_dict(cfg_dict, key, cfg_dict[key])
        repalce_kwargs_in_dict(cfg_dict)
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def _file2dict(filename):
        fileExtname = osp.splitext(filename)[1]
        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname)
            temp_config_name = osp.basename(temp_config_file.name)
            shutil.copyfile(filename, temp_config_file.name)

            if filename.endswith('.py'):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Config._validate_py_syntax(filename)
                temp_module = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in temp_module.__dict__.items()
                    if not name.startswith('__')
                }
                del sys.modules[temp_module_name]
            elif filename.endswith(('.json')):
                with open(temp_config_file.name, 'r') as f:
                    cfg_dict = json.load(f)
            temp_config_file.close()

        cfg_text = filename + '\n'
        with open(filename, 'r') as f:
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(
                base_filename, list) else [base_filename]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for _cfg_dict in cfg_dict_list:
                base_cfg_dict.update(_cfg_dict)

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            cfg_text_list.append(cfg_text)
            cfg_text = '\n'.join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b):
        b = b.copy()
        for k, v in a.items():
            if not isinstance(v, dict):
                b[k] = v
            elif k not in b:
                b[k] = v
            elif v.pop(OVERWRITE_KEY, False):
                b[k] = v
            elif not isinstance(b[k], dict):
                raise TypeError('You may set '
                    f'`{OVERWRITE_KEY}=True` to force an overwrite for %s.' %k)
            else:
                b[k] = Config._merge_a_into_b(v, b[k])
        return b

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self):
        return (self._cfg_dict, self._filename, self._text)

    def __setstate__(self, state):
        _cfg_dict, _filename, _text = state
        super(Config, self).__setattr__('_cfg_dict', _cfg_dict)
        super(Config, self).__setattr__('_filename', _filename)
        super(Config, self).__setattr__('_text', _text)

    def to_dict(self):
        return dict(self._cfg_dict)

    @property
    def text(self):
        return self._text