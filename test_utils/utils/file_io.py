# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/11 下午2:20
import os
from importlib import import_module
import json
import os.path as osp
import shutil
import sys
import tempfile
import platform


def mkdir(path):
    if not os.path.isdir(path):
        parent_path = os.path.dirname(path)
        mkdir(parent_path)
        os.mkdir(path)


def _merge_a_into_b(a, b):
    b = b.copy()
    for k, v in a.items():
        if not isinstance(v, dict):
            b[k] = v
        elif k not in b:
            b[k] = v
        else:
            if not isinstance(b[k], dict):
                b[k] = dict()   #
            b[k] = _merge_a_into_b(v, b[k])
    return b


def _file2dict(filename):
    BASE_KEY = '_base_'
    fileExtname = osp.splitext(filename)[1]
    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_dir = os.path.dirname(__file__)
        temp_config_file = tempfile.NamedTemporaryFile(
            dir=temp_dir, suffix=fileExtname)
        temp_config_file.name = os.path.join(temp_dir, "temp.json")

        shutil.copyfile(filename, temp_config_file.name)
        filename.endswith(('.json'))
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
            _cfg_dict, _cfg_text = _file2dict(osp.join(cfg_dir, f))
            cfg_dict_list.append(_cfg_dict)
            cfg_text_list.append(_cfg_text)

        base_cfg_dict = dict()
        for _cfg_dict in cfg_dict_list:
            # base_cfg_dict.update(_cfg_dict)
            base_cfg_dict = _merge_a_into_b(base_cfg_dict, _cfg_dict)

        base_cfg_dict = _merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = base_cfg_dict

        cfg_text_list.append(cfg_text)
        cfg_text = '\n'.join(cfg_text_list)

    return cfg_dict, cfg_text


def fromfile(cfg):

    if '__base__' in cfg:
        base_files = cfg.pop('__base__')
        if isinstance(base_files, str):
            base_files = [base_files]
        assert isinstance(base_files, list), '__base__ must be a str or list'

        for base_file in base_files:
            cfg_dict, cfg_text = _file2dict(base_file)
            keys = cfg_dict.keys()
            for key in keys:
                if key.startswith('_') and key.endswith('_'):
                    update_value_of_dict(cfg_dict, key, cfg_dict[key])
            repalce_kwargs_in_dict(cfg_dict)
            cfg = _merge_a_into_b(cfg, cfg_dict)

    return cfg


def update_value_of_dict(_dict, old_value, new_value):
    if not isinstance(_dict, dict):
        return
    tmp = _dict
    for k, v in tmp.items():
        if isinstance(v, str) and v == old_value:
            _dict[k] = new_value
        else:
            if isinstance(v, dict):
                update_value_of_dict(_dict[k], old_value, new_value)
            elif isinstance(v, list):
                for _item in v:
                    update_value_of_dict(_item, old_value, new_value)


def repalce_kwargs_in_dict(_dict):
    if not isinstance(_dict, dict):
        return
    _items = _dict.copy().items()
    for k, v in _items:
        if 'kwargs' == k:
            _kwargs = _dict.pop('kwargs')
            _dict.update(_kwargs)
        else:
            if isinstance(v, dict):
                repalce_kwargs_in_dict(_dict[k])
            elif isinstance(v, list):
                for _item in v:
                    repalce_kwargs_in_dict(_item)
