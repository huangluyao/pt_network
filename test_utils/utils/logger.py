# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/11 下午2:13
import logging
import sys
import os
import time
from termcolor import colored
from .file_io import mkdir


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def setup_logger(
    cfg, distributed_rank=0, *, color=True, name="DeepSight", abbrev_name=None
):
    output = cfg.get("output_dir", None)
    cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    task_name = cfg['model'].get('type', None)
    backbone_name = cfg['model'].get('backbone', dict()).get('type', None)
    neck_name = cfg['model'].get('neck', None)
    head_name = cfg['model'].get("decode_head", None)
    datasets_name = cfg['dataset'].get('type')
    if neck_name is not None:
        neck_name = neck_name.get('type', None)
        task_name = f"{task_name}_{backbone_name}_{neck_name}_{datasets_name}"
    elif head_name is not None:
        head_name = head_name.get('type', None)
        task_name = f"{task_name}_{backbone_name}_{head_name}_{datasets_name}"
    else:
        task_name = f"{task_name}_{backbone_name}_{datasets_name}"
    if output is None:
        output = os.path.join('./export', cfg["task"], task_name, cur_time)
    else:
        output = os.path.join(output)
    cfg.update({"output_dir": output})

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if abbrev_name is None:
        abbrev_name = "ds" if name == "DeepSight" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(filename)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if not os.path.exists(output):
            mkdir(output)
        filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

    return logger

