import argparse
import json
from collections import OrderedDict
from test_utils.utils import setup_logger, fromfile
from test_utils.engine.trainer import Trainer


def parse_config_file(config_file):
    with open(config_file, 'r') as f:
        config_dict = json.load(f, object_pairs_hook=OrderedDict)

    return config_dict


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        # default='tools/config/det/fcos/fcos_resnet18_caiqiebuliang.json',
                        # default='tools/config/det/fcos/fcos_resnet18_voc.json',
                        # default='tools/config/det/fcos/fcos_resnet18_finetune.json',
                        # default='tools/config/det/yolof/yolof_resnet18_caiqiebuliang_finetune.json',
                        # default='tools/config/det/yolof/yolof_resnet18_voc.json',
                        default="tools/config/det/fcos/fcos_resnet18_csp-pan_caiqiebuliang.json",
                        # default="tools/config/seg/caiqiebuliang_unet_resnet18.json",
                        # default= "tools/config/seg/caiqiebuliang_unet_mobilenetv3.json",
                        # default= "tools/config/cls/mobilnetv3_imagenet20.json",
                        type=str)
    parser.add_argument('-p','--pretrained', type=str, default='', help='initial weights path')

    args = parser.parse_args()

    cfg = parse_config_file(args.config)
    cfg = fromfile(cfg)
    if len(args.pretrained):
        cfg.update(dict(pretrained=args.pretrained))
    logger = setup_logger(cfg)

    def show_config_values(config, prefix='  '):
        for key in config.keys():
            if isinstance(config[key], dict):
                logger.info(f'{prefix} {key}')
                show_config_values(config[key], prefix='    '+prefix)
            else:
                logger.info(f'{prefix} {key.ljust(20) + str(config[key])}')

    logger.info('-' * 25 + 'log info'+'-' * 25)
    show_config_values(cfg)
    logger.info('-' * 25 + '-'*len('log info')+'-' * 25)
    return cfg, logger


if __name__ == "__main__":
    cfg, logger = setup()
    trainer = Trainer(cfg, logger)
    trainer.train()
