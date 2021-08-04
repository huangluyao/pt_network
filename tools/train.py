import argparse
import json
from collections import OrderedDict
from test_utils.utils import setup_logger, fromfile
from test_utils.engine.trainer import Trainer


def parse_config_file(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        # data = f.read().decode(encoding='gbk').encode(encoding='utf-8')
        # config_dict= json.loads(data)
        config_dict = json.load(f, object_pairs_hook=OrderedDict)

    return config_dict


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        # default="tools/config/det/yolox/resnet18_csp-pan_yolox.json",
                        default="tools/config/det/fcos/smart-backbone_csp-pan_fcos.json",
                        type=str)
    parser.add_argument('-p', '--pretrained', type=str,
                        default='',
                        help='initial weights path')

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
