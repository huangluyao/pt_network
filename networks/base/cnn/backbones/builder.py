from ...utils import Registry, build_module

BACKBONES = Registry('backbone')


def build_backbone(cfg):
    model_type = cfg.get("type")
    if BACKBONES.get(model_type) is not None:
        return build_module(cfg, BACKBONES)
    else:
        import timm
        return timm.create_model(model_name=cfg.pop("type"), pretrained=True, **cfg)
