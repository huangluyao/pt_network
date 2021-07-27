from ...utils import Registry, build_from_cfg

BLOCK_LAYERS = Registry('block_layer')
CONV_LAYERS = Registry('conv_layer')
NORM_LAYERS = Registry('norm_layer')
ACTIVATION_LAYERS = Registry('activation_layer')
PADDING_LAYERS = Registry('padding_layer')
UPSAMPLE_LAYERS = Registry('upsample_layer')
PLUGIN_LAYERS = Registry('plugin_layer')


def build_plugin_layers(cfg):
    return build_from_cfg(cfg, PLUGIN_LAYERS)


