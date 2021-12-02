import os

import torch
import matplotlib.pyplot as plt
from networks.base.cnn import ACTIVATION_LAYERS, build_activation_layer


def plot_with_tensor(x, y, name, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    plt.title(name)
    plt.plot(x, y, )
    plt.savefig(os.path.join(save_path, name))
    plt.grid()
    plt.clf()
    plt.close()


if __name__ == "__main__":
    acts = ACTIVATION_LAYERS

    cfg_list = [
        dict(type="ReLU"),
        dict(type="LeakyReLU"),
        dict(type="PReLU"),
        dict(type="RReLU"),
        dict(type="ReLU6"),
        dict(type="ELU"),
        dict(type="Sigmoid"),
        dict(type="Tanh"),
        dict(type="GELU"),
        dict(type="SiLU"),
        dict(type="HardSwish"),
        dict(type="HardSigmoid"),
        dict(type="Sine"),
        dict(type="Mish"),
        dict(type="FReLU"),
        dict(type="AconC"),
        dict(type="MetaAconC"),
    ]
    x = torch.arange(-5, 5, 0.05)

    act_list = [
        'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', "SiLU", "HardSwish", "HardSigmoid",
        'MetaAconC', 'AconC', 'FReLU', 'Mish', 'Sine', 'GELU'
    ]

    for cfg in cfg_list:
        if not cfg.get('type') in act_list:
            print(cfg.get('type') )

    for cfg in cfg_list:
        act = build_activation_layer(cfg)
        y = act(x)
        plot_with_tensor(x, y, name=act._get_name(), save_path="test/activates")
    pass


