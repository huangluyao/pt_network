import torch.nn as nn
from ..conv_module import ConvModule


class TransformerLayer(nn.Module):

    def __init__(self, in_channel, num_heads):
        super(TransformerLayer, self).__init__()
        self.q = nn.Linear(in_channel, in_channel, bias=False)
        self.k = nn.Linear(in_channel, in_channel, bias=False)
        self.v = nn.Linear(in_channel, in_channel, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=in_channel, num_heads=num_heads)
        self.fc1 = nn.Linear(in_channel, in_channel, bias=False)
        self.fc2 = nn.Linear(in_channel, in_channel, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x))+x
        return x


class TransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers):
        super(TransformerBlock, self).__init__()
        self.conv = None
        if c1 != c2:
            norm_cfg = dict(type="BN2d"),
            act_cfg = dict(type='SiLu'),
            self.conv = ConvModule(c1,c2, kernel_size=1, padding=0, stride=1, norm_cfg=norm_cfg,act_cfg=act_cfg)

        self.linear = nn.Linear(c2, c2)
        self.transformer = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0,3).squeeze(3)
        return self.transformer(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)