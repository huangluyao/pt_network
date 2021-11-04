import torch.cuda

from networks.gan.models.architectures import StyleGANv2Generator, StyleGAN2Discriminator


if __name__=="__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = StyleGANv2Generator(512,
                              style_channels=512,
                              num_mlps=8,
                              channel_multiplier=1,
                              lr_mlp=0.01,
                              ).to(device)
    out = gen(None, num_batches=4)
    print(out.shape)

    dis = StyleGAN2Discriminator(512,channel_multiplier=1).to(device)

    print(dis(out).shape)