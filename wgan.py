import torch
from torch import nn
from torch.nn import functional as F
from vae import SimpleDecoder, SimpleEncoder, Reshape, Flatten


class DResBlock(nn.Module):
    def __init__(self, in_c, out_c, input_shape) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm((in_c, input_shape, input_shape)),
            nn.LeakyReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.LayerNorm((out_c, input_shape, input_shape)),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )
        self.skip = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_c, out_c, kernel_size=1)
        )

    def forward(self, X):
        return self.net(X) + self.skip(X)


class PrintShape(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, X):
        print('X', X.shape)
        return X


class GResBlock(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )
        self.skip = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=1)
        )

    def forward(self, X):
        return self.net(X) + self.skip(X)


class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            DResBlock(3, dim, 64),
            DResBlock(dim, dim * 2, 32),
            DResBlock(dim * 2, dim * 4, 16),
            DResBlock(dim * 4, dim * 8, 8),
            DResBlock(dim * 8, dim * 8, 4)
        )
        self.head = nn.Sequential(
            nn.Linear(dim * 8 * 4, 1)
        )

    def forward(self, X):
        X = self.backbone(X)
        X = X.reshape(X.shape[0], -1)
        return self.head(X)

class Generator(nn.Module):
    def __init__(self, d_image, dim=64) -> None:
        super().__init__()
        self.d_image = d_image
        self.noise_shape = [dim, d_image, d_image]
        self.net = torch.nn.Sequential(
            GResBlock(dim, 8 * dim),
            GResBlock(8 * dim, 8 * dim),
            GResBlock(8 * dim, 8 * dim),
            GResBlock(8 * dim, 8 * dim),
            nn.BatchNorm2d(8 * dim),
            nn.LeakyReLU(),
            nn.Conv2d(8 * dim, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, noise):
        out = self.net(noise)
        return out
    
    def generate(self, B, device):
        noise = torch.randn([B] + self.noise_shape, device=device)
        out = self.net(noise)
        return out


def init_weights(m):
    # from https://arxiv.org/pdf/1511.06434v2.pdf
    if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.Dropout2d, nn.GELU, nn.LeakyReLU, \
                      nn.Sigmoid, Reshape, nn.Sequential, nn.LazyBatchNorm2d, Flatten)):
        return
    torch.nn.init.normal_(m.weight, 0, 0.02)
    torch.nn.init.normal_(m.bias, 0, 0.02)


class SimpleGenerator(SimpleDecoder):
    def __init__(self, dim, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.postnet.__delitem__(1) # batchnorm
        self.prenet.apply(init_weights)
        self.net.apply(init_weights)
        self.postnet.apply(init_weights)
        self.dim = dim
    
    def generate(self, N, device):
        noise = torch.rand((N, self.dim), device=device) # uniform in https://arxiv.org/pdf/1511.06434v2.pdf
        return self.forward(noise)


class SimpleDiscriminator(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, C, 3, stride=2, padding=1), # 32
            nn.LayerNorm((C, 32, 32)),
            nn.GELU(),
            nn.Conv2d(C, 2 * C, 3, stride=2, padding=1), # 16
            nn.LayerNorm((C * 2, 16, 16)),
            nn.GELU(),
            nn.Conv2d(2 * C, 4 *  C, 3, stride=2, padding=1), # 8
            nn.LayerNorm((C * 4, 8, 8)),
            nn.GELU(),
            nn.Conv2d(4 * C, 8 * C, 3, stride=2, padding=1), # 4
            nn.LayerNorm((C * 8, 4, 4)),
            nn.GELU(),
            nn.Conv2d(8 * C, 16 * C, 3, stride=2, padding=1), # 2
            nn.LayerNorm((C * 16, 2, 2)),
            nn.GELU(),
            Flatten(),
            nn.Linear(4 * 16 * C, 1)
        )
        self.net.apply(init_weights)
    
    def forward(self, X):
        X = self.net(X)
        return X


if __name__ == "__main__":
    device = torch.device('cuda:0')
    generator = SimpleGenerator(100, C=64).to(device)
    print(generator)
    out = generator.generate(1, device)
    print(out.shape)
    print(sum(el.numel() for el in generator.parameters()) / 1e6, 'M params in gen')
    discriminator = SimpleDiscriminator(64).to(device)
    print(discriminator(out))
    print(sum(el.numel() for el in discriminator.parameters()) / 1e6, 'M params in dis')