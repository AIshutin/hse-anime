import torch
from torch import nn
from torch.nn import functional as F


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
    
    def forward(self, B, device):
        noise = torch.randn([B] + self.noise_shape, device=device)
        out = self.net(noise)
        return out


if __name__ == "__main__":
    device = torch.device('cuda:0')
    generator = Generator(64, 64).to(device)
    out = generator(1, device)
    print(out.shape)
    print(sum(el.numel() for el in generator.parameters()) / 1e6, 'M params in gen')
    discriminator = Discriminator(64).to(device)
    print(discriminator(out))
    print(sum(el.numel() for el in discriminator.parameters()) / 1e6, 'M params in dis')