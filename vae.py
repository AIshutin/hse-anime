import torch
from torch import nn
from torch.nn import functional as F
import random


class Resblock(nn.Module):
    def __init__(self, C_in, C, dropout_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C, kernel_size=3, padding=1),
            nn.BatchNorm2d(C),
            nn.GELU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(C, C * 2, kernel_size=1),
            nn.BatchNorm2d(C * 2),
            nn.GELU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(C * 2, C, kernel_size=3, padding=1),
            nn.BatchNorm2d(C),
            nn.GELU(),
            nn.Dropout2d(dropout_p),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(C_in, C, kernel_size=1),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )
    def forward(self, X):
        return self.skip(X) + self.net(X)


class Flatten(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, X):
        return X.flatten(1)


class Encoder(nn.Module):
    def __init__(self, dim, C=64, dropout_p=0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            Resblock(3, C, dropout_p), 
            nn.MaxPool2d(2), #32x32
            Resblock(C, 2 * C, dropout_p),
            nn.MaxPool2d(2), #16x16
            Resblock(2 * C, 4 * C, dropout_p),
            nn.MaxPool2d(2), #8x8
            Resblock(4 * C, 8 * C, dropout_p),
            nn.MaxPool2d(2), # 4x4
            Resblock(8 * C, 8 * C, dropout_p),
            nn.MaxPool2d(2), # 2x2
            Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * 8 * C, 2 * dim)
        )
    
    def forward(self, X):
        X = self.net(X)
        X = self.head(X)
        means, logvars = X.chunk(2, dim=-1)
        return means, logvars


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = [-1] + list(shape)
    
    def forward(self, X):
        return X.reshape(self.shape)


class TransposeBlock(nn.Module):
    def __init__(self, C_in, C_out, dropout_p=0.1):
        super().__init__()
        self.transpose = nn.ConvTranspose2d(C_in, C_out, 3, stride=2, padding=1, output_padding=1)
        self.net = nn.Sequential(
            nn.BatchNorm2d(C_out),
            nn.GELU(),
            nn.Conv2d(C_out, C_out, 3, padding=1),
            nn.BatchNorm2d(C_out),
            nn.GELU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(C_out, C_out, 3, padding=1),
            nn.BatchNorm2d(C_out),
            nn.GELU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(C_out, C_out, 1),
            nn.BatchNorm2d(C_out),
            nn.GELU(),
            nn.Dropout2d(dropout_p),
        )
    
    def forward(self, X):
        out_shape = list(X.shape)
        out_shape[-1] *= 2
        out_shape[-2] *= 2
        X = self.transpose(X) #, output_size=out_shape)
        return X + self.net(X)


class Decoder(nn.Module):
    def __init__(self, dim, C=64, dropout_p=0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2 * 2 * C),
            Reshape(C, 2, 2),
            TransposeBlock(C, C * 8, dropout_p),
            TransposeBlock(8 * C, 4 * C, dropout_p),
            TransposeBlock(4 * C, 2 * C, dropout_p),
            TransposeBlock(2 * C, C, dropout_p),
            TransposeBlock(C, 3),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        X = self.net(X)
        return X


class Vae(nn.Module):
    def __init__(self, dim, C_enc, C_dec, dropout_p):
        super().__init__()
        self.encoder = Encoder(dim, C_enc, dropout_p)
        self.decoder = Decoder(dim, C_dec, dropout_p)
        self.dim = dim
    
    def forward(self, X):
        means, logvars = self.encoder(X)
        latent = self.reparametrization_trick(means, logvars)
        X_hat = self.decoder(latent)
        return X_hat, means, logvars

    def reparametrization_trick(self, means, logvars):
        noise = torch.randn_like(means)
        noise *= (0.5 * logvars).exp()
        noise += means
        return noise

    def generate(self, N, device):
        noise = torch.randn((N, self.dim), device=device)
        return self.decoder(noise)
    

class SimpleEncoder(nn.Module):
    def __init__(self, dim, C) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, C, 3, stride=2, padding=1), # 32
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            nn.Conv2d(C, 2 * C, 3, stride=2, padding=1), # 16
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            nn.Conv2d(2 * C, 4 *  C, 3, stride=2, padding=1), # 8
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            nn.Conv2d(4 * C, 8 * C, 3, stride=2, padding=1), # 4
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            nn.Conv2d(8 * C, 16 * C, 3, stride=2, padding=1), # 2
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            Flatten(),
        )
        self.head = nn.Linear(4 * 16 * C, 2 * dim)
    
    def forward(self, X):
        X = self.net(X)
        X = self.head(X)
        return  X.chunk(2, dim=-1)


class SimpleDecoder(nn.Module):
    def __init__(self, dim, C) -> None:
        super().__init__()
        self.prenet = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            Reshape(dim, 2, 2)
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(dim, C * 16, 3, stride=2, padding=1, output_padding=1),
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            nn.ConvTranspose2d(C * 16, C * 8, 3, stride=2, padding=1, output_padding=1),
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            nn.ConvTranspose2d(C * 8, C * 4, 3, stride=2, padding=1, output_padding=1),
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            nn.ConvTranspose2d(C * 4, C * 2, 3, stride=2, padding=1, output_padding=1),
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            nn.ConvTranspose2d(C * 2, C, 3, stride=2, padding=1, output_padding=1),
            nn.LazyBatchNorm2d(),
            nn.GELU(),
        )
        self.postnet = nn.Sequential(
            nn.Conv2d(C, 3, 3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid(),
        )
    
    def forward(self, X):
        X = self.prenet(X)
        X = self.net(X)
        X = self.postnet(X)
        return X


class SimpleVae(Vae):
    def __init__(self, dim, C_enc, C_dec, dropout_p):
        super().__init__(dim, 1, 1, 0.0)
        self.encoder = SimpleEncoder(dim, C_enc)
        self.decoder = SimpleDecoder(dim, C_dec)


if __name__ == "__main__":
    device = torch.device('cuda:0')
    model = SimpleVae(64, 64, 64, 0.1).to(device)
    model(torch.randn(1, 3, 64, 64, device=device))
    print(sum(el.numel() for el in model.encoder.parameters()) / 1e6, 'M params in encoder')
    print(sum(el.numel() for el in model.decoder.parameters()) / 1e6, 'M params in decoder')
    imgs = torch.randn(5, 3, 64, 64, device=device)
    X, means, stds = model(imgs)
    print(X.shape)