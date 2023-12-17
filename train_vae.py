import argparse
from typing import Any
import torchvision
from pathlib import Path
import torchvision
torchvision.disable_beta_transforms_warning()
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from vae import Vae, SimpleVae
from utils import *
import wandb
from datautils import ImageDataset
from torch import nn
from piq.feature_extractors.fid_inception import InceptionV3
from piq import SSIMLoss
from collections import defaultdict


class FIDLoss(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.inception = InceptionV3()
        self.inception.eval()
        self.mse = nn.MSELoss()
    
    def forward(self, X_gt, X_restored):
        X_gt = self.inception(X_gt)
        X_restored = self.inception(X_restored)
        loss = 0
        for gt, restored in zip(X_gt, X_restored):
            loss += self.mse(gt, restored)
        loss /= len(X_gt)
        return loss


class SuperLoss(nn.Module):
    def __init__(self, k_mse=1.0, k_fid=1.0, k_kl=1.0, k_ssim=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.fid = FIDLoss()
        self.k_mse = k_mse
        self.k_fid = k_fid
        self.k_kl = k_kl
        self.k_ssim = k_ssim
        self.ssim = SSIMLoss()

    def kl(self, means, logstds):
        # Check here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
        return 0.5 * (logvars.exp() + means.pow(2) - logvars).sum(dim=-1).mean()

    def forward(self, X_gt, X_restored, means, logvars):
        mse = self.mse(X_restored, X_gt)
        fid = self.fid(X_restored, X_gt)
        kl = self.kl(means, logvars)
        ssim = self.ssim(X_restored, X_gt)
        return {
            "loss": self.k_mse * mse + self.k_fid * fid + self.k_kl * kl + self.k_ssim * ssim,
            "kl-loss": kl,
            "fid-loss": fid,
            "mse-loss": mse,
            "ssim-loss": ssim
        }


class Averager:
    def __init__(self):
        self.total = 0
        self.cnt = 0
    
    def add(self, value, k=1):
        self.total += value
        self.cnt += k
    
    def get(self):
        return self.total / (self.cnt + 1e-7)

    def reset(self):
        self.total = self.cnt = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_n", default=10, type=int)
    parser.add_argument("--edim", default=64, type=int, help="encoder size multiplier")
    parser.add_argument("--ddim", default=64, type=int, help="decoder size multiplier")
    parser.add_argument("--dim", default=64, type=int, help="latent dim")
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--out_path", default="saved", type=Path)
    parser.add_argument("--random_erasing_p", default=0.0, type=float)
    parser.add_argument("--k_kl", default=.001, type=float)
    parser.add_argument("--k_mse", default=1.0, type=float)
    parser.add_argument("--k_fid", default=0.1, type=float)
    parser.add_argument("--k_ssim", default=0.1, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    args = parser.parse_args()
    train_dataset = ImageDataset("train_data/faces",
                                 image_augs=v2.Compose([
                                    v2.RandomHorizontalFlip(p=0.5),
                                 ]),
                                 tensor_augs=v2.Compose([
                                     v2.RandomErasing(p=args.random_erasing_p, scale=(0.02, 0.2))
                                 ]))
    
    test_dataset = ImageDataset("test_data/faces",
                                image_augs=v2.Compose([]),
                                 tensor_augs=v2.Compose([]))
    assert(len(test_dataset) == 1028)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleVae(args.dim, args.edim, args.ddim, args.dropout).to(device)
    model(torch.randn(1, 3, 64, 64, device=device))
    print(model)
    opt = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.gamma)
    criterion = SuperLoss(args.k_mse, args.k_fid, args.k_kl, args.k_ssim).to(device)
    averagers = defaultdict(Averager)

    wandb.init(
        project="anemo-gan",
        config=args
    )
    wandb.run.log_code(".")
    print(sum(el.numel() for el in model.encoder.parameters()) / 1e6, 'M params in encoder')
    print(sum(el.numel() for el in model.decoder.parameters()) / 1e6, 'M params in decoder')

    for epoch in tqdm(range(1, 1 + args.epochs)):
        model.train()
        for X_gt, X_aug in tqdm(train_dataloader, desc=f"epoch {epoch}"):
            opt.zero_grad()
            X_gt = X_gt.to(device)
            X_aug = X_aug.to(device)
            X_restored, means, logvars = model(X_aug)
            loss_dict = criterion(X_gt, X_restored, means, logvars)
            loss_dict['loss'].backward()
            averagers['grad_norm'].add(calc_grad_norm(model))
            for el, value in loss_dict.items():
                averagers[el].add(value.item())
            gt_images = X_gt[:10].detach().cpu()
            reconstructed_images = X_restored[:10].detach().cpu()
            opt.step()
        scheduler.step()

        gt_images = None
        reconstructed_images = None
        model.eval()
        with torch.no_grad():
            for X_gt, X_aug in tqdm(test_dataloader, desc=f"test epoch {epoch}"):
                X_gt = X_gt.to(device)
                X_aug = X_aug.to(device)
                X_restored, means, logvars = model(X_aug)
                loss_dict = criterion(X_gt, X_restored, means, logvars)
                for el, value in loss_dict.items():
                    averagers['test/' + el].add(value.item())
                if gt_images is None:
                    gt_images = X_gt[:10].detach().cpu()
                    reconstructed_images = X_restored[:10].detach().cpu()
        
        table = wandb.Table(columns=["#", "gt", "restored"])
        for i in range(gt_images.shape[0]):
            gt = to_tensor(gt_images[i]).permute(1, 2, 0).numpy()
            rec = to_tensor(reconstructed_images[i]).permute(1, 2, 0).numpy()
            table.add_data(i, wandb.Image(gt), wandb.Image(rec))
        
        out_dict = {}
        for el in averagers:
            out_dict[el] = averagers[el].get()
            averagers[el].reset()
        out_dict['reconstructed'] = table
        out_dict["lr"] = scheduler.get_last_lr()[0]

        if epoch % args.save_n == 0:
            base_path = args.out_path / f"epoch_{epoch}"
            checkpoint_path = base_path / "checkpoint.pth"
            image_path = base_path / "generated/images"
            image_path.mkdir(parents=True, exist_ok=True)
            generate_images_vae(model, len(test_dataset), image_path, device, bs=args.bs)
            generated_dataset = torchvision.datasets.ImageFolder(
                image_path.parent,
                transform=to_tensor
            )

            fid = compute_fid(generated_dataset, test_dataset)
            ssim = compute_ssim(generated_dataset, test_dataset)
            out_dict["fid"] = fid
            out_dict['ssim'] = ssim

            table = wandb.Table(columns=["#", "img"])
            for i in range(10):
                table.add_data(i, wandb.Image(str(image_path / f"{i}.png")))
            out_dict['generated'] = table
        
            torch.save({
                "model": model.state_dict(),
                "args": args,
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch
            }, checkpoint_path)
        
        for el, value in out_dict.items():
            if el == 'generated' or el == 'reconstructed':
                continue
            print(f"{el}:\t\t{value:.4}")
        wandb.log(out_dict)