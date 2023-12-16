import argparse
from typing import Any
import torchvision
from pathlib import Path
import torchvision
torchvision.disable_beta_transforms_warning()
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from model import Generator, Discriminator
from utils import *
import wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=64, type=int, help="batch size")
    parser.add_argument("--n_critic", default=5, type=int, help="critic iterations per generator iteration")
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--iterations", default=10000, type=int)
    parser.add_argument("--gdim", default=64, type=int, help="generator size multiplier")
    parser.add_argument("--ddim", default=64, type=int, help="discriminator size multiplier")
    parser.add_argument("--epoch_len", default=100, type=int)
    parser.add_argument("--out_path", default="saved", type=Path)
    parser.add_argument("--lambda_m", default=10, type=int)
    args = parser.parse_args()
    train_dataset = torchvision.datasets.ImageFolder("train_data",
                                               transform=v2.Compose([
                                                    v2.RandomHorizontalFlip(p=0.5),
                                                    to_tensor,
                                                    v2.Normalize(mean=means, std=stds),
                                                ]))
    test_dataset = torchvision.datasets.ImageFolder("test_data",
                                            transform=v2.Compose([
                                                to_tensor,
                                            ]))
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(64, args.gdim).to(device)
    discriminator = Discriminator(args.ddim).to(device)
    opt_g = torch.optim.RMSprop(generator.parameters(), args.lr)
    opt_d = torch.optim.RMSprop(discriminator.parameters(), args.lr)
    ids = torch.tensor([1] * args.bs + [0] * args.bs, device=device).float()
    criterion = torch.nn.BCEWithLogitsLoss()

    d_loss = 0
    g_loss = 0
    d_grad_norm = 0
    g_grad_norm = 0
    gradient_penalty_total = 0
    steps = 0

    wandb.init(
        project="anemo-gan",
        config=args
    )
    wandb.run.log_code(".")
    print(sum(el.numel() for el in generator.parameters()) / 1e6, 'M params in generator')
    print(sum(el.numel() for el in discriminator.parameters()) / 1e6, 'M params in discriminator')

    for iteration in tqdm(range(1, 1 + args.iterations)):
        steps += 1
        for i, (gt, _) in enumerate(train_dataloader):
            if i == args.n_critic:
                break
            opt_d.zero_grad()
            fake = generator(gt.shape[0], device)
            gt = gt.to(device)
            batch = torch.cat((gt, fake), dim=0)
            scores = discriminator(batch)
            loss = criterion(scores.flatten(), ids)
            d_loss += loss.item()
            loss.backward()
            d_grad_norm += calc_grad_norm(discriminator)
            
            lambdas = torch.rand(args.bs, device=device)
            new_data = fake * lambdas + (1 - lambdas) * gt
            
            # Modified from https://discuss.pytorch.org/t/gradient-penalty-with-respect-to-the-network-parameters/11944/5
            # Do not remove variable from here, even in torch>=2.0
            new_data = torch.autograd.Variable(new_data, requires_grad=True) 
            scores_interp = discriminator(new_data)

            gradients = torch.autograd.grad(outputs=scores_interp, inputs=new_data,
                                            grad_outputs=torch.ones_like(scores_interp).cuda(),
                                            retain_graph=True, create_graph=True, only_inputs=True)[0]
            gradients = gradients.reshape(gradients.shape[0],  -1)
            gradient_penalty = (gradients.norm(2, dim=-1) - 1).square().mean()

            gradient_penalty_total += gradient_penalty.item()
            (gradient_penalty * args.lambda_m).backward()
            opt_d.step()
        
        opt_g.zero_grad()
        fake = generator(args.bs, device)
        scores = discriminator(fake)
        loss = criterion(scores, torch.ones_like(scores))
        g_loss += loss.item()
        loss.backward()
        g_grad_norm += calc_grad_norm(generator)
        opt_g.step()

        if iteration % args.epoch_len == 0 or iteration == args.iterations:
            d_loss /= steps * args.n_critic
            g_loss /= steps
            d_grad_norm /= steps * args.n_critic
            g_grad_norm /= steps
            gradient_penalty_total /= steps * args.n_critic
            print(f"Iteration: {iteration}")
            base_path = args.out_path / f"iteration_{iteration}"
            checkpoint_path = base_path / "checkpoint.pth"
            image_path = base_path / "generated/images"
            image_path.mkdir(parents=True, exist_ok=True)
            generate_images(generator, len(test_dataset), image_path, device, bs=args.bs)
            generated_dataset = torchvision.datasets.ImageFolder(
                image_path.parent,
                transform=to_tensor
            )

            fid = compute_fid(generated_dataset, test_dataset)
            ssim = compute_ssim(generated_dataset, test_dataset)
                
            print(f"D loss: {d_loss:.4f}")
            print(f"G loss: {g_loss:.4f}")            
            print(f"D grad norm: {d_grad_norm:.4f}")
            print(f"G grad norm: {g_grad_norm:.4f}")
            print(f"D grad penalty: {gradient_penalty_total:.4f}")
            print(f"FID: {fid:.4f}")
            print(f"SSIM: {ssim:.4f}")
            table = wandb.Table(columns=["#", "img"])
            for i in range(10):
                table.add_data(i, wandb.Image(str(image_path / f"{i}.png")))

            wandb.log({
                "d_loss": d_loss,
                "g_loss_ce": g_loss,
                "g_loss_total": g_loss + gradient_penalty_total * args.lambda_m,
                "d_grad_norm": d_grad_norm,
                "g_grad_norm": g_grad_norm,
                "d_grad_penalty": gradient_penalty_total,
                "fid": fid,
                "ssim": ssim,
                "generated": table 
            })
            torch.save({
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "args": args,
                "g_opt": opt_g.state_dict(),
                "d_opt": opt_d.state_dict(),
                "iteration": iteration
            }, checkpoint_path)
            d_loss = 0
            g_loss = 0
            d_grad_norm = 0
            g_grad_norm = 0
            steps = 0
            gradient_penalty_total = 0