import argparse
from typing import Any
import torchvision
from pathlib import Path
import torchvision
torchvision.disable_beta_transforms_warning()
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from wgan import SimpleGenerator, SimpleDiscriminator
from utils import *
import wandb
from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=128, type=int, help="batch size")
    parser.add_argument("--n_critic", default=5, type=int, help="critic iterations per generator iteration")
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--iterations", default=100000, type=int)
    parser.add_argument("--gdim", default=64, type=int, help="generator size multiplier")
    parser.add_argument("--ddim", default=64, type=int, help="discriminator size multiplier")
    parser.add_argument("--dim", default=100, type=int, help="discriminator size multiplier")
    parser.add_argument("--epoch_len", default=100, type=int)
    parser.add_argument("--out_path", default="saved_gan", type=Path)
    parser.add_argument("--lambda_m", default=10, type=int)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--ew", default=0.95, type=float)
    args = parser.parse_args()
    train_dataset = torchvision.datasets.ImageFolder("train_data",
                                               transform=v2.Compose([
                                                    v2.RandomHorizontalFlip(p=0.5),
                                                    to_tensor,
                                                ]))
    test_dataset = torchvision.datasets.ImageFolder("test_data",
                                            transform=v2.Compose([
                                                to_tensor,
                                            ]))
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = SimpleGenerator(args.dim, C=args.gdim).to(device)
    discriminator = SimpleDiscriminator(C=args.ddim).to(device)
    # old_discriminator = SimpleDiscriminator(C=args.ddim).to(device)
    # old_discriminator.load_state_dict(discriminator.state_dict())

    # from https://arxiv.org/pdf/1511.06434v2.pdf
    opt_g = torch.optim.Adam(generator.parameters(), args.lr, betas=[args.beta1, 0.999])
    opt_d = torch.optim.Adam(discriminator.parameters(), args.lr, betas=[args.beta1, 0.999])
    ids = torch.tensor([1] * args.bs + [0] * args.bs, device=device).float()
    criterion = torch.nn.BCEWithLogitsLoss()
    averagers = defaultdict(Averager)

    wandb.init(
        project="anemo-gan",
        config=args
    )
    wandb.run.log_code(".")

    discriminator(generator.generate(1, device))

    print(generator)
    print(discriminator)

    print(sum(el.numel() for el in generator.parameters()) / 1e6, 'M params in generator')
    print(sum(el.numel() for el in discriminator.parameters()) / 1e6, 'M params in discriminator')

    for iteration in tqdm(range(1, 1 + args.iterations)):
        for i, (gt, _) in enumerate(train_dataloader):
            if i == args.n_critic:
                break
            opt_d.zero_grad()
            fake = generator.generate(gt.shape[0], device)
            gt = gt.to(device)
            batch = torch.cat((gt, fake), dim=0)
            scores = discriminator(batch)
            loss = criterion(scores.flatten(), ids)
            averagers['d_loss'].add(loss.item())
            loss.backward()
            averagers['d_grad_norm'].add(calc_grad_norm(discriminator))
            lambdas = torch.rand(gt.shape[0], device=device).reshape(-1, 1, 1, 1)
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

            averagers['gradient_penalty'].add(gradient_penalty.item())
            (gradient_penalty * args.lambda_m).backward()
            opt_d.step()
        
        # old_state_dict = old_discriminator.state_dict()
        # for el, w in discriminator.state_dict().items():
        #    old_state_dict[el] = old_state_dict[el] * args.ew + (1 - args.ew) * w
        # old_discriminator.load_state_dict(old_state_dict)
        
        opt_g.zero_grad()
        fake = generator.generate(args.bs, device)
        new_scores = discriminator(fake)
        new_loss = criterion(new_scores, torch.ones_like(new_scores))
        # old_scores = old_discriminator(fake)
        # old_loss = criterion(old_scores, torch.ones_like(old_scores))
        loss = new_loss
        averagers['g_loss'].add(loss.item())
        averagers['new_loss'].add(new_loss.item())
        # averagers['old_loss'].add(old_loss.item())

        loss.backward()
        averagers['g_grad_norm'].add(calc_grad_norm(generator))
        opt_g.step()

        if iteration % args.epoch_len == 0 or iteration == args.iterations:
            print(f"Iteration: {iteration}")
            base_path = args.out_path / f"iteration_{iteration}"
            checkpoint_path = base_path / "checkpoint.pth"
            image_path = base_path / "generated/images"
            image_path.mkdir(parents=True, exist_ok=True)
            generate_images_vae(generator, len(test_dataset), image_path, device, bs=args.bs)
            generated_dataset = torchvision.datasets.ImageFolder(
                image_path.parent,
                transform=to_tensor
            )

            fid = compute_fid(generated_dataset, test_dataset)
            ssim = compute_ssim(generated_dataset, test_dataset)

            out_dict = {
                "fid": fid,
                "ssim": ssim,
            }
            for el in averagers:
                out_dict[el] = averagers[el].get()
                averagers[el].reset()
            
            table = wandb.Table(columns=["#", "img"])
            for i in range(10):
                table.add_data(i, wandb.Image(str(image_path / f"{i}.png")))
            out_dict['generated'] = table

            for el, value in out_dict.items():
                if el == 'generated' or el == 'reconstructed':
                    continue
                print(f"{el}:\t\t{value:.4}")
            
            wandb.log(out_dict)

            torch.save({
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
            #    "old_discriminator": old_discriminator.state_dict(),
                "args": args,
                "g_opt": opt_g.state_dict(),
                "d_opt": opt_d.state_dict(),
                "iteration": iteration
            }, checkpoint_path)
