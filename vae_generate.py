from vae import SimpleVae
import torch
import argparse
from pathlib import Path
import os
from datautils import to_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edim", default=64, type=int, help="encoder size multiplier")
    parser.add_argument("--ddim", default=64, type=int, help="decoder size multiplier")
    parser.add_argument("--dim", default=64, type=int, help="latent dim")
    parser.add_argument("--checkpoint", default="vae.pth", type=Path)
    parser.add_argument("--img_dir", default="generated/", type=Path)
    parser.add_argument("--N", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleVae(args.dim, args.edim, args.ddim, 0).to(device)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.eval()    
    args.img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 1 + args.N):
        with torch.no_grad():
            noise = torch.randn(1, args.dim, device=device)
            frame = model.decoder(noise).detach().cpu()[0]
            frame = to_image(frame)
            frame.save(args.img_dir / f"{i}.png")

