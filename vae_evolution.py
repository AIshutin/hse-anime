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
    parser.add_argument("--checkpoint_dir", default="saved", type=Path)
    parser.add_argument("--gif", default="evolution.gif", type=Path)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleVae(args.dim, args.edim, args.ddim, 0).to(device)

    noise = torch.randn(9, args.dim, device=device)
    checkpoints = []
    for el in os.listdir(args.checkpoint_dir):
        if 'epoch_' not in el:
            continue
        n = int(el.replace('epoch_', ''))
        checkpoints.append((n, os.path.join(args.checkpoint_dir, el, 'checkpoint.pth')))
    checkpoints.sort()

    frames = []
    for n, checkpoint in checkpoints:
        with torch.no_grad():
            model.load_state_dict(torch.load(checkpoint)['model'])
            model.eval()    
            frame = model.decoder(noise).detach().cpu()
            # B, C, 64, 64
            frame = frame.transpose(0, 1)
            # C, B, 64, 64
            frame = frame.reshape(3, 3, 3, 64, 64)
            # C, 3, 3, 64, 64
            frame = frame.transpose(2, 3)
            # C, 3, 64, 3, 64
            frame = frame.reshape(3, 64 * 3, 64 * 3)
            frame = to_image(frame)
            frames.append(frame)
    frames[0].save(args.gif, format="GIF", append_images=frames[1:],
               save_all=True, duration=2, loop=False)

