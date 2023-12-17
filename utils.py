from torchvision.transforms import v2
import torch
import os
from torch.utils.data import DataLoader
from piq import FID, ssim

means = [0.] * 3 # [0.7007, 0.6006, 0.5895]
stds = [1.] * 3 #[0.2938, 0.2973, 0.2702]

to_tensor = v2.Compose([
    v2.ToImageTensor(), 
    v2.ConvertImageDtype(),
#    v2.ToDtype(torch.float32),
])



denormalize = v2.Compose([])
'''
    v2.Normalize(mean =[0.] * 3, std=[1/el for el in stds]), 
    v2.Normalize(mean =[-el for el in means], std=[1.] * 3),
])
'''

to_image = v2.Compose([
    v2.ToImagePIL()
])


def calc_grad_norm(model):
    return sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5


def generate_images_vae(model, N, output_path, device, bs=64):
    model.eval()
    with torch.no_grad():
        cnt = 0
        while cnt < N:
            if N - cnt < bs:
                bs = N - cnt
            x = model.generate(bs, device).cpu()
            for i in range(x.shape[0]):
                image = to_image(x[i])
                image.save(os.path.join(output_path, f"{cnt + i}.png"))
            cnt += bs


def piq_collater(X):
    return {
        "images": torch.cat([el[0].unsqueeze(0) for el in X], dim=0)
    }


def compute_fid(dataset1, dataset2):
    with torch.no_grad():
        dataloader1 = DataLoader(dataset1, 16, collate_fn=piq_collater)
        dataloader2 = DataLoader(dataset2, 16, collate_fn=piq_collater)
        fid_metric = FID()
        first_feats = fid_metric.compute_feats(dataloader1)
        second_feats = fid_metric.compute_feats(dataloader2)
        fid = fid_metric(first_feats, second_feats).item()
        return fid


def compute_ssim(dataset1, dataset2):
    with torch.no_grad():
        dataloader1 = DataLoader(dataset1, 1)
        dataloader2 = DataLoader(dataset2, 1)
        total_ssim = 0
        cnt = 0
        for b1, b2 in zip(dataloader1, dataloader2):
            total_ssim += ssim(b1[0], b2[0])
            cnt += 1
        return total_ssim / cnt


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
    from PIL import Image
    totals = torch.zeros(3, dtype=torch.float)
    cnt = 0
    for el in os.listdir('data/data'):
        init_tensor = to_tensor(Image.open(f'data/data/{el}'))
        assert(init_tensor.shape[0] == 3)
        totals += init_tensor.mean(dim=(-1, -2))
        cnt += 1
    means = totals / cnt
    print(means, 'means')
    stds = torch.zeros(3, dtype=torch.float)
    cnt = 0
    for el in os.listdir('data/data'):
        init_tensor = to_tensor(Image.open(f'data/data/{el}'))
        init_tensor -= means.reshape(3, 1, 1)
        stds += init_tensor.square().sum(dim=(-1, -2))
        cnt += init_tensor.shape[-1] * init_tensor.shape[-2]
    stds /= cnt
    stds = stds.sqrt()
    print(stds, 'stds')
    img_path = "test_data/faces/8.png"
    init_tensor = to_tensor(Image.open(img_path))
    normalize = v2.Normalize(mean=means, std=stds)
    tensor = normalize(init_tensor)
    print(init_tensor.max(), init_tensor.min())
    print(tensor.min(), tensor.max(), tensor.mean())
    to_image(denormalize(tensor)).save('test8.png')