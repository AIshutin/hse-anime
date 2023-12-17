from torch.utils.data import Dataset
import os
from PIL import Image
from utils import *

class ImageDataset(Dataset):
    def __init__(self, path, image_augs, tensor_augs, normalize=False):
        self.index = []
        for el in os.listdir(path):
            if '.png' in el:
                self.index.append(os.path.join(path, el))
        self.image_augs = image_augs
        self.tensor_augs = tensor_augs
        self.normalize = normalize
        assert(normalize is False)
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        img = Image.open(self.index[index])
        img = self.image_augs(img)
        normal_t = to_tensor(img)
        aug_t = self.tensor_augs(normal_t)
        return normal_t, aug_t
