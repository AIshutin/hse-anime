import os
import random
from tqdm import tqdm
random.seed(0)
os.system('rm data/*.png')
os.system('mkdir train_data; mkdir train_data/faces')
os.system('mkdir test_data; mkdir test_data/faces')
for el in tqdm(sorted(os.listdir('data/data'))):
    if '.png' not in el:
        continue
    if random.randint(0, 5) == 0:
        os.system(f'cp data/data/{el} test_data/faces/{el}')
    else:
        os.system(f'cp data/data/{el} train_data/faces/{el}')