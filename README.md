# Anime-Faces

### Preparation

```shell
pip3 install -r requirements.txt
```

Download dataset https://www.kaggle.com/datasets/soumikrakshit/anime-faces/data and unzip it data folder. Run `python3 prepare_data.py`

## VAE 

### Training 

```shell
python3 train_vae.py --dim 128 --ddim 16 --edim 16 --bs 128 --lr 0.001 --save_n 5 --k_kl 0.001 --k_mse 10 --k_fid 0 --k_ssim 1 --gamma 0.965 --out_path vae
```

### Visualization

```shell
python3 vae_evolution.py --dim 128 --ddim 16 --edim 16 --gif vae.gif --checkpoint_dir vae
```

![Evolution gif](https://github.com/AIshutin/hse-anime/blob/master/vae.gif?raw=true "VAE is improving")

### Generation

```
gdown https://drive.google.com/file/d/1yYd8f__s5LUvGya7ic_Y6rrKHNPLLOsZ/view?usp=sharing --fuzzy -O vae.pth
python3 vae_generate.py --dim 128 --ddim 16 --edim 16 --checkpoint vae.pth
```

## WGAN-GP

Actually, it suffers from mode collapse and doesn't really work

You can try it with:

```shell
python3 train_gan.py
```