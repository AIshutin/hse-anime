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
