import os
from typing import Literal
import torch
import torch.distributed as dist

from datasets import CESM2
from common.utils import CheckpointLoader
from workers import BaselineTrainer, VAETrainer, DiffusionTrainer
from common.configs import MetaData, CNNConfig, UnetConfig, ViTConfig, VAEConfig, DiffusionConfig
from models.benchmarks import CNN, UNet, ViT
from models.diffusion import (
    VAE, VAE_Wind, VAE_Mass, VAE_Thermal, VAE_Hydro, VAE_Precip,
    VAEEncoder, VAEDecoder, UNetDenoiser, 
    LinearNoiseScheduler, CosineNoiseScheduler, ForwardProcess, ReverseProcess,
)


# def main(
#     model: Literal[
#         "cnn", "unet", "vit", 
#         "vae-wind", "vae-mass", "vae-thermal", "vae-hydro", "vae-precip",
#         "diffusion",
#     ],
# ) -> None:
#     if model.lower() == "cnn":
#         cnn_config: CNNConfig = CNNConfig()
#     elif model.lower() == "unet":
#         unet_config: UnetConfig = UnetConfig()
#     elif model.lower() == "vit":
#         vit_config: ViTConfig = ViTConfig()
#     elif model.lower() == "vae-wind":
#         vae_config: VAE_Wind = VAE_Wind()
#     elif model.lower() == "vae-mass":
#         vae_config: VAE_Mass = VAE_Mass()
#     elif model.lower() == "vae-thermal":
#         vae_config: VAE_Thermal = VAE_Thermal()
#     elif model.lower() == "vae-hydro":
#         vae_config: VAE_Hydro = VAE_Hydro()
#     elif model.lower() == "vae-precip":
#         vae_config: VAE_Precip = VAE_Precip()
#     elif model.lower() == "diffusion":
#         vae_config: VAE_Precip = DiffusionConfig()
    

from common.utils import TorchDictIO
torchio = TorchDictIO(dirpath="/scratch/zgp2ps/s2s_results/diffusion_v23_cosine/tensors/")
from workers.predictors import Visualizer
v = Visualizer("/scratch/zgp2ps/s2s_results/diffusion_v23_cosine/tensors/", "/scratch/zgp2ps/s2s_results/diffusion_v23_cosine/tensors/")

for f in sorted(v.source_dir.glob("*")):
    print(f)
    v.plot_diffusion_prediction(f.name)


# from common.utils import TorchDictIO
# torchio = TorchDictIO(dirpath="/scratch/zgp2ps/s2s_results/unet256/tensors/")
# from workers.predictors import Visualizer
# v = Visualizer("/scratch/zgp2ps/s2s_results/unet256/tensors/", "/scratch/zgp2ps/s2s_results/unet256/plots/")

# for f in sorted(v.source_dir.glob("*")):
#     print(f)
#     v.plot_baseline_prediction(f.name)


# from common.utils import TorchDictIO
# torchio = TorchDictIO(dirpath="/scratch/zgp2ps/s2s_results/cnn/tensors/")
# from workers.predictors import Visualizer
# v = Visualizer("/scratch/zgp2ps/s2s_results/cnn/tensors/", "/scratch/zgp2ps/s2s_results/cnn/plots/")

# for f in sorted(v.source_dir.glob("*")):
#     print(f)
#     v.plot_baseline_prediction(f.name)


# from common.utils import TorchDictIO
# torchio = TorchDictIO(dirpath="/scratch/zgp2ps/s2s_results/vit/tensors/")
# from workers.predictors import Visualizer
# v = Visualizer("/scratch/zgp2ps/s2s_results/vit/tensors/", "/scratch/zgp2ps/s2s_results/vit/plots/")

# for f in sorted(v.source_dir.glob("*")):
#     print(f)
#     v.plot_baseline_prediction(f.name)

