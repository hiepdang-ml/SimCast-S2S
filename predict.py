import os
from typing import Literal
import torch
import torch.distributed as dist

from datapipeline.dataset import CESM2, ERA5, metadata
from common.utils import CheckpointLoader
from workers import BaselinePredictor, VAEPredictor, DiffusionPredictor
from common.configs import MetaData, CNNConfig, UnetConfig, ViTConfig, VAEConfig, DiffusionConfig
from models.benchmarks import CNN, UNet, ViT
from models.diffusion import (
    VAE, VAE_Wind, VAE_Mass, VAE_Thermal, VAE_Hydro, VAE_Precip, UNetDenoiser,
    LinearNoiseScheduler, CosineNoiseScheduler,
)


def main(
    model: Literal[
        "cnn", "unet", "vit",
        "vae-wind", "vae-mass", "vae-thermal", "vae-hydro", "vae-precip",
        "diffusion",
    ],
    dataset: Literal["cesm2", "era5"],
    local_rank: int,
) -> None:

    # Dataset
    test_metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
    if dataset == "cesm2":
        test_dataset: CESM2 = CESM2(metadata=test_metadata)
    else:
        test_dataset: ERA5 = ERA5(metadata=test_metadata)

    # Model
    if model.lower() == "cnn":
        model_config: CNNConfig = CNNConfig()
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net: CNN = checkpoint_loader.load(scope=globals())
        assert isinstance(net, CNN)
        assert net.n_input_days == test_metadata.n_input_days
        BaselinePredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank,
        ).predict()

    elif model.lower() == "unet":
        model_config: UnetConfig = UnetConfig()
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net: UNet = checkpoint_loader.load(scope=globals())
        assert isinstance(net, UNet)
        assert net.n_input_days == test_metadata.n_input_days
        BaselinePredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank,
        ).predict()

    elif model.lower() == "vit":
        model_config: ViTConfig = ViTConfig()
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net: ViT = checkpoint_loader.load(scope=globals())
        assert isinstance(net, ViT)
        assert net.n_input_days == test_metadata.n_input_days
        BaselinePredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank,
        ).predict()

    elif model.lower() == "vae-wind":
        test_metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="wind")
        if dataset == "cesm2":
            test_dataset: CESM2 = CESM2(metadata=test_metadata)

        model_config: VAEConfig = VAEConfig(context_group="wind")
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net: VAE_Wind = checkpoint_loader.load(scope=globals())
        assert isinstance(net, VAE_Wind)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank,
        ).predict()

    elif model.lower() == "vae-mass":
        test_metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="mass")
        if dataset == "cesm2":
            test_dataset: CESM2 = CESM2(metadata=test_metadata)

        model_config: VAEConfig = VAEConfig(context_group="mass")
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net: VAE_Mass = checkpoint_loader.load(scope=globals())
        assert isinstance(net, VAE_Mass)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank,
        ).predict()

    elif model.lower() == "vae-thermal":
        test_metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="thermal")
        if dataset == "cesm2":
            test_dataset: CESM2 = CESM2(metadata=test_metadata)

        model_config: VAEConfig = VAEConfig(context_group="thermal")
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net: VAE_Thermal = checkpoint_loader.load(scope=globals())
        assert isinstance(net, VAE_Thermal)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank,
        ).predict()

    elif model.lower() == "vae-hydro":
        test_metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="hydro")
        if dataset == "cesm2":
            test_dataset: CESM2 = CESM2(metadata=test_metadata)

        model_config: VAEConfig = VAEConfig(context_group="hydro")
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net: VAE_Hydro = checkpoint_loader.load(scope=globals())
        assert isinstance(net, VAE_Hydro)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank,
        ).predict()

    elif model.lower() == "vae-precip":
        test_metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="precip")
        if dataset == "cesm2":
            test_dataset: CESM2 = CESM2(metadata=test_metadata)

        model_config: VAEConfig = VAEConfig(context_group="precip")
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net: VAE_Precip = checkpoint_loader.load(scope=globals())
        assert isinstance(net, VAE_Precip)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank,
        ).predict()

    elif model.lower() == "diffusion":
        model_config: DiffusionConfig = DiffusionConfig()
        # Denoiser
        assert model_config.from_checkpoint is not None
        print(f"Loading denoiser from {model_config.from_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net: UNetDenoiser = checkpoint_loader.load(scope=globals())
        assert isinstance(net, UNetDenoiser)
        # Wind encoder
        print(f"Loading wind_encoder from {model_config.vae_wind_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.vae_wind_checkpoint.as_posix())
        wind_vae: VAE = checkpoint_loader.load(scope=globals())
        # Mass encoder
        print(f"Loading mass_encoder from {model_config.vae_mass_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.vae_mass_checkpoint.as_posix())
        mass_vae: VAE = checkpoint_loader.load(scope=globals())
        # Thermal encoder
        print(f"Loading thermal_encoder from {model_config.vae_thermal_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.vae_thermal_checkpoint.as_posix())
        thermal_vae: VAE = checkpoint_loader.load(scope=globals())
        # Hydro encoder
        print(f"Loading hydro_encoder from {model_config.vae_hydro_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.vae_hydro_checkpoint.as_posix())
        hydro_vae: VAE = checkpoint_loader.load(scope=globals())
        # Precip encoder
        print(f"Loading precip_encoder from {model_config.vae_precip_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.vae_precip_checkpoint.as_posix())
        precip_vae: VAE = checkpoint_loader.load(scope=globals())
        # Noise scheduler
        if model_config.noise_scheduler.lower() == "linear":
            noise_scheduler = LinearNoiseScheduler(
                n_steps=model_config.n_steps,
                beta_min=model_config.beta_min,
                beta_max=model_config.beta_max,
            )
        elif model_config.noise_scheduler.lower() == "cosine":
            noise_scheduler = CosineNoiseScheduler(n_steps=model_config.n_steps)
        else:
            raise ValueError(f"Invalid diffusion_config.noise_scheduler={model_config.noise_scheduler}")

        DiffusionPredictor(
            denoiser=net,
            wind_encoder=wind_vae.encoder, mass_encoder=mass_vae.encoder,
            thermal_encoder=thermal_vae.encoder, hydro_encoder=hydro_vae.encoder,
            precip_encoder=precip_vae.encoder, precip_decoder=precip_vae.decoder,
            noise_scheduler=noise_scheduler,
            eta=model_config.eta,
            ensemble_size=model_config.ensemble_size,
            dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank,
        ).predict()

    else:
        raise NotImplementedError(f"Unknown model: {model}")


if __name__ == "__main__":
    import argparse
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    def setup_ddp() -> int:
        assert "RANK" in os.environ
        assert "WORLD_SIZE" in os.environ
        local_rank: int = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device=local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank

    def cleanup_ddp() -> None:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str, choices=[
            "cnn", "unet", "vit",
            "vae-wind", "vae-mass", "vae-thermal", "vae-hydro", "vae-precip", "vae-target",
            "diffusion"
        ],
        required=True,
    )
    parser.add_argument("--dataset", type=str, choices=["cesm2", "era5"], required=True)
    args: argparse.Namespace = parser.parse_args()

    local_rank: int = setup_ddp()
    try:
        main(model=args.model, dataset=args.dataset, local_rank=local_rank)
    finally:
        cleanup_ddp()
