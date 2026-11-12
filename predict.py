import os
from typing import Literal
import torch
import torch.distributed as dist

from datapipeline.dataset import CESM2, ERA5, MetaData
from common.utils import CheckpointLoader
from workers import BaselinePredictor, VAEPredictor, SimCastS2SPredictor
from common.configs import CNNConfig, UnetConfig, VAEConfig, DiffusionConfig
from models.baselines import CNN, UNet
from models.simcasts2s import (
    VAE, VAE_Wind, VAE_Mass, VAE_Thermal, VAE_Hydro, VAE_Precip, Denoiser,
    build_noise_scheduler,
)


def main(
    model: Literal[
        "cnn", "unet",
        "vae-wind", "vae-mass", "vae-thermal", "vae-hydro", "vae-precip",
        "simcast-s2s",
    ],
    dataset: Literal["cesm2", "era5"],
    local_rank: int,
    batch_size: int,
) -> None:

    # Dataset
    test_metadata: MetaData
    test_dataset: CESM2 | ERA5
    model_config: CNNConfig | UnetConfig | VAEConfig | DiffusionConfig
    net: torch.nn.Module
    test_metadata = MetaData(dataset_name=dataset, tp="test")
    if dataset == "cesm2":
        test_dataset = CESM2(metadata=test_metadata)
    else:
        test_dataset = ERA5(metadata=test_metadata)

    # Model
    if model.lower() == "cnn":
        model_config = CNNConfig()
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net = checkpoint_loader.load_as(CNN, scope=globals())
        assert isinstance(net, CNN)
        assert net.n_input_days == test_metadata.n_input_days
        BaselinePredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank, batch_size=batch_size,
        ).predict()

    elif model.lower() == "unet":
        model_config = UnetConfig()
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net = checkpoint_loader.load_as(UNet, scope=globals())
        assert isinstance(net, UNet)
        assert net.n_input_days == test_metadata.n_input_days
        BaselinePredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank, batch_size=batch_size,
        ).predict()

    elif model.lower() == "vae-wind":
        test_metadata = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="wind")
        if dataset == "cesm2":
            test_dataset = CESM2(metadata=test_metadata)
        else:
            test_dataset = ERA5(metadata=test_metadata)

        model_config = VAEConfig(context_group="wind")
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net = checkpoint_loader.load_as(VAE_Wind, scope=globals())
        assert isinstance(net, VAE_Wind)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank, batch_size=batch_size,
        ).predict()

    elif model.lower() == "vae-mass":
        test_metadata = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="mass")
        if dataset == "cesm2":
            test_dataset = CESM2(metadata=test_metadata)
        else:
            test_dataset = ERA5(metadata=test_metadata)

        model_config = VAEConfig(context_group="mass")
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net = checkpoint_loader.load_as(VAE_Mass, scope=globals())
        assert isinstance(net, VAE_Mass)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank, batch_size=batch_size,
        ).predict()

    elif model.lower() == "vae-thermal":
        test_metadata = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="thermal")
        if dataset == "cesm2":
            test_dataset = CESM2(metadata=test_metadata)
        else:
            test_dataset = ERA5(metadata=test_metadata)

        model_config = VAEConfig(context_group="thermal")
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net = checkpoint_loader.load_as(VAE_Thermal, scope=globals())
        assert isinstance(net, VAE_Thermal)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank, batch_size=batch_size,
        ).predict()

    elif model.lower() == "vae-hydro":
        test_metadata = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="hydro")
        if dataset == "cesm2":
            test_dataset = CESM2(metadata=test_metadata)
        else:
            test_dataset = ERA5(metadata=test_metadata)

        model_config = VAEConfig(context_group="hydro")
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net = checkpoint_loader.load_as(VAE_Hydro, scope=globals())
        assert isinstance(net, VAE_Hydro)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank, batch_size=batch_size,
        ).predict()

    elif model.lower() == "vae-precip":
        test_metadata = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="precip")
        if dataset == "cesm2":
            test_dataset = CESM2(metadata=test_metadata)
        else:
            test_dataset = ERA5(metadata=test_metadata)

        model_config = VAEConfig(context_group="precip")
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net = checkpoint_loader.load_as(VAE_Precip, scope=globals())
        assert isinstance(net, VAE_Precip)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(
            net=net, dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank, batch_size=batch_size,
        ).predict()

    elif model.lower() == "simcast-s2s":
        model_config = DiffusionConfig()
        # Denoiser
        assert model_config.from_checkpoint is not None
        print(f"Loading denoiser from {model_config.from_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint.as_posix())
        net = checkpoint_loader.load_as(Denoiser, scope=globals())
        assert isinstance(net, Denoiser)
        noise_scheduler = build_noise_scheduler(
            checkpoint_loader.scheduler_configs["diffusion_schedule"],
        )
        if net.is_finetuning:
            net.freeze_backbone()
            assert net.is_backbone_frozen()
        # Wind encoder
        print(f"Loading wind_encoder from {model_config.vae_wind_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.vae_wind_checkpoint.as_posix())
        wind_vae: VAE = checkpoint_loader.load_as(VAE_Wind, scope=globals())
        # Mass encoder
        print(f"Loading mass_encoder from {model_config.vae_mass_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.vae_mass_checkpoint.as_posix())
        mass_vae: VAE = checkpoint_loader.load_as(VAE_Mass, scope=globals())
        # Thermal encoder
        print(f"Loading thermal_encoder from {model_config.vae_thermal_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.vae_thermal_checkpoint.as_posix())
        thermal_vae: VAE = checkpoint_loader.load_as(VAE_Thermal, scope=globals())
        # Hydro encoder
        print(f"Loading hydro_encoder from {model_config.vae_hydro_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.vae_hydro_checkpoint.as_posix())
        hydro_vae: VAE = checkpoint_loader.load_as(VAE_Hydro, scope=globals())
        # Precip encoder
        print(f"Loading precip_encoder from {model_config.vae_precip_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.vae_precip_checkpoint.as_posix())
        precip_vae: VAE = checkpoint_loader.load_as(VAE_Precip, scope=globals())
        SimCastS2SPredictor(
            denoiser=net,
            wind_encoder=wind_vae.encoder, mass_encoder=mass_vae.encoder,
            thermal_encoder=thermal_vae.encoder, hydro_encoder=hydro_vae.encoder,
            precip_encoder=precip_vae.encoder, precip_decoder=precip_vae.decoder,
            noise_scheduler=noise_scheduler,
            inference_steps=model_config.inference_steps,
            eta=model_config.eta,
            guidance_scale=model_config.guidance_scale,
            ensemble_size=model_config.ensemble_size,
            dataset=test_dataset,
            landmask_path=test_metadata.landmask_path.as_posix(),
            target_path=model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix(),
            local_rank=local_rank, batch_size=batch_size,
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
            "cnn", "unet",
            "vae-wind", "vae-mass", "vae-thermal", "vae-hydro", "vae-precip",
            "simcast-s2s"
        ],
        required=True,
    )
    parser.add_argument("--dataset", type=str, choices=["cesm2", "era5"], required=True)
    parser.add_argument(
        "--batch-size", type=int, required=True,
        help="Number of forecast samples per GPU inference batch",
    )
    args: argparse.Namespace = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    local_rank: int = setup_ddp()
    try:
        main(model=args.model, dataset=args.dataset, local_rank=local_rank, batch_size=args.batch_size)
    finally:
        cleanup_ddp()
