import os
from typing import Literal
import torch
import torch.distributed as dist

from datasets.cesm2 import CESM2
from common.utils import CheckpointLoader
from workers import BaselinePredictor, VAEPredictor, DiffusionPredictor
from common.configs import MetaData, CNNConfig, UnetConfig, ViTConfig, VAEConfig, DiffusionConfig
from models.benchmarks import CNN, UNet, ViT
from models.diffusion import (
    VAE, VAE_Wind, VAE_Geopotential, VAE_ThermalDynamic, VAE_Precipitation, 
    VAEEncoder, VAEDecoder, UNetDenoiser, 
    LinearNoiseScheduler, CosineNoiseScheduler, ForwardProcess, ReverseProcess,
)


def main(
    model: Literal[
        "cnn", "unet", "vit", 
        "vae-wind", "vae-geopotential", "vae-thermaldynamic", "vae-precipitation", "diffusion"
    ],
    dataset: Literal["cesm2", "era5"]
) -> None:

    # Dataset
    # TODO:
    test_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
    # test_metadata: MetaData = MetaData(dataset_name=dataset, tp="val")
    # test_metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
    if dataset == "cesm2":
        test_dataset: CESM2 = CESM2(metadata=test_metadata)
    else:
        # TODO
        ...

    # Model
    if model.lower() == "cnn":
        model_config: CNNConfig = CNNConfig()
        assert model_config.from_checkpoint is not None
        checkpoint_loader = CheckpointLoader(checkpoint_path=str(model_config.from_checkpoint))
        net: CNN = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, CNN)
        assert net.n_input_days == test_metadata.n_input_days
        BaselinePredictor(net=net, dataset=test_dataset).predict()

    elif model.lower() == "unet":
        model_config: UnetConfig = UnetConfig()
        checkpoint_loader = CheckpointLoader(checkpoint_path=str(model_config.from_checkpoint))
        net: UNet = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, UNet)
        assert net.n_input_days == test_metadata.n_input_days
        BaselinePredictor(net=net, dataset=test_dataset).predict()

    elif model.lower() == "vit":
        model_config: ViTConfig = ViTConfig()
        checkpoint_loader = CheckpointLoader(checkpoint_path=str(model_config.from_checkpoint))
        net: ViT = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, ViT)
        assert net.n_input_days == test_metadata.n_input_days
        BaselinePredictor(net=net, dataset=test_dataset).predict()

    elif model.lower() == "vae-wind":
        test_metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="wind")
        if dataset == "cesm2":
            test_dataset: CESM2 = CESM2(metadata=test_metadata)

        model_config: VAEConfig = VAEConfig(context_group="wind")
        checkpoint_loader = CheckpointLoader(checkpoint_path=str(model_config.from_checkpoint))
        net: VAE_Wind = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, VAE_Wind)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(net=net, dataset=test_dataset).predict()

    elif model.lower() == "vae-geopotential":
        test_metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="geopotential")
        if dataset == "cesm2":
            test_dataset: CESM2 = CESM2(metadata=test_metadata)
    
        model_config: VAEConfig = VAEConfig(context_group="geopotential")
        checkpoint_loader = CheckpointLoader(checkpoint_path=str(model_config.from_checkpoint))
        net: VAE_Geopotential = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, VAE_Geopotential)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(net=net, dataset=test_dataset).predict()

    elif model.lower() == "vae-thermaldynamic":
        test_metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="thermaldynamic")
        if dataset == "cesm2":
            test_dataset: CESM2 = CESM2(metadata=test_metadata)

        model_config: VAEConfig = VAEConfig(context_group="thermaldynamic")
        checkpoint_loader = CheckpointLoader(checkpoint_path=str(model_config.from_checkpoint))
        net: VAE_ThermalDynamic = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, VAE_ThermalDynamic)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(net=net, dataset=test_dataset).predict()

    elif model.lower() == "vae-precipitation":
        test_metadata: MetaData = MetaData(dataset_name=dataset, tp="test")
        test_metadata = test_metadata.with_var_subset(context_group="precipitation")
        if dataset == "cesm2":
            test_dataset: CESM2 = CESM2(metadata=test_metadata)

        model_config: VAEConfig = VAEConfig(context_group="precipitation")
        checkpoint_loader = CheckpointLoader(checkpoint_path=str(model_config.from_checkpoint))
        net: VAE_Precipitation = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, VAE_Precipitation)
        assert net.pixel_dim == len(test_metadata.input_vars)
        VAEPredictor(net=net, dataset=test_dataset).predict()

    elif model.lower() == "diffusion":
        model_config: DiffusionConfig = DiffusionConfig()
        # Denoiser
        print(f"Loading denoiser from {model_config.from_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=str(model_config.from_checkpoint))
        net: UNetDenoiser = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, UNetDenoiser)
        # Wind encoder
        print(f"Loading wind_encoder from {model_config.wind_vae_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=str(model_config.wind_vae_checkpoint))
        wind_vae: VAE = checkpoint_loader.load(scope=globals())
        wind_encoder: VAEEncoder = wind_vae.encoder.to(device=model_config.device)
        # Geopotential encoder
        print(f"Loading geopotential_encoder from {model_config.geopotential_vae_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=str(model_config.geopotential_vae_checkpoint))
        geopotential_vae: VAE = checkpoint_loader.load(scope=globals())
        geopotential_encoder: VAEEncoder = geopotential_vae.encoder.to(device=model_config.device)
        # Thermaldynamic encoder
        print(f"Loading thermaldynamic_encoder from {model_config.thermaldynamic_vae_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=str(model_config.thermaldynamic_vae_checkpoint))
        thermaldynamic_vae: VAE = checkpoint_loader.load(scope=globals())
        thermaldynamic_encoder: VAEEncoder = thermaldynamic_vae.encoder.to(device=model_config.device)
        # Precipitation encoder/decoder
        print(f"Loading precipitation_encoder from {model_config.precipitation_vae_checkpoint}")
        print(f"Loading precipitation_decoder from {model_config.precipitation_vae_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=str(model_config.precipitation_vae_checkpoint))
        precipitation_vae: VAE = checkpoint_loader.load(scope=globals())
        precipitation_encoder: VAEEncoder = precipitation_vae.encoder.to(device=model_config.device)
        precipitation_decoder: VAEEncoder = precipitation_vae.decoder.to(device=model_config.device)
        # Noise scheduler
        if model_config.noise_scheduler.lower() == "linear":
            noise_scheduler = LinearNoiseScheduler(
                n_steps=model_config.n_steps, 
                beta_min=model_config.beta_min, 
                beta_max=model_config.beta_max, 
                device=model_config.device,
            )
        elif model_config.noise_scheduler.lower() == "cosine":
            noise_scheduler = CosineNoiseScheduler(
                n_steps=model_config.n_steps,
                device=model_config.device,
            )
        else:
            raise ValueError(f"Invalid diffusion_config.noise_scheduler={model_config.noise_scheduler}")

        
        DiffusionPredictor(
            denoiser=net, 
            wind_encoder=wind_encoder, 
            geopotential_encoder=geopotential_encoder, thermaldynamic_encoder=thermaldynamic_encoder, 
            precipitation_encoder=precipitation_encoder, precipitation_decoder=precipitation_decoder,
            noise_scheduler=noise_scheduler,
            eta=model_config.eta,
            dataset=test_dataset,
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
            "vae-wind", "vae-geopotential", "vae-thermaldynamic", "vae-precipitation", 
            "diffusion"
        ],
        required=True,
    )
    parser.add_argument("--dataset", type=str, choices=["cesm2", "era5"], required=True)
    args: argparse.Namespace = parser.parse_args()

    local_rank: int = setup_ddp()
    try:
        main(model=args.model, dataset=args.dataset)
    finally:
        cleanup_ddp()

