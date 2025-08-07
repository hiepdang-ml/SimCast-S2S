from typing import *
import argparse
from torch.nn import DataParallel

from datasets.cesm2 import CESM2
from common.utils import CheckpointLoader
from workers import BaselinePredictor, VAEPredictor, DDPMPredictor
from common.configs import MetaData, CNNConfig, UnetConfig, ViTConfig, VAEContextConfig, VAETargetConfig, DDPMConfig
from models.benchmarks import CNN, UNet, ViT
from models.diffusion import (
    VAE, VAEEncoder, VAEDecoder, UNetDenoiser, 
    LinearNoiseScheduler, CosineNoiseScheduler, 
    DDPMForwardProcess, DDPMReverseProcess,
)


def main(model: Literal["cnn", "unet", "vit", "vae-context", "vae-target"]) -> None:

    # Dataset
    # TODO:
    # test_metadata: MetaData = MetaData(tp="train")
    # test_metadata: MetaData = MetaData(tp="val")
    test_metadata: MetaData = MetaData(tp="test")
    test_dataset: CESM2 = CESM2(metadata=test_metadata)

    # Model
    if model.lower() == "cnn":
        model_config: CNNConfig = CNNConfig()
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint)
        net: CNN = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, CNN)
        assert net.n_input_days == test_metadata.n_input_days

    elif model.lower() == "unet":
        model_config: UnetConfig = UnetConfig()
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint)
        net: UNet = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, UNet)
        assert net.n_input_days == test_metadata.n_input_days

    elif model.lower() == "vit":
        model_config: UnetConfig = ViTConfig()
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint)
        net: ViT = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, ViT)
        assert net.n_input_days == test_metadata.n_input_days

    elif model.lower() == "vae-context":
        model_config: UnetConfig = VAEContextConfig()
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint)
        net: VAE = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, VAE)
        assert net.pixel_dim == test_metadata.n_input_days * len(test_metadata.input_vars)

    elif model.lower() == "vae-target":
        model_config: UnetConfig = VAETargetConfig()
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint)
        net: VAE = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, VAE)
        assert net.pixel_dim == 1 * len(test_metadata.output_vars)

    elif model.lower() == "ddpm":
        model_config: DDPMConfig = DDPMConfig()
        # Denoiser
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.from_checkpoint)
        net: UNetDenoiser = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, UNetDenoiser)
        # Target encoder/decoder
        print(f"Loading target_encoder and target_decoder from {model_config.target_vae_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.target_vae_checkpoint)
        target_vae: VAE = checkpoint_loader.load(scope=globals())
        target_encoder: VAEEncoder = target_vae.encoder.to(device=model_config.device)
        target_decoder: VAEDecoder = target_vae.decoder.to(device=model_config.device)
        # Context encoder
        print(f"Loading context_encoder from {model_config.context_vae_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.context_vae_checkpoint)
        context_vae: VAE = checkpoint_loader.load(scope=globals())
        context_encoder: VAEEncoder = context_vae.encoder.to(device=model_config.device)
        # Noise scheduler
        if model_config.noise_scheduler_scheme.lower() == "linear":
            noise_scheduler = LinearNoiseScheduler(
                beta_min=model_config.beta_min,
                beta_max=model_config.beta_max,
                n_steps=model_config.n_steps,
                device=model_config.device,
            )
        elif model_config.noise_scheduler_scheme.lower() == "cosine":
            noise_scheduler = CosineNoiseScheduler(n_steps=model_config.n_steps, device=model_config.device)
        else:
            raise ValueError(f"Invalid noiser_scheduler_scheme in config {model_config.noise_scheduler_scheme}")

    else:
        raise NotImplementedError(f"Unknown model: {model}")

    if model.lower() in ["cnn", "unet", "vit"]:
        predictor = BaselinePredictor(net=net)
        predictor.predict(dataset=test_dataset)

    elif model.lower() == "vae-context":
        predictor = VAEPredictor(net=net, tp="context")
        predictor.predict(dataset=test_dataset)

    elif model.lower() == "vae-target":
        predictor = VAEPredictor(net=net, tp="target")
        predictor.predict(dataset=test_dataset)

    elif model.lower() == "ddpm":
        predictor = DDPMPredictor(
            denoiser=net, 
            target_encoder=target_encoder, target_decoder=target_decoder, 
            context_encoder=context_encoder, noise_scheduler=noise_scheduler,
        )
        predictor.predict(dataset=test_dataset)

    else:
        raise NotImplementedError(f"Unknown model: {model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=["cnn", "unet", "vit", "vae-context", "vae-target", "ddpm"], 
        required=True,
    )
    args: argparse.Namespace = parser.parse_args()
    main(model=args.model)

