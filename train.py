import argparse
from typing import *
from torch.nn import DataParallel

from datasets.cesm2 import CESM2
from common.utils import CheckpointLoader
from workers import BaselineTrainer, VAETrainer, DDPMTrainer
from common.configs import MetaData, CNNConfig, UnetConfig, ViTConfig, VAEContextConfig, VAETargetConfig, DDPMConfig
from models.benchmarks import CNN, UNet, ViT
from models.diffusion import (
    VAE, VAEEncoder, VAEDecoder, UNetDenoiser, 
    LinearNoiseScheduler, CosineNoiseScheduler, 
    DDPMForwardProcess, DDPMReverseProcess,
)


def main(
    model: Literal["cnn", "unet", "vit", "vae-context", "vae-target"],
    dataset: Literal["cesm2", "era5"]
) -> None:

    # Dataset
    train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
    val_metadata: MetaData = MetaData(dataset_name=dataset, tp="val")
    if dataset == "cesm2":
        train_dataset: CESM2 = CESM2(metadata=train_metadata)
        val_dataset: CESM2 = CESM2(metadata=val_metadata)
    # TODO
    else:
        ...

    # Model
    if model.lower() == "cnn":
        print("Training CNN")
        model_config: CNNConfig = CNNConfig()
        if (checkpoint_path := model_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path)
            net: CNN = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
            assert isinstance(net, CNN)
            assert net.n_input_days == train_metadata.n_input_days
        else:
            print("Training from scratch")
            net: CNN = CNN(
                n_input_days=train_metadata.n_input_days,
                in_features=model_config.in_features,
                out_features=model_config.out_features,
                embedding_dim=model_config.embedding_dim,
                n_hidden_layers=model_config.n_hidden_layers,
            ).to(device=model_config.device)
    
    elif model.lower() == "unet":
        print("Training UNet")
        model_config: UnetConfig = UnetConfig()
        if (checkpoint_path := model_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path)
            # TODL
            net: UNet = checkpoint_loader.load(scope=globals(), ignored_modules=["module.decoder.mlp.6", "module.decoder.mlp.6"]).to(device=model_config.device)
            assert isinstance(net, UNet)
            assert net.n_input_days == train_metadata.n_input_days
        else:
            print("Training from scratch")
            net: UNet = UNet(
                n_input_days=train_metadata.n_input_days,
                in_features=model_config.in_features,
                out_features=model_config.out_features,
                embedding_dim=model_config.embedding_dim,
            ).to(device=model_config.device)

    elif model.lower() == "vit":
        print("Training ViT")
        model_config: ViTConfig = ViTConfig()
        if (checkpoint_path := model_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path)
            net: ViT = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
            assert isinstance(net, ViT)
            assert net.n_input_days == train_metadata.n_input_days
        else:
            print("Training from scratch")
            net: ViT = ViT(
                n_input_days=train_metadata.n_input_days,
                in_features=model_config.in_features,
                out_features=model_config.out_features,
                embedding_dim=model_config.embedding_dim,
                patch_size=model_config.patch_size,
                n_heads=model_config.n_heads,
                n_transformer_layers=model_config.n_transformer_layers,
                dropout=model_config.dropout,
            ).to(device=model_config.device)

    elif model.lower() == "vae-context":
        print("Training VAE-Context")
        model_config: VAEContextConfig = VAEContextConfig()
        if (checkpoint_path := model_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path)
            net: VAE = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
            assert isinstance(net, VAE)
            assert net.pixel_dim == train_metadata.n_input_days * len(train_metadata.input_vars)
        else:
            print("Training from scratch")
            net: VAE = VAE(
                n_days=train_metadata.n_input_days,
                n_features=len(train_metadata.input_vars),
                latent_dim=model_config.latent_dim,
                hidden_dim=model_config.hidden_dim,
                n_scaling_blocks=model_config.n_scaling_blocks,
                n_convstack_layers=model_config.n_convstack_layers,
                n_convhead_layers=model_config.n_convhead_layers,
            ).to(device=model_config.device)

    elif model.lower() == "vae-target":
        print("Training VAE-Target")
        model_config: VAETargetConfig = VAETargetConfig()
        if (checkpoint_path := model_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path)
            net: VAE = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
            assert isinstance(net, VAE)
            assert net.pixel_dim == 1 * len(train_metadata.output_vars)
        else:
            print("Training from scratch")
            net: VAE = VAE(
                n_days=1,
                n_features=len(train_metadata.output_vars),
                latent_dim=model_config.latent_dim,
                hidden_dim=model_config.hidden_dim,
                n_scaling_blocks=model_config.n_scaling_blocks,
                n_convstack_layers=model_config.n_convstack_layers,
                n_convhead_layers=model_config.n_convhead_layers,
            ).to(device=model_config.device)

    elif model.lower() == "ddpm":
        print("Training DDPM")
        # Denoiser
        model_config: DDPMConfig = DDPMConfig()
        if (checkpoint_path := model_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path)
            net: UNetDenoiser = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
            assert isinstance(net, UNetDenoiser)
        else:
            print("Training UNetDenoiser from scratch")
            net: UNetDenoiser = UNetDenoiser(
                target_in_dim=model_config.target_in_dim,
                condition_in_dim=model_config.condition_in_dim,
                step_in_dim=model_config.step_in_dim,
                down_out_dims=model_config.down_out_dims,
                down_hidden_dims=model_config.down_hidden_dims,
                mid_out_dims=model_config.mid_out_dims,
                mid_hidden_dims=model_config.mid_hidden_dims,
                up_out_dims=model_config.up_out_dims,
                up_hidden_dims=model_config.up_hidden_dims,
                n_layers_per_scaling_block=model_config.n_layers_per_scaling_block,
                n_layers_per_mid_block=model_config.n_layers_per_mid_block,
                n_attention_heads=model_config.n_attention_heads,
                condition_dropout=model_config.condition_dropout,
            ).to(device=model_config.device)
        
        # Target encoder
        print(f"Loading target_encoder from {model_config.target_vae_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.target_vae_checkpoint)
        target_encoder: VAEEncoder = checkpoint_loader.load(scope=globals()).encoder.to(device=model_config.device)
        # Context encoder
        print(f"Loading context_encoder from {model_config.context_vae_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.context_vae_checkpoint)
        context_encoder: VAEEncoder = checkpoint_loader.load(scope=globals()).encoder.to(device=model_config.device)
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

    # Train
    if model.lower() in ["cnn", "unet", "vit"]:
        trainer = BaselineTrainer(
            net=net,
            lr=model_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=model_config.train_batch_size,
            val_batch_size=model_config.val_batch_size,
        )
        trainer.train(
            n_epochs=model_config.n_epochs,
            patience=model_config.patience,
            tolerance=model_config.tolerance,
            checkpoint_directory=model_config.saved_checkpoint_directory,
            save_frequency=model_config.save_frequency,
        )

    elif model.lower() == "vae-context":
        trainer = VAETrainer(
            net=net,
            tp="context",
            lambda_=model_config.lambda_,
            lr=model_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=model_config.train_batch_size,
            val_batch_size=model_config.val_batch_size,
        )
        trainer.train(
            n_epochs=model_config.n_epochs,
            patience=model_config.patience,
            tolerance=model_config.tolerance,
            checkpoint_directory=model_config.saved_checkpoint_directory,
            save_frequency=model_config.save_frequency,
        )

    elif model.lower() == "vae-target":
        trainer = VAETrainer(
            net=net,
            tp="target",
            lambda_=model_config.lambda_,
            lr=model_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=model_config.train_batch_size,
            val_batch_size=model_config.val_batch_size,
        )
        trainer.train(
            n_epochs=model_config.n_epochs,
            patience=model_config.patience,
            tolerance=model_config.tolerance,
            checkpoint_directory=model_config.saved_checkpoint_directory,
            save_frequency=model_config.save_frequency,
        )

    elif model.lower() == "ddpm":
        trainer = DDPMTrainer(
            denoiser=net,
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            noise_scheduler=noise_scheduler,
            lr=model_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=model_config.train_batch_size,
            val_batch_size=model_config.val_batch_size,
        )
        trainer.train(
            n_epochs=model_config.n_epochs,
            patience=model_config.patience,
            tolerance=model_config.tolerance,
            checkpoint_directory=model_config.saved_checkpoint_directory,
            save_frequency=model_config.save_frequency,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=["cnn", "unet", "vit", "vae-context", "vae-target", "ddpm"],
        required=True,
    )
    parser.add_argument(
        "--dataset", type=str, choices=["cesm2", "era5"],
        required=True,
    )
    args: argparse.Namespace = parser.parse_args()
    main(model=args.model, dataset=args.dataset)

