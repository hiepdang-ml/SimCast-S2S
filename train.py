import os
from typing import Literal
import torch
import torch.distributed as dist

from datapipeline.dataset import CESM2, ERA5
from common.utils import CheckpointLoader
from workers import BaselineTrainer, VAETrainer, DiffusionTrainer
from common.configs import MetaData, CNNConfig, UnetConfig, ViTConfig, VAEConfig, DiffusionConfig
from models.benchmarks import CNN, UNet, ViT
from models.diffusion import (
    VAE_Wind, VAE_Mass, VAE_Thermal, VAE_Hydro, VAE_Precip, VAEEncoder, UNetDenoiser,
    LinearNoiseScheduler, CosineNoiseScheduler
)


def main(
    model: Literal[
        "cnn", "unet", "vit",
        "vae-wind", "vae-mass", "vae-thermal", "vae-hydro", "vae-precip",
        "diffusion",
    ],
    dataset: Literal["cesm2", "era5"],
    is_finetuning: bool, lora_rank: int,
    local_rank: int,
) -> None:

    if is_finetuning and lora_rank <= 0:
        raise ValueError(f"lora_rank must be > 0, got {lora_rank}")

    # Dataset
    train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
    val_metadata: MetaData = MetaData(dataset_name=dataset, tp="val")
    if dataset == "cesm2":
        train_dataset: CESM2 = CESM2(metadata=train_metadata)
        val_dataset: CESM2 = CESM2(metadata=val_metadata)
    else:
        train_dataset: ERA5 = ERA5(metadata=train_metadata)
        val_dataset: ERA5 = ERA5(metadata=val_metadata)

    # Model
    if model.lower() == "cnn":
        print("Training CNN")
        cnn_config: CNNConfig = CNNConfig()
        if (checkpoint_path := cnn_config.from_checkpoint) is not None:
            print(f"Training CNN from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: CNN = checkpoint_loader.load(
                scope=globals(),
                is_finetuning=is_finetuning,
                lora_rank=lora_rank,
            )
            assert isinstance(net, CNN)
            assert net.n_input_days == train_metadata.n_input_days
        else:
            assert not is_finetuning
            print(f"Training CNN from scratch with: {cnn_config.to_dict()}")
            net: CNN = CNN(
                n_input_days=train_metadata.n_input_days,
                n_output_days=train_metadata.n_output_days,
                in_features=cnn_config.in_features,
                out_features=cnn_config.out_features,
                embedding_dim=cnn_config.embedding_dim,
                n_hidden_layers=cnn_config.n_hidden_layers,
                is_finetuning=False,
                lora_rank=0,
            )
            assert isinstance(net, CNN)    # for type checking only

        if is_finetuning:
            net.freeze_backbone()
            assert net.is_backbone_frozen()

        trainer = BaselineTrainer(
            net=net,
            lr=cnn_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=cnn_config.train_batch_size,
            val_batch_size=cnn_config.val_batch_size,
            local_rank=local_rank,
        )
        trainer.train(
            n_epochs=cnn_config.n_epochs,
            patience=cnn_config.patience,
            tolerance=cnn_config.tolerance,
            checkpoint_directory=cnn_config.saved_checkpoint_directory.as_posix(),
            save_frequency=cnn_config.save_frequency,
        )

    elif model.lower() == "unet":
        print("Training UNet")
        unet_config: UnetConfig = UnetConfig()
        if (checkpoint_path := unet_config.from_checkpoint) is not None:
            print(f"Training UNet from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: UNet = checkpoint_loader.load(
                scope=globals(),
                is_finetuning=is_finetuning,
                lora_rank=lora_rank,
            )
            assert isinstance(net, UNet)
            assert net.n_input_days == train_metadata.n_input_days
        else:
            assert not is_finetuning
            print(f"Training UNet from scratch with: {unet_config.to_dict()}")
            net: UNet = UNet(
                n_input_days=train_metadata.n_input_days,
                n_output_days=train_metadata.n_output_days,
                in_features=unet_config.in_features,
                out_features=unet_config.out_features,
                embedding_dim=unet_config.embedding_dim,
                is_finetuning=False,
                lora_rank=0,
            )
            assert isinstance(net, UNet)    # for type checking only

        if is_finetuning:
            net.freeze_backbone()
            assert net.is_backbone_frozen()

        trainer = BaselineTrainer(
            net=net,
            lr=unet_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=unet_config.train_batch_size,
            val_batch_size=unet_config.val_batch_size,
            local_rank=local_rank,
        )
        trainer.train(
            n_epochs=unet_config.n_epochs,
            patience=unet_config.patience,
            tolerance=unet_config.tolerance,
            checkpoint_directory=unet_config.saved_checkpoint_directory.as_posix(),
            save_frequency=unet_config.save_frequency,
        )

    elif model.lower() == "vit":
        print("Training ViT")
        vit_config: ViTConfig = ViTConfig()
        if (checkpoint_path := vit_config.from_checkpoint) is not None:
            print(f"Training ViT from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: ViT = checkpoint_loader.load(scope=globals())
            assert net.n_input_days == train_metadata.n_input_days
        else:
            print(f"Training ViT from scratch with: {vit_config.to_dict()}")
            net: ViT = ViT(
                n_input_days=train_metadata.n_input_days,
                n_output_days=train_metadata.n_output_days,
                in_features=vit_config.in_features,
                out_features=vit_config.out_features,
                embedding_dim=vit_config.embedding_dim,
                patch_size=vit_config.patch_size,
                n_heads=vit_config.n_heads,
                n_transformer_layers=vit_config.n_transformer_layers,
                dropout=vit_config.dropout,
            )

        assert isinstance(net, ViT)
        trainer = BaselineTrainer(
            net=net,
            lr=vit_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=vit_config.train_batch_size,
            val_batch_size=vit_config.val_batch_size,
            local_rank=local_rank,
        )
        trainer.train(
            n_epochs=vit_config.n_epochs,
            patience=vit_config.patience,
            tolerance=vit_config.tolerance,
            checkpoint_directory=vit_config.saved_checkpoint_directory.as_posix(),
            save_frequency=vit_config.save_frequency,
        )

    elif model.lower() == "vae-wind":
        vae_config: VAEConfig = VAEConfig(context_group="wind")
        assert vae_config.context_group == "wind"
        train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
        val_metadata: MetaData = MetaData(dataset_name=dataset, tp="val")
        if dataset == "cesm2":
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: CESM2 = CESM2(metadata=train_metadata)
            val_dataset: CESM2 = CESM2(metadata=val_metadata)
        else:
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: ERA5 = ERA5(metadata=train_metadata)
            val_dataset: ERA5 = ERA5(metadata=val_metadata)

        if (checkpoint_path := vae_config.from_checkpoint) is not None:
            print(f"Training VAE_Wind from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: VAE_Wind = checkpoint_loader.load(scope=globals())
            assert net.pixel_dim == len(train_metadata.input_vars)
        else:
            print(f"Training VAE_Wind from scratch with: {vae_config.to_dict()}")
            net: VAE_Wind = VAE_Wind(
                n_days=1,
                n_features=len(train_metadata.input_vars),
                latent_dim=vae_config.latent_dim,
                hidden_dim=vae_config.hidden_dim,
                n_scaling_blocks=vae_config.n_scaling_blocks,
                n_convstack_layers=vae_config.n_convstack_layers,
                n_convhead_layers=vae_config.n_convhead_layers,
            )

        assert isinstance(net, VAE_Wind)
        trainer = VAETrainer(
            net=net,
            lambda_=vae_config.lambda_,
            lr=vae_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=vae_config.train_batch_size,
            val_batch_size=vae_config.val_batch_size,
            local_rank=local_rank,
        )
        trainer.train(
            n_epochs=vae_config.n_epochs,
            patience=vae_config.patience,
            tolerance=vae_config.tolerance,
            checkpoint_directory=vae_config.saved_checkpoint_directory.as_posix(),
            save_frequency=vae_config.save_frequency,
        )

    elif model.lower() == "vae-mass":
        vae_config: VAEConfig = VAEConfig(context_group="mass")
        assert vae_config.context_group == "mass"
        train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
        val_metadata: MetaData = MetaData(dataset_name=dataset, tp="val")
        if dataset == "cesm2":
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: CESM2 = CESM2(metadata=train_metadata)
            val_dataset: CESM2 = CESM2(metadata=val_metadata)
        else:
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: ERA5 = ERA5(metadata=train_metadata)
            val_dataset: ERA5 = ERA5(metadata=val_metadata)

        if (checkpoint_path := vae_config.from_checkpoint) is not None:
            print(f"Training VAE_Mass from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: VAE_Mass = checkpoint_loader.load(scope=globals())
            assert net.pixel_dim == len(train_metadata.input_vars)
        else:
            print(f"Training VAE_Mass from scratch with: {vae_config.to_dict()}")
            net: VAE_Mass = VAE_Mass(
                n_days=1,
                n_features=len(train_metadata.input_vars),
                latent_dim=vae_config.latent_dim,
                hidden_dim=vae_config.hidden_dim,
                n_scaling_blocks=vae_config.n_scaling_blocks,
                n_convstack_layers=vae_config.n_convstack_layers,
                n_convhead_layers=vae_config.n_convhead_layers,
            )

        assert isinstance(net, VAE_Mass)
        trainer = VAETrainer(
            net=net,
            lambda_=vae_config.lambda_,
            lr=vae_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=vae_config.train_batch_size,
            val_batch_size=vae_config.val_batch_size,
            local_rank=local_rank,
        )
        trainer.train(
            n_epochs=vae_config.n_epochs,
            patience=vae_config.patience,
            tolerance=vae_config.tolerance,
            checkpoint_directory=vae_config.saved_checkpoint_directory.as_posix(),
            save_frequency=vae_config.save_frequency,
        )

    elif model.lower() == "vae-thermal":
        vae_config: VAEConfig = VAEConfig(context_group="thermal")
        assert vae_config.context_group == "thermal"
        train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
        val_metadata: MetaData = MetaData(dataset_name=dataset, tp="val")
        if dataset == "cesm2":
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: CESM2 = CESM2(metadata=train_metadata)
            val_dataset: CESM2 = CESM2(metadata=val_metadata)
        else:
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: ERA5 = ERA5(metadata=train_metadata)
            val_dataset: ERA5 = ERA5(metadata=val_metadata)

        if (checkpoint_path := vae_config.from_checkpoint) is not None:
            print(f"Training VAE_Thermal from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: VAE_Thermal = checkpoint_loader.load(scope=globals())
            assert net.pixel_dim == len(train_metadata.input_vars)
        else:
            print(f"Training VAE_Thermal from scratch with: {vae_config.to_dict()}")
            net: VAE_Thermal = VAE_Thermal(
                n_days=1,
                n_features=len(train_metadata.input_vars),
                latent_dim=vae_config.latent_dim,
                hidden_dim=vae_config.hidden_dim,
                n_scaling_blocks=vae_config.n_scaling_blocks,
                n_convstack_layers=vae_config.n_convstack_layers,
                n_convhead_layers=vae_config.n_convhead_layers,
            )

        assert isinstance(net, VAE_Thermal)
        trainer = VAETrainer(
            net=net,
            lambda_=vae_config.lambda_,
            lr=vae_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=vae_config.train_batch_size,
            val_batch_size=vae_config.val_batch_size,
            local_rank=local_rank,
        )
        trainer.train(
            n_epochs=vae_config.n_epochs,
            patience=vae_config.patience,
            tolerance=vae_config.tolerance,
            checkpoint_directory=vae_config.saved_checkpoint_directory.as_posix(),
            save_frequency=vae_config.save_frequency,
        )

    elif model.lower() == "vae-hydro":
        vae_config: VAEConfig = VAEConfig(context_group="hydro")
        assert vae_config.context_group == "hydro"
        train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
        val_metadata: MetaData = MetaData(dataset_name=dataset, tp="val")
        if dataset == "cesm2":
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: CESM2 = CESM2(metadata=train_metadata)
            val_dataset: CESM2 = CESM2(metadata=val_metadata)
        else:
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: ERA5 = ERA5(metadata=train_metadata)
            val_dataset: ERA5 = ERA5(metadata=val_metadata)

        if (checkpoint_path := vae_config.from_checkpoint) is not None:
            print(f"Training VAE_Hydro from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: VAE_Hydro = checkpoint_loader.load(scope=globals())
            assert net.pixel_dim == len(train_metadata.input_vars)
        else:
            print(f"Training VAE_Hydro from scratch with: {vae_config.to_dict()}")
            net: VAE_Hydro = VAE_Hydro(
                n_days=1,
                n_features=len(train_metadata.input_vars),
                latent_dim=vae_config.latent_dim,
                hidden_dim=vae_config.hidden_dim,
                n_scaling_blocks=vae_config.n_scaling_blocks,
                n_convstack_layers=vae_config.n_convstack_layers,
                n_convhead_layers=vae_config.n_convhead_layers,
            )

        assert isinstance(net, VAE_Hydro)
        trainer = VAETrainer(
            net=net,
            lambda_=vae_config.lambda_,
            lr=vae_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=vae_config.train_batch_size,
            val_batch_size=vae_config.val_batch_size,
            local_rank=local_rank,
        )
        trainer.train(
            n_epochs=vae_config.n_epochs,
            patience=vae_config.patience,
            tolerance=vae_config.tolerance,
            checkpoint_directory=vae_config.saved_checkpoint_directory.as_posix(),
            save_frequency=vae_config.save_frequency,
        )

    elif model.lower() == "vae-precip":
        vae_config: VAEConfig = VAEConfig(context_group="precip")
        assert vae_config.context_group == "precip"
        train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
        val_metadata: MetaData = MetaData(dataset_name=dataset, tp="val")
        if dataset == "cesm2":
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: CESM2 = CESM2(metadata=train_metadata)
            val_dataset: CESM2 = CESM2(metadata=val_metadata)
        else:
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: ERA5 = ERA5(metadata=train_metadata)
            val_dataset: ERA5 = ERA5(metadata=val_metadata)

        if (checkpoint_path := vae_config.from_checkpoint) is not None:
            print(f"Training VAE_Precip from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: VAE_Precip = checkpoint_loader.load(
                scope=globals(),
                is_finetuning=is_finetuning,
                lora_rank=lora_rank,
            )
            assert net.pixel_dim == len(train_metadata.input_vars)
        else:
            print(f"Training VAE_Precip from scratch with: {vae_config.to_dict()}")
            assert not is_finetuning
            net: VAE_Precip = VAE_Precip(
                n_days=1,
                n_features=len(train_metadata.input_vars),
                latent_dim=vae_config.latent_dim,
                hidden_dim=vae_config.hidden_dim,
                n_scaling_blocks=vae_config.n_scaling_blocks,
                n_convstack_layers=vae_config.n_convstack_layers,
                n_convhead_layers=vae_config.n_convhead_layers,
                is_finetuning=False,
                lora_rank=0,
            )

        assert isinstance(net, VAE_Precip)
        if is_finetuning:
            net.freeze_backbone()
            assert net.is_backbone_frozen()

        trainer = VAETrainer(
            net=net,
            lambda_=vae_config.lambda_,
            lr=vae_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=vae_config.train_batch_size,
            val_batch_size=vae_config.val_batch_size,
            local_rank=local_rank,
        )
        trainer.train(
            n_epochs=vae_config.n_epochs,
            patience=vae_config.patience,
            tolerance=vae_config.tolerance,
            checkpoint_directory=vae_config.saved_checkpoint_directory.as_posix(),
            save_frequency=vae_config.save_frequency,
        )

    elif model.lower() == "diffusion":
        print("Training Diffusion")
        diffusion_config: DiffusionConfig = DiffusionConfig()

        # Wind encoder
        print(f"Loading wind_encoder from {diffusion_config.vae_wind_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=diffusion_config.vae_wind_checkpoint.as_posix())
        wind_encoder: VAEEncoder = checkpoint_loader.load(scope=globals()).encoder
        # Mass encoder
        print(f"Loading mass_encoder from {diffusion_config.vae_mass_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=diffusion_config.vae_mass_checkpoint.as_posix())
        mass_encoder: VAEEncoder = checkpoint_loader.load(scope=globals()).encoder
        # Thermodynamic encoder
        print(f"Loading thermal_encoder from {diffusion_config.vae_thermal_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=diffusion_config.vae_thermal_checkpoint.as_posix())
        thermal_encoder: VAEEncoder = checkpoint_loader.load(scope=globals()).encoder
        # Hydro encoder
        print(f"Loading hydro_encoder from {diffusion_config.vae_hydro_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=diffusion_config.vae_hydro_checkpoint.as_posix())
        hydro_encoder: VAEEncoder = checkpoint_loader.load(scope=globals()).encoder
        # Precipitation encoder
        print(f"Loading precip_encoder from {diffusion_config.vae_precip_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=diffusion_config.vae_precip_checkpoint.as_posix())
        precip_encoder: VAEEncoder = checkpoint_loader.load(scope=globals()).encoder

        # Denoiser
        if (checkpoint_path := diffusion_config.from_checkpoint) is not None:
            print(f"Training UNetDenoiser from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: UNetDenoiser = checkpoint_loader.load(
                scope=globals(),
                is_finetuning=is_finetuning,
                lora_rank=lora_rank,
            )
            assert isinstance(net, UNetDenoiser)
        else:
            print(f"Training UNetDenoiser from scratch with: {diffusion_config.to_dict()}")
            assert not is_finetuning
            net: UNetDenoiser = UNetDenoiser(
                target_dim=diffusion_config.target_dim,
                condition_dim=diffusion_config.condition_dim,
                in_H=precip_encoder.expected_H, in_W=precip_encoder.expected_W,
                down_out_dims=diffusion_config.down_out_dims,
                mid_out_dims=diffusion_config.mid_out_dims,
                up_out_dims=diffusion_config.up_out_dims,
                down_transformer_model_dims=diffusion_config.down_transformer_model_dims,
                mid_transformer_model_dims=diffusion_config.mid_transformer_model_dims,
                up_transformer_model_dims=diffusion_config.up_transformer_model_dims,
                n_conv_layers_per_scaling_block=diffusion_config.n_conv_layers_per_scaling_block,
                n_transformer_encoder_layers_per_scaling_block=diffusion_config.n_transformer_encoder_layers_per_scaling_block,
                n_transformer_decoder_layers_per_scaling_block=diffusion_config.n_transformer_decoder_layers_per_scaling_block,
                n_conv_layers_per_mid_block=diffusion_config.n_conv_layers_per_mid_block,
                n_transformer_encoder_layers_per_mid_block=diffusion_config.n_transformer_encoder_layers_per_mid_block,
                n_transformer_decoder_layers_per_mid_block=diffusion_config.n_transformer_decoder_layers_per_mid_block,
                transformer_feedforward_dim=diffusion_config.transformer_feedforward_dim,
                n_attention_heads=diffusion_config.n_attention_heads,
                transformer_maxlength=diffusion_config.transformer_maxlength,
                is_finetuning=False,
                lora_rank=0,
            )

        # Noise scheduler
        if diffusion_config.noise_scheduler.lower() == "linear":
            print("Training Diffusion with 'linear' scheduler")
            noise_scheduler = LinearNoiseScheduler(
                n_steps=diffusion_config.n_steps,
                beta_min=diffusion_config.beta_min,
                beta_max=diffusion_config.beta_max,
            )

        elif diffusion_config.noise_scheduler.lower() == "cosine":
            print("Training Diffusion with 'cosine' scheduler")
            noise_scheduler = CosineNoiseScheduler(n_steps=diffusion_config.n_steps)
        else:
            raise ValueError(f"Invalid diffusion_config.noise_scheduler={diffusion_config.noise_scheduler}")

        trainer = DiffusionTrainer(
            denoiser=net,
            wind_encoder=wind_encoder, mass_encoder=mass_encoder, thermal_encoder=thermal_encoder,
            hydro_encoder=hydro_encoder, precip_encoder=precip_encoder,
            noise_scheduler=noise_scheduler,
            lr=diffusion_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=diffusion_config.train_batch_size,
            val_batch_size=diffusion_config.val_batch_size,
            is_finetuning=is_finetuning,
            local_rank=local_rank,
        )
        trainer.train(
            n_epochs=diffusion_config.n_epochs,
            patience=diffusion_config.patience,
            tolerance=diffusion_config.tolerance,
            checkpoint_directory=diffusion_config.saved_checkpoint_directory.as_posix(),
            save_frequency=diffusion_config.save_frequency,
        )

    else:
        raise NotImplementedError(f"Unknown model: {model}")


if __name__ == "__main__":
    import argparse
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    def setup_ddp() -> int:
        assert "RANK" in os.environ
        assert "WORLD_SIZE" in os.environ
        rank: int = int(os.environ["RANK"])
        local_rank: int = int(os.environ["LOCAL_RANK"])
        world_size: int = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(device=local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        return local_rank

    def cleanup_ddp() -> None:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=[
            "cnn", "unet", "vit",
            "vae-wind", "vae-mass", "vae-thermal", "vae-hydro", "vae-precip", "diffusion",
        ],
        required=True,
    )
    parser.add_argument(
        "--dataset", type=str, choices=["cesm2", "era5"],
        required=True,
    )
    parser.add_argument(
        "--finetune", action="store_true", dest="is_finetuning", required=False,
    )
    parser.add_argument("--lora-rank", dest="lora_rank", type=int, default=0, required=False)
    args: argparse.Namespace = parser.parse_args()

    local_rank: int = setup_ddp()
    try:
        main(
            model=args.model, dataset=args.dataset,
            is_finetuning=args.is_finetuning, lora_rank=args.lora_rank,
            local_rank=local_rank,
        )
    finally:
        cleanup_ddp()
