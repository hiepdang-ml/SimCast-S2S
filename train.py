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
    VAE, VAE_Wind, VAE_Geopotential, VAE_ThermalDynamic, VAE_Precipitation, 
    VAEEncoder, VAEDecoder, UNetDenoiser, 
    LinearNoiseScheduler, CosineNoiseScheduler, ForwardProcess, ReverseProcess,
)


def main(
    model: Literal[
        "cnn", "unet", "vit", 
        "vae-wind", "vae-geopotential", "vae-thermaldynamic", "vae-precipitation"
    ],
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
        cnn_config: CNNConfig = CNNConfig()
        if (checkpoint_path := cnn_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: CNN = checkpoint_loader.load(scope=globals()).to(device=cnn_config.device)
            assert isinstance(net, CNN)
            assert net.n_input_days == train_metadata.n_input_days
        else:
            print("Training from scratch")
            net: CNN = CNN(
                n_input_days=train_metadata.n_input_days,
                in_features=cnn_config.in_features,
                out_features=cnn_config.out_features,
                embedding_dim=cnn_config.embedding_dim,
                n_hidden_layers=cnn_config.n_hidden_layers,
            ).to(device=cnn_config.device)

        trainer = BaselineTrainer(
            net=net,
            lr=cnn_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=cnn_config.train_batch_size,
            val_batch_size=cnn_config.val_batch_size,
        )
        trainer.train(
            n_epochs=cnn_config.n_epochs,
            patience=cnn_config.patience,
            tolerance=cnn_config.tolerance,
            checkpoint_directory=cnn_config.saved_checkpoint_directory,
            save_frequency=cnn_config.save_frequency,
        )
    
    elif model.lower() == "unet":
        print("Training UNet")
        unet_config: UnetConfig = UnetConfig()
        if (checkpoint_path := unet_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: UNet = checkpoint_loader.load(scope=globals()).to(device=unet_config.device)
            assert isinstance(net, UNet)
            assert net.n_input_days == train_metadata.n_input_days
        else:
            print("Training from scratch")
            net: UNet = UNet(
                n_input_days=train_metadata.n_input_days,
                in_features=unet_config.in_features,
                out_features=unet_config.out_features,
                embedding_dim=unet_config.embedding_dim,
            ).to(device=unet_config.device)

        trainer = BaselineTrainer(
            net=net,
            lr=unet_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=unet_config.train_batch_size,
            val_batch_size=unet_config.val_batch_size,
        )
        trainer.train(
            n_epochs=unet_config.n_epochs,
            patience=unet_config.patience,
            tolerance=unet_config.tolerance,
            checkpoint_directory=unet_config.saved_checkpoint_directory,
            save_frequency=unet_config.save_frequency,
        )

    elif model.lower() == "vit":
        print("Training ViT")
        vit_config: ViTConfig = ViTConfig()
        if (checkpoint_path := vit_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: ViT = checkpoint_loader.load(scope=globals()).to(device=vit_config.device)
            assert isinstance(net, ViT)
            assert net.n_input_days == train_metadata.n_input_days
        else:
            print("Training from scratch")
            net: ViT = ViT(
                n_input_days=train_metadata.n_input_days,
                in_features=vit_config.in_features,
                out_features=vit_config.out_features,
                embedding_dim=vit_config.embedding_dim,
                patch_size=vit_config.patch_size,
                n_heads=vit_config.n_heads,
                n_transformer_layers=vit_config.n_transformer_layers,
                dropout=vit_config.dropout,
            ).to(device=vit_config.device)

        trainer = BaselineTrainer(
            net=net,
            lr=vit_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=vit_config.train_batch_size,
            val_batch_size=vit_config.val_batch_size,
        )
        trainer.train(
            n_epochs=vit_config.n_epochs,
            patience=vit_config.patience,
            tolerance=vit_config.tolerance,
            checkpoint_directory=vit_config.saved_checkpoint_directory,
            save_frequency=vit_config.save_frequency,
        )

    elif model.lower() == "vae-wind":
        vae_config: VAEConfig = VAEConfig(context_group="wind")
        train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
        val_metadata: MetaData = MetaData(dataset_name=dataset, tp="val")
        if dataset == "cesm2":
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: CESM2 = CESM2(metadata=train_metadata)
            val_dataset: CESM2 = CESM2(metadata=val_metadata)
        # TODO
        else:
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            ...

        if (checkpoint_path := vae_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: VAE_Wind = checkpoint_loader.load(scope=globals()).to(device=vae_config.device)
            assert isinstance(net, VAE_Wind)
            assert net.pixel_dim == len(train_metadata.input_vars)
        else:
            print("Training from scratch")
            net: VAE_Wind = VAE_Wind(
                n_days=1,
                n_features=len(train_metadata.input_vars),
                latent_dim=vae_config.latent_dim,
                hidden_dim=vae_config.hidden_dim,
                n_scaling_blocks=vae_config.n_scaling_blocks,
                n_convstack_layers=vae_config.n_convstack_layers,
                n_convhead_layers=vae_config.n_convhead_layers,
            ).to(device=vae_config.device)

        trainer = VAETrainer(
            net=net,
            lambda_=vae_config.lambda_,
            lr=vae_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=vae_config.train_batch_size,
            val_batch_size=vae_config.val_batch_size,
        )
        trainer.train(
            n_epochs=vae_config.n_epochs,
            patience=vae_config.patience,
            tolerance=vae_config.tolerance,
            checkpoint_directory=vae_config.saved_checkpoint_directory,
            save_frequency=vae_config.save_frequency,
        )

    elif model.lower() == "vae-geopotential":
        vae_config: VAEConfig = VAEConfig(context_group="geopotential")
        train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
        val_metadata: MetaData = MetaData(dataset_name=dataset, tp="val")
        if dataset == "cesm2":
            print(f"Training {model} on {dataset}")
            input_subset: list[str] = ["Z200", "Z850", "Z500"]
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: CESM2 = CESM2(metadata=train_metadata)
            val_dataset: CESM2 = CESM2(metadata=val_metadata)
        # TODO
        else:
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            ...

        if (checkpoint_path := vae_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: VAE_Geopotential = checkpoint_loader.load(scope=globals()).to(device=vae_config.device)
            assert isinstance(net, VAE_Geopotential)
            assert net.pixel_dim == len(train_metadata.input_vars)
        else:
            print("Training from scratch")
            net: VAE_Geopotential = VAE_Geopotential(
                n_days=1,
                n_features=len(train_metadata.input_vars),
                latent_dim=vae_config.latent_dim,
                hidden_dim=vae_config.hidden_dim,
                n_scaling_blocks=vae_config.n_scaling_blocks,
                n_convstack_layers=vae_config.n_convstack_layers,
                n_convhead_layers=vae_config.n_convhead_layers,
            ).to(device=vae_config.device)

        trainer = VAETrainer(
            net=net,
            lambda_=vae_config.lambda_,
            lr=vae_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=vae_config.train_batch_size,
            val_batch_size=vae_config.val_batch_size,
        )
        trainer.train(
            n_epochs=vae_config.n_epochs,
            patience=vae_config.patience,
            tolerance=vae_config.tolerance,
            checkpoint_directory=vae_config.saved_checkpoint_directory,
            save_frequency=vae_config.save_frequency,
        )

    elif model.lower() == "vae-thermaldynamic":
        vae_config: VAEConfig = VAEConfig(context_group="thermaldynamic")
        train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
        val_metadata: MetaData = MetaData(dataset_name=dataset, tp="val")
        if dataset == "cesm2":
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: CESM2 = CESM2(metadata=train_metadata)
            val_dataset: CESM2 = CESM2(metadata=val_metadata)
        # TODO
        else:
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            ...

        if (checkpoint_path := vae_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: VAE_ThermalDynamic = checkpoint_loader.load(scope=globals()).to(device=vae_config.device)
            assert isinstance(net, VAE_ThermalDynamic)
            assert net.pixel_dim == len(train_metadata.input_vars)
        else:
            print("Training from scratch")
            net: VAE_ThermalDynamic = VAE_ThermalDynamic(
                n_days=1,
                n_features=len(train_metadata.input_vars),
                latent_dim=vae_config.latent_dim,
                hidden_dim=vae_config.hidden_dim,
                n_scaling_blocks=vae_config.n_scaling_blocks,
                n_convstack_layers=vae_config.n_convstack_layers,
                n_convhead_layers=vae_config.n_convhead_layers,
            ).to(device=vae_config.device)

        trainer = VAETrainer(
            net=net,
            lambda_=vae_config.lambda_,
            lr=vae_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=vae_config.train_batch_size,
            val_batch_size=vae_config.val_batch_size,
        )
        trainer.train(
            n_epochs=vae_config.n_epochs,
            patience=vae_config.patience,
            tolerance=vae_config.tolerance,
            checkpoint_directory=vae_config.saved_checkpoint_directory,
            save_frequency=vae_config.save_frequency,
        )

    elif model.lower() == "vae-precipitation":
        vae_config: VAEConfig = VAEConfig(context_group="precipitation")
        train_metadata: MetaData = MetaData(dataset_name=dataset, tp="train")
        val_metadata: MetaData = MetaData(dataset_name=dataset, tp="val")
        if dataset == "cesm2":
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            train_dataset: CESM2 = CESM2(metadata=train_metadata)
            val_dataset: CESM2 = CESM2(metadata=val_metadata)
        # TODO
        else:
            print(f"Training {model} on {dataset}")
            train_metadata = train_metadata.with_var_subset(context_group=vae_config.context_group)
            val_metadata = val_metadata.with_var_subset(context_group=vae_config.context_group)
            ...

        if (checkpoint_path := vae_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: VAE_Precipitation = checkpoint_loader.load(scope=globals()).to(device=vae_config.device)
            assert isinstance(net, VAE_Precipitation)
            assert net.pixel_dim == len(train_metadata.input_vars)
        else:
            print("Training from scratch")
            net: VAE_Precipitation = VAE_Precipitation(
                n_days=1,
                n_features=len(train_metadata.input_vars),
                latent_dim=vae_config.latent_dim,
                hidden_dim=vae_config.hidden_dim,
                n_scaling_blocks=vae_config.n_scaling_blocks,
                n_convstack_layers=vae_config.n_convstack_layers,
                n_convhead_layers=vae_config.n_convhead_layers,
            ).to(device=vae_config.device)

        trainer = VAETrainer(
            net=net,
            lambda_=vae_config.lambda_,
            lr=vae_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=vae_config.train_batch_size,
            val_batch_size=vae_config.val_batch_size,
        )
        trainer.train(
            n_epochs=vae_config.n_epochs,
            patience=vae_config.patience,
            tolerance=vae_config.tolerance,
            checkpoint_directory=vae_config.saved_checkpoint_directory,
            save_frequency=vae_config.save_frequency,
        )

    elif model.lower() == "diffusion":
        print("Training Diffusion")
        # Denoiser
        diffusion_config: DiffusionConfig = DiffusionConfig()
        if (checkpoint_path := diffusion_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=str(checkpoint_path))
            net: UNetDenoiser = checkpoint_loader.load(scope=globals())
            assert isinstance(net, UNetDenoiser)
        else:
            print("Training UNetDenoiser from scratch")
            net: UNetDenoiser = UNetDenoiser(
                target_dim=diffusion_config.target_dim,
                condition_dim=diffusion_config.condition_dim,
                step_dim=diffusion_config.step_dim,
                day_dim=diffusion_config.day_dim,
                n_condition_days=diffusion_config.n_condition_days,
                down_out_dims=diffusion_config.down_out_dims,
                down_hidden_dims=diffusion_config.down_hidden_dims,
                mid_out_dims=diffusion_config.mid_out_dims,
                mid_hidden_dims=diffusion_config.mid_hidden_dims,
                up_out_dims=diffusion_config.up_out_dims,
                up_hidden_dims=diffusion_config.up_hidden_dims,
                n_layers_per_scaling_block=diffusion_config.n_layers_per_scaling_block,
                n_layers_per_mid_block=diffusion_config.n_layers_per_mid_block,
                n_attention_heads=diffusion_config.n_attention_heads,
                condition_dropout=diffusion_config.condition_dropout,
            ).to(device=diffusion_config.device)
        
        # Wind encoder
        print(f"Loading wind_encoder from {diffusion_config.wind_vae_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=diffusion_config.wind_vae_checkpoint)
        wind_encoder: VAE_Wind = checkpoint_loader.load(scope=globals()).encoder
        # Geopotential encoder
        print(f"Loading geopotential_encoder from {diffusion_config.geopotential_vae_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=diffusion_config.geopotential_vae_checkpoint)
        geopotential_encoder: VAE_Geopotential = checkpoint_loader.load(scope=globals()).encoder
        # Thermaldynamic encoder
        print(f"Loading thermaldynamic_encoder from {diffusion_config.thermaldynamic_vae_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=diffusion_config.thermaldynamic_vae_checkpoint)
        thermaldynamic_encoder: VAE_ThermalDynamic = checkpoint_loader.load(scope=globals()).encoder
        # Precipitation encoder
        print(f"Loading precipitation_encoder from {diffusion_config.precipitation_vae_checkpoint}")
        checkpoint_loader = CheckpointLoader(checkpoint_path=diffusion_config.precipitation_vae_checkpoint)
        precipitation_encoder: VAE_Precipitation = checkpoint_loader.load(scope=globals()).encoder
        # Noise scheduler
        if diffusion_config.noise_scheduler.lower() == "linear":
            noise_scheduler = LinearNoiseScheduler(
                n_steps=diffusion_config.n_steps, 
                beta_min=diffusion_config.beta_min, 
                beta_max=diffusion_config.beta_max, 
                device=diffusion_config.device,
            )
        elif diffusion_config.noise_scheduler.lower() == "cosine":
            noise_scheduler = CosineNoiseScheduler(
                n_steps=diffusion_config.n_steps,
                device=diffusion_config.device,
            )
        else:
            raise ValueError(f"Invalid diffusion_config.noise_scheduler={diffusion_config.noise_scheduler}")

        trainer = DiffusionTrainer(
            denoiser=net,
            wind_encoder=wind_encoder,
            geopotential_encoder=geopotential_encoder,
            thermaldynamic_encoder=thermaldynamic_encoder,
            precipitation_encoder=precipitation_encoder,
            noise_scheduler=noise_scheduler,
            lr=diffusion_config.learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_batch_size=diffusion_config.train_batch_size,
            val_batch_size=diffusion_config.val_batch_size,
        )
        trainer.train(
            n_epochs=diffusion_config.n_epochs,
            patience=diffusion_config.patience,
            tolerance=diffusion_config.tolerance,
            checkpoint_directory=diffusion_config.saved_checkpoint_directory,
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
        local_rank: int = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device=local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank

    def cleanup_ddp() -> None:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=[
            "cnn", "unet", "vit", 
            "vae-wind", "vae-geopotential", "vae-thermaldynamic", "vae-precipitation", "diffusion",
        ],
        required=True,
    )
    parser.add_argument(
        "--dataset", type=str, choices=["cesm2", "era5"],
        required=True,
    )
    args: argparse.Namespace = parser.parse_args()

    local_rank: int = setup_ddp()
    try:
        main(model=args.model, dataset=args.dataset)
    finally:
        cleanup_ddp()

