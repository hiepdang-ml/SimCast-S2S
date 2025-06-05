import argparse
from typing import *

from models import CNN
from cesm2 import CESM2
from common.utils import CheckpointLoader
from workers import BaselineTrainer, VAETrainer
from common.configs import MetaData, CNNConfig, UnetConfig, ViTConfig, VAEContextConfig, VAETargetConfig
from models import CNN, UNet, ViT, VAE


def main(model: Literal["cnn", "unet", "vit", "vae-context", "vae-target"]) -> None:

    # Dataset
    train_metadata: MetaData = MetaData(tp="train")
    val_metadata: MetaData = MetaData(tp="val")
    train_dataset: CESM2 = CESM2(metadata=train_metadata)
    val_dataset: CESM2 = CESM2(metadata=val_metadata)

    # Model
    if model.lower() == "cnn":
        print("Training CNN")
        model_config: CNNConfig = CNNConfig()
        if (checkpoint_path := model_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path)
            net: CNN = checkpoint_loader.load(scope=globals()).cuda()
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
            net: UNet = checkpoint_loader.load(scope=globals()).cuda()
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
            net: ViT = checkpoint_loader.load(scope=globals()).cuda()
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
            ).to(device=model_config.device)

    elif model.lower() == "vae-context":
        print("Training VAE-Context")
        model_config: VAEContextConfig = VAEContextConfig()
        pixel_dim: int = train_metadata.n_input_days * len(train_metadata.input_vars)
        if (checkpoint_path := model_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path)
            net: VAE = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, VAE)
            assert net.pixel_dim == pixel_dim
        else:
            print("Training from scratch")
            net: VAE = VAE(
                pixel_dim=pixel_dim,
                latent_dim=model_config.latent_dim,
                n_layers=model_config.n_layers,
            ).to(device=model_config.device)

    elif model.lower() == "vae-target":
        print("Training VAE-Target")
        model_config: VAETargetConfig = VAETargetConfig()
        pixel_dim: int = 1 * len(train_metadata.output_vars)
        if (checkpoint_path := model_config.from_checkpoint) is not None:
            print(f"Training from {checkpoint_path}")
            checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path)
            net: VAE = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, VAE)
            assert net.pixel_dim == pixel_dim
        else:
            print("Training from scratch")
            net: VAE = VAE(
                pixel_dim=pixel_dim,
                latent_dim=model_config.latent_dim,
                n_layers=model_config.n_layers,
            ).to(device=model_config.device)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=["cnn", "unet", "vit", "vae-context", "vae-target"],
        required=True,
    )
    args: argparse.Namespace = parser.parse_args()
    main(model=args.model)

