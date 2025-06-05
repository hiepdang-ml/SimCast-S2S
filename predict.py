from typing import *
import argparse

from models import CNN
from cesm2 import CESM2
from common.utils import CheckpointLoader
from workers import BaselinePredictor, VAEPredictor
from common.configs import MetaData, CNNConfig, UnetConfig, ViTConfig, VAEContextConfig, VAETargetConfig
from models import CNN, UNet, ViT, VAE


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
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.best_checkpoint)
        net: CNN = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, CNN)
        assert net.n_input_days == test_metadata.n_input_days

    elif model.lower() == "unet":
        model_config: UnetConfig = UnetConfig()
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.best_checkpoint)
        net: UNet = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, UNet)
        assert net.n_input_days == test_metadata.n_input_days

    elif model.lower() == "vit":
        model_config: UnetConfig = ViTConfig()
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.best_checkpoint)
        net: ViT = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, ViT)
        assert net.n_input_days == test_metadata.n_input_days

    elif model.lower() == "vae-context":
        model_config: UnetConfig = VAEContextConfig()
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.best_checkpoint)
        net: VAE = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, VAE)
        assert net.pixel_dim == test_metadata.n_input_days * len(test_metadata.input_vars)

    elif model.lower() == "vae-target":
        model_config: UnetConfig = VAETargetConfig()
        checkpoint_loader = CheckpointLoader(checkpoint_path=model_config.best_checkpoint)
        net: VAE = checkpoint_loader.load(scope=globals()).to(device=model_config.device)
        assert isinstance(net, VAE)
        assert net.pixel_dim == test_metadata.n_input_days * len(test_metadata.input_vars)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=["cnn", "unet", "vit", "vae-context", "vae-target"], 
        required=True,
    )
    args: argparse.Namespace = parser.parse_args()
    main(model=args.model)

