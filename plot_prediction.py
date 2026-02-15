from typing import Literal
from common.configs import MetaData, CNNConfig, UnetConfig, ViTConfig, VAEConfig, DiffusionConfig
from workers.predictors import Visualizer


def main(
    model: Literal[
        "cnn", "unet", "vit",
        "vae-wind", "vae-mass", "vae-thermal", "vae-hydro", "vae-precip",
        "diffusion",
    ],
    dataset: Literal["cesm2", "era5"],
) -> None:

    if model.lower() == "cnn":
        model_config = CNNConfig()
    elif model.lower() == "unet":
        model_config = UnetConfig()
    elif model.lower() == "vit":
        model_config = ViTConfig()
    elif model.lower() == "vae-wind":
        model_config = VAEConfig(context_group="wind")
    elif model.lower() == "vae-mass":
        model_config = VAEConfig(context_group="mass")
    elif model.lower() == "vae-thermal":
        model_config = VAEConfig(context_group="thermal")
    elif model.lower() == "vae-hydro":
        model_config = VAEConfig(context_group="hydro")
    elif model.lower() == "vae-precip":
        model_config = VAEConfig(context_group="precip")
    elif model.lower() == "diffusion":
        model_config = DiffusionConfig()
    else:
        raise ValueError(f"Invalid model_name: {model}")

    metadata = MetaData(dataset_name=dataset, tp="test")
    source_dir: str = model_config.target_path.joinpath(f"{dataset}/tensors/").as_posix()
    target_dir: str = model_config.target_path.joinpath(f"{dataset}/plots/").as_posix()
    visualizer = Visualizer(metadata=metadata, source_dir=source_dir, target_dir=target_dir)
    for f in sorted(visualizer.source_dir.glob("*")):
        print(f)
        visualizer.plot_diffusion_prediction(f.name)
