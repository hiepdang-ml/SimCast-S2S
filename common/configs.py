from abc import ABC, abstractmethod
import yaml

from pathlib import Path
from itertools import product
from typing import Literal, Any


class BaseConfig(ABC):

    _config: dict[str, Any]  # Subclasses populate this before calling super().__init__().

    def __init__(self) -> None:
        self.learning_rate: float = float(self._config["learning_rate"])
        self.train_batch_size: int = self._config["train_batch_size"]
        self.val_batch_size: int = self._config["val_batch_size"]
        self.n_epochs: int = self._config["n_epochs"]
        self.patience: int = self._config["patience"]
        self.tolerance: float = float(self._config["tolerance"])
        self.save_frequency: int = self._config["save_frequency"]
        self.from_checkpoint: Path | None = Path(value) if (value := self._config["from_checkpoint"]) else None
        self.saved_checkpoint_directory: Path = Path(self._config["saved_checkpoint_directory"])
        self.target_path: Path = Path(self._config["target_path"])

    @abstractmethod
    def _load(self) -> None:
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass

class MetaData(BaseConfig):

    VAR_LOOKUP_TABLE: dict[str, dict[str, list[str]]] = {
        "cesm2": {
            "wind": ["U200", "V200", "U500", "V500", "OMEGA500", "U850", "V850"],   # flows
            "mass": ["Z200", "Z500", "Z850", "PSL"],    # mass heights + surface pressure
            "thermal": ["TS", "T200", "T500", "T850", "FLUT"],   # temperature + radiation
            "hydro": ["TMQ", "Q200", "Q500", "Q850"],   # moisture
            "precip": ["PRECT"], # precip
        },
        "era5": {
            "wind": ["u200", "v200", "u500", "v500", "w500", "u850", "v850"],
            "mass": ["z200", "z500", "z850", "msl"],
            "thermal": ["skt", "t200", "t500", "t850", "avgtnlwrf"],
            "hydro": ["tcwv", "q200", "q500", "q850"],
            "precip": ["tp"],
        }
    }

    def __init__(self, dataset_name: Literal["cesm2", "era5"], tp: Literal["train", "val", "test"]) -> None:
        self.dataset_name: Literal["cesm2", "era5"] = dataset_name
        self.tp: Literal["train", "val", "test"] = tp
        self.var_table: dict[str, list[str]] = MetaData.VAR_LOOKUP_TABLE[dataset_name]
        with open("./config.yaml", mode="r") as file:
            self._config: dict[str, Any] = yaml.safe_load(file)[dataset_name]

        self._load()
        # Note: must preserve order
        self.var_names: list[str] = []
        for var_name in self.input_vars + self.output_vars:
            if var_name not in self.var_names:
                self.var_names.append(var_name)

        self.years: list[int] = list(range(self.start_year, self.end_year + 1))
        self.combinations: list[tuple[str, int]] = list(product(self.sim_ids, self.years))
        self.n_years: int = len(self.years)

    def _load(self) -> None:

        self.raw_root: Path = Path(self._config["raw_root"])
        if self.dataset_name == "era5":
            self.array_root: Path = Path(self._config["array_root"])

        self.landmask_root: Path = Path(self._config["landmask_root"])
        self.landmask_path: Path = next(self.landmask_root.glob("*.nc"))

        self.input_vars: list[str] = self._config["input_vars"]
        self.output_vars: list[str] = self._config["output_vars"]
        self.resolution: tuple[int, int] = tuple(self._config["resolution"])
        self.sim_ids: list[str] = self._config["sim_ids"]

        self.start_year: int
        self.end_year: int
        self.write_directory: Path
        if self.tp == "train":
            self.start_year = self._config["train_start_year"]
            self.end_year = self._config["train_end_year"]
            self.write_directory = Path(self._config["train_write_directory"])
        elif self.tp == "val":
            self.start_year = self._config["val_start_year"]
            self.end_year = self._config["val_end_year"]
            self.write_directory = Path(self._config["val_write_directory"])
        elif self.tp == "test":
            self.start_year = self._config["test_start_year"]
            self.end_year = self._config["test_end_year"]
            self.write_directory = Path(self._config["test_write_directory"])
        else:
            raise ValueError(f"Invalid tp for MetaData, expected one of ['train', 'val', 'test'], get: '{self.tp}'")

        self.n_input_days: int = self._config["n_input_days"]
        self.n_lead_days: int = self._config["n_lead_days"]
        self.n_output_days: int = self._config["n_output_days"]
        self.n_step_days: int = self._config["n_step_days"]

        self.detrender_state_directory: Path = Path(self._config["detrender_state_directory"])
        self.climatology_state_directory: Path = Path(self._config["climatology_state_directory"])

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "tp": self.tp,
            "raw_root": self.raw_root.as_posix(),
            "landmask_root": self.landmask_root.as_posix(),
            "landmask_path": self.landmask_path.as_posix(),
            "input_vars": self.input_vars,
            "output_vars": self.output_vars,
            "resolution": self.resolution,
            "sim_ids": self.sim_ids,
            "years": self.years,
            "n_input_days": self.n_input_days,
            "n_lead_days": self.n_lead_days,
            "n_output_days": self.n_output_days,
            "n_step_days": self.n_step_days,
        }
        if self.dataset_name == "era5":
            d.update({"array_root": self.array_root.as_posix()})
        return d

    def with_var_subset(
        self,
        context_group: Literal["wind", "mass", "thermal", "hydro", "precip"],
    ) -> "MetaData":
        input_subset: list[str] = self.var_table[context_group]
        assert set(input_subset).issubset(set(self.input_vars))
        # override, preserve order
        self.input_vars = [v for v in self.input_vars if v in input_subset]
        self.var_names = [v for v in self.var_names if v in (input_subset + self.output_vars)]
        return self


class CNNConfig(BaseConfig):

    def __init__(self) -> None:
        with open("./config.yaml", mode="r") as file:
            self._config: dict[str, Any] = yaml.safe_load(file)["cnn"]

        super().__init__()
        self._load()

    def _load(self) -> None:
        self.in_features: int = self._config["in_features"]
        self.out_features: int = self._config["out_features"]
        self.embedding_dim: int = self._config["embedding_dim"]
        self.n_hidden_layers: int = self._config["n_hidden_layers"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "embedding_dim": self.embedding_dim,
            "n_hidden_layers": self.n_hidden_layers,
            "learning_rate": self.learning_rate,
            "train_batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "tolerance": self.tolerance,
            "save_frequency": self.save_frequency,
            "from_checkpoint": self.from_checkpoint.as_posix() if self.from_checkpoint else None,
            "saved_checkpoint_directory": self.saved_checkpoint_directory.as_posix(),
            "target_path": self.target_path.as_posix(),
        }

class UnetConfig(BaseConfig):

    def __init__(self) -> None:
        with open("./config.yaml", mode="r") as file:
            self._config: dict[str, Any] = yaml.safe_load(file)["unet"]

        super().__init__()
        self._load()

    def _load(self) -> None:
        self.in_features: int = self._config["in_features"]
        self.out_features: int = self._config["out_features"]
        self.embedding_dim: int = self._config["embedding_dim"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "embedding_dim": self.embedding_dim,
            "learning_rate": self.learning_rate,
            "train_batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "tolerance": self.tolerance,
            "save_frequency": self.save_frequency,
            "from_checkpoint": self.from_checkpoint.as_posix() if self.from_checkpoint else None,
            "saved_checkpoint_directory": self.saved_checkpoint_directory.as_posix(),
            "target_path": self.target_path.as_posix(),
        }

class ECMWFS2SConfig:

    def __init__(self) -> None:
        with open("./config.yaml", mode="r") as file:
            self._config: dict[str, Any] = yaml.safe_load(file)["ecmwf-s2s"]

        self.groundtruth_path: Path = Path(self._config["groundtruth_path"])
        self.prediction_path: Path = Path(self._config["prediction_path"])
        self.target_path: Path = Path(self._config["target_path"])

    def to_dict(self) -> dict[str, Path]:
        return {
            "groundtruth_path": self.groundtruth_path,
            "prediction_path": self.prediction_path,
            "target_path": self.target_path,
        }


class VAEConfig(BaseConfig):

    def __init__(self, context_group: Literal["wind", "mass", "thermal", "hydro", "precip"]):
        self.context_group: Literal["wind", "mass", "thermal", "hydro", "precip"] = context_group

        with open("./config.yaml", mode="r") as file:
            self._config: dict[str, Any] = yaml.safe_load(file)[f"vae-{context_group}"]

        super().__init__()
        self._load()

    def _load(self) -> None:
        self.latent_dim: int = self._config["latent_dim"]
        self.hidden_dim: int = self._config["hidden_dim"]
        self.n_scaling_blocks: int = self._config["n_scaling_blocks"]
        self.n_convstack_layers: int = self._config["n_convstack_layers"]
        self.n_convhead_layers: int = self._config["n_convhead_layers"]

        self.lambda_: float = self._config["lambda"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "n_scaling_blocks": self.n_scaling_blocks,
            "n_convstack_layers": self.n_convstack_layers,
            "n_convhead_layers": self.n_convhead_layers,
            "lambda_": self.lambda_,
            "learning_rate": self.learning_rate,
            "train_batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "tolerance": self.tolerance,
            "save_frequency": self.save_frequency,
            "from_checkpoint": self.from_checkpoint.as_posix() if self.from_checkpoint else None,
            "saved_checkpoint_directory": self.saved_checkpoint_directory.as_posix(),
            "target_path": self.target_path.as_posix(),
        }


class DiffusionConfig(BaseConfig):

    def __init__(self) -> None:
        with open("./config.yaml", mode="r") as file:
            self._config: dict[str, Any] = yaml.safe_load(file)["diffusion"]

        super().__init__()
        self._load()

    def _load(self) -> None:
        self.target_dim: int = self._config["target_dim"]
        self.condition_dim: int = self._config["condition_dim"]
        self.down_out_dims: list[int] = self._config["down_out_dims"]
        self.down_transformer_model_dims: list[int] = self._config["down_transformer_model_dims"]
        self.mid_out_dims: list[int] = self._config["mid_out_dims"]
        self.mid_transformer_model_dims: list[int] = self._config["mid_transformer_model_dims"]
        self.up_out_dims: list[int] = self._config["up_out_dims"]
        self.up_transformer_model_dims: list[int] = self._config["up_transformer_model_dims"]
        self.n_conv_layers_per_scaling_block: int = self._config["n_conv_layers_per_scaling_block"]
        self.n_transformer_encoder_layers_per_scaling_block: list[int] = self._config["n_transformer_encoder_layers_per_scaling_block"]
        self.n_transformer_decoder_layers_per_scaling_block: list[int] = self._config["n_transformer_decoder_layers_per_scaling_block"]
        self.n_conv_layers_per_mid_block: int = self._config["n_conv_layers_per_mid_block"]
        self.n_transformer_encoder_layers_per_mid_block: list[int] = self._config["n_transformer_encoder_layers_per_mid_block"]
        self.n_transformer_decoder_layers_per_mid_block: list[int] = self._config["n_transformer_decoder_layers_per_mid_block"]
        self.transformer_feedforward_dim: int = self._config["transformer_feedforward_dim"]
        self.n_attention_heads: int = self._config["n_attention_heads"]
        self.transformer_maxlength: int = self._config["transformer_maxlength"]
        self.train_steps: int = self._config["train_steps"]
        self.inference_steps: int = self._config["inference_steps"]
        self.cosine_offset: float = float(self._config.get("cosine_offset", 0.008))
        self.eta: float = float(self._config["eta"])
        self.ensemble_size: int = int(self._config["ensemble_size"])
        self.condition_dropout_prob: float = float(self._config["condition_dropout_prob"])
        self.guidance_scale: float = float(self._config["guidance_scale"])
        self.learning_rate: float = float(self._config["learning_rate"])
        self.vae_wind_checkpoint: Path = Path(self._config["vae_wind_checkpoint"])
        self.vae_mass_checkpoint: Path = Path(self._config["vae_mass_checkpoint"])
        self.vae_thermal_checkpoint: Path = Path(self._config["vae_thermal_checkpoint"])
        self.vae_hydro_checkpoint: Path = Path(self._config["vae_hydro_checkpoint"])
        self.vae_precip_checkpoint: Path = Path(self._config["vae_precip_checkpoint"])

    @property
    def noise_schedule_config(self) -> dict[str, Any]:
        return {
            "train_steps": self.train_steps,
            "cosine_offset": self.cosine_offset,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_dim": self.target_dim,
            "condition_dim": self.condition_dim,
            "down_out_dims": self.down_out_dims,
            "down_transformer_model_dims": self.down_transformer_model_dims,
            "mid_out_dims": self.mid_out_dims,
            "mid_transformer_model_dims": self.mid_transformer_model_dims,
            "up_out_dims": self.up_out_dims,
            "up_transformer_model_dims": self.up_transformer_model_dims,
            "n_conv_layers_per_scaling_block": self.n_conv_layers_per_scaling_block,
            "n_transformer_encoder_layers_per_scaling_block": self.n_transformer_encoder_layers_per_scaling_block,
            "n_transformer_decoder_layers_per_scaling_block": self.n_transformer_decoder_layers_per_scaling_block,
            "n_conv_layers_per_mid_block": self.n_conv_layers_per_mid_block,
            "n_transformer_encoder_layers_per_mid_block": self.n_transformer_encoder_layers_per_mid_block,
            "n_transformer_decoder_layers_per_mid_block": self.n_transformer_decoder_layers_per_mid_block,
            "transformer_feedforward_dim": self.transformer_feedforward_dim,
            "n_attention_heads": self.n_attention_heads,
            "transformer_maxlength": self.transformer_maxlength,
            "train_steps": self.train_steps,
            "inference_steps": self.inference_steps,
            "cosine_offset": self.cosine_offset,
            "eta": self.eta,
            "ensemble_size": self.ensemble_size,
            "condition_dropout_prob": self.condition_dropout_prob,
            "guidance_scale": self.guidance_scale,
            "learning_rate": self.learning_rate,
            "train_batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "tolerance": self.tolerance,
            "save_frequency": self.save_frequency,
            "vae_wind_checkpoint": self.vae_wind_checkpoint.as_posix(),
            "vae_mass_checkpoint": self.vae_mass_checkpoint.as_posix(),
            "vae_thermal_checkpoint": self.vae_thermal_checkpoint.as_posix(),
            "vae_hydro_checkpoint": self.vae_hydro_checkpoint.as_posix(),
            "vae_precip_checkpoint": self.vae_precip_checkpoint.as_posix(),
            "from_checkpoint": self.from_checkpoint.as_posix() if self.from_checkpoint else None,
            "saved_checkpoint_directory": self.saved_checkpoint_directory.as_posix(),
            "target_path": self.target_path.as_posix(),
        }
