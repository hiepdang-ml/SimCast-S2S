from abc import ABC, abstractmethod
import yaml

import pathlib
from itertools import product
from typing import Literal, Any


class BaseConfig(ABC):

    def __init__(self) -> None:
        self.learning_rate: float = float(self._config["learning_rate"])
        self.train_batch_size: int = self._config["train_batch_size"]
        self.val_batch_size: int = self._config["val_batch_size"]
        self.n_epochs: int = self._config["n_epochs"]
        self.patience: float = self._config["patience"]
        self.tolerance: float = float(self._config["tolerance"])
        self.save_frequency: int = self._config["save_frequency"]
        self.from_checkpoint: pathlib.Path | None = pathlib.Path(value) if (value := self._config["from_checkpoint"]) else None
        self.saved_checkpoint_directory: pathlib.Path = pathlib.Path(self._config["saved_checkpoint_directory"])

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
            "wind": ["u_200", "v_200", "u_500", "v_500", "w_500", "u_850", "v_850"],
            "mass": ["z_200", "z_500", "z_850"],
            "thermal": ["avg_tnlwrf", "t2m", "msl"],
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
        self.input_vars: list[str] = self._config["input_vars"]
        self.output_vars: list[str] = self._config["output_vars"]
        self.resolution: tuple[int, int] = tuple(self._config["resolution"])
        self.sim_ids: list[str] = self._config["sim_ids"]

        if self.tp == "train":
            self.start_year: int = self._config["train_start_year"]
            self.end_year: int = self._config["train_end_year"]
            self.write_directory: pathlib.Path = pathlib.Path(self._config["train_write_directory"])
        elif self.tp == "val":
            self.start_year: int = self._config["val_start_year"]
            self.end_year: int = self._config["val_end_year"]
            self.write_directory: pathlib.Path = pathlib.Path(self._config["val_write_directory"])
        elif self.tp == "test":
            self.start_year: int = self._config["test_start_year"]
            self.end_year: int = self._config["test_end_year"]
            self.write_directory: pathlib.Path = pathlib.Path(self._config["test_write_directory"])
        else:
            raise ValueError(f"Invalid tp for MetaData, expected one of ['train', 'val', 'test'], get: '{self.tp}'")

        self.n_input_days: int = self._config["n_input_days"]
        self.n_lead_days: int = self._config["n_lead_days"]
        self.n_output_days: int = self._config["n_output_days"]
        self.n_step_days: int = self._config["n_step_days"]
        self.climatological_window_size: int = self._config["climatological_window_size"]

        self.detrender_state_directory: pathlib.Path = pathlib.Path(self._config["detrender_state_directory"])
        self.climatology_state_directory: pathlib.Path = pathlib.Path(self._config["climatology_state_directory"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "tp": self.tp,
            "input_vars": self.input_vars,
            "output_vars": self.output_vars,
            "resolution": self.resolution,
            "sim_ids": self.sim_ids,
            "years": self.years,
            "n_input_days": self.n_input_days,
            "n_lead_days": self.n_lead_days,
            "n_output_days": self.n_output_days,
            "n_step_days": self.n_step_days,
            "climatological_window_size": self.climatological_window_size,
        }
    
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
            "from_checkpoint": self.from_checkpoint,
            "saved_checkpoint_directory": self.saved_checkpoint_directory,
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
            "from_checkpoint": self.from_checkpoint,
            "saved_checkpoint_directory": self.saved_checkpoint_directory,
        }

class ViTConfig(BaseConfig):

    def __init__(self) -> None:
        with open("./config.yaml", mode="r") as file:
            self._config: dict[str, Any] = yaml.safe_load(file)["vit"]
        
        super().__init__()
        self._load()

    def _load(self) -> None:
        self.in_features: int = self._config["in_features"]
        self.out_features: int = self._config["out_features"]
        self.embedding_dim: int = self._config["embedding_dim"]
        self.patch_size: int = self._config["patch_size"]
        self.n_heads: int = self._config["n_heads"]
        self.n_transformer_layers: int = self._config["n_transformer_layers"]
        self.dropout: float = self._config["dropout"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "embedding_dim": self.embedding_dim,
            "patch_size": self.patch_size,
            "n_heads": self.n_heads,
            "n_transformer_layers": self.n_transformer_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "train_batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "tolerance": self.tolerance,
            "save_frequency": self.save_frequency,
            "from_checkpoint": self.from_checkpoint,
            "saved_checkpoint_directory": self.saved_checkpoint_directory,
        }


class VAEConfig(BaseConfig):

    def __init__(self, context_group: Literal["wind", "mass", "thermal", "hydro", "precip"] | None):
        self.context_group: Literal["wind", "mass", "thermal", "hydro", "precip"] | None = context_group

        with open("./config.yaml", mode="r") as file:
            if self.context_group is None:
                self._config: dict[str, Any] = yaml.safe_load(file)[f"vae-target"]
            else:
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
            "from_checkpoint": self.from_checkpoint,
            "saved_checkpoint_directory": self.saved_checkpoint_directory,
        }
    

class DiffusionConfig(BaseConfig):

    def __init__(self) -> None:
        with open("./config.yaml", mode="r") as file:
            self._config: dict[str, Any] = yaml.safe_load(file)["diffusion"]
        
        super().__init__()
        self._load()

    def _load(self) -> None:
        self.target_dim: int = self._config["target_dim"]
        self.wind_condition_dim: int = self._config["wind_condition_dim"]
        self.mass_condition_dim: int = self._config["mass_condition_dim"]
        self.thermal_condition_dim: int = self._config["thermal_condition_dim"]
        self.hydro_condition_dim: int = self._config["hydro_condition_dim"]
        self.precip_condition_dim: int = self._config["precip_condition_dim"]
        self.n_condition_days: int = self._config["n_condition_days"]
        self.down_out_dims: list[int] = self._config["down_out_dims"]
        self.down_hidden_dims: list[int] = self._config["down_hidden_dims"]
        self.mid_out_dims: list[int] = self._config["mid_out_dims"]
        self.mid_hidden_dims: list[int] = self._config["mid_hidden_dims"]
        self.up_out_dims: list[int] = self._config["up_out_dims"]
        self.up_hidden_dims: list[int] = self._config["up_hidden_dims"]
        self.n_conv_layers_per_scaling_block: int = self._config["n_conv_layers_per_scaling_block"]
        self.n_transformer_encoder_layers_per_scaling_block: int = self._config["n_transformer_encoder_layers_per_scaling_block"]
        self.n_transformer_decoder_layers_per_scaling_block: int = self._config["n_transformer_decoder_layers_per_scaling_block"]
        self.n_conv_layers_per_mid_block: int = self._config["n_conv_layers_per_mid_block"]
        self.n_transformer_encoder_layers_per_mid_block: int = self._config["n_transformer_encoder_layers_per_mid_block"]
        self.n_transformer_decoder_layers_per_mid_block: int = self._config["n_transformer_decoder_layers_per_mid_block"]
        self.n_attention_heads: int = self._config["n_attention_heads"]
        self.max_sequence_length: int = self._config["max_sequence_length"]
        self.switch_ratio: float = float(self._config["switch_ratio"])
        self.noise_scheduler: Literal["linear", "cosine"] = self._config["noise_scheduler"]
        self.beta_min: float = float(self._config["beta_min"])
        self.beta_max: float = float(self._config["beta_max"])
        self.n_steps: int = self._config["n_steps"]
        self.eta: float = float(self._config["eta"])

        self.learning_rate: float = float(self._config["learning_rate"])
        self.vae_wind_checkpoint: pathlib.Path = pathlib.Path(self._config["vae_wind_checkpoint"])
        self.vae_mass_checkpoint: pathlib.Path = pathlib.Path(self._config["vae_mass_checkpoint"])
        self.vae_thermal_checkpoint: pathlib.Path = pathlib.Path(self._config["vae_thermal_checkpoint"])
        self.vae_hydro_checkpoint: pathlib.Path = pathlib.Path(self._config["vae_hydro_checkpoint"])
        self.vae_precip_checkpoint: pathlib.Path = pathlib.Path(self._config["vae_precip_checkpoint"])
        self.vae_target_checkpoint: pathlib.Path = pathlib.Path(self._config["vae_target_checkpoint"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_dim": self.target_dim,
            "wind_condition_dim": self.wind_condition_dim,
            "mass_condition_dim": self.mass_condition_dim,
            "thermal_condition_dim": self.thermal_condition_dim,
            "hydro_condition_dim": self.hydro_condition_dim,
            "precip_condition_dim": self.precip_condition_dim,
            "down_out_dims": self.down_out_dims,
            "down_hidden_dims": self.down_hidden_dims,
            "mid_out_dims": self.mid_out_dims,
            "mid_hidden_dims": self.mid_hidden_dims,
            "up_out_dims": self.up_out_dims,
            "up_hidden_dims": self.up_hidden_dims,
            "n_conv_layers_per_scaling_block": self.n_conv_layers_per_scaling_block,
            "n_transformer_encoder_layers_per_scaling_block": self.n_transformer_encoder_layers_per_scaling_block,
            "n_transformer_decoder_layers_per_scaling_block": self.n_transformer_decoder_layers_per_scaling_block,
            "n_conv_layers_per_mid_block": self.n_conv_layers_per_mid_block,
            "n_transformer_encoder_layers_per_mid_block": self.n_transformer_encoder_layers_per_mid_block,
            "n_transformer_decoder_layers_per_mid_block": self.n_transformer_decoder_layers_per_mid_block,
            "max_sequence_length": self.max_sequence_length,
            "n_attention_heads": self.n_attention_heads,
            "switch_ratio": self.switch_ratio,
            "noise_scheduler": self.noise_scheduler,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "n_steps": self.n_steps,
            "eta": self.eta,
            "learning_rate": self.learning_rate,
            "train_batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "tolerance": self.tolerance,
            "save_frequency": self.save_frequency,
            "vae_wind_checkpoint": self.vae_wind_checkpoint,
            "vae_mass_checkpoint": self.vae_mass_checkpoint,
            "vae_thermal_checkpoint": self.vae_thermal_checkpoint,
            "vae_hydro_checkpoint": self.vae_hydro_checkpoint,
            "vae_precip_checkpoint": self.vae_precip_checkpoint,
            "vae_target_checkpoint": self.vae_target_checkpoint,
            "from_checkpoint": self.from_checkpoint,
            "saved_checkpoint_directory": self.saved_checkpoint_directory,
        }
