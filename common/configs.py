from abc import ABC, abstractmethod
import yaml

import pathlib
from itertools import product
from typing import *


class BaseConfig(ABC):

    def __init__(self):
        raise SyntaxError(f"{self.__class__.__name__} is not meant for initilization")

    @abstractmethod
    def _load(self) -> None:
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

class MetaData(BaseConfig):

    def __init__(self, tp: Literal["train", "val", "test"]):
        with open("./config.yaml", mode="r") as file:
            self.__config: Dict[str, Any] = yaml.safe_load(file)["cesm2"]

        self.tp: Literal["train", "val", "test"] = tp
        self._load()
        self.var_names: List[str] = sorted(set(self.input_vars + self.output_vars))
        self.years: List[int] = list(range(self.start_year, self.end_year + 1))
        self.combinations: List[Tuple[str, str, int]] = list(product(self.sim_ids, self.var_names, self.years))
        self.n_years: int = len(self.years)

    def _load(self) -> None:
        self.device: str = self.__config["device"]
        self.input_vars: List[str] = self.__config["input_vars"]
        self.output_vars: List[str] = self.__config["output_vars"]
        self.sim_ids: List[str] = self.__config["sim_ids"]

        if self.tp == "train":
            self.start_year: int = self.__config["train_start_year"]
            self.end_year: int = self.__config["train_end_year"]
            self.write_directory: pathlib.Path = pathlib.Path(self.__config["train_write_directory"])
        elif self.tp == "val":
            self.start_year: int = self.__config["val_start_year"]
            self.end_year: int = self.__config["val_end_year"]
            self.write_directory: pathlib.Path = pathlib.Path(self.__config["val_write_directory"])
        elif self.tp == "test":
            self.start_year: int = self.__config["test_start_year"]
            self.end_year: int = self.__config["test_end_year"]
            self.write_directory: pathlib.Path = pathlib.Path(self.__config["test_write_directory"])
        else:
            raise ValueError(f"Invalid tp for MetaData, expected one of ['train', 'val', 'test'], get: '{self.tp}'")

        self.n_input_days: int = self.__config["n_input_days"]
        self.n_lead_days: int = self.__config["n_lead_days"]
        self.n_output_days: int = self.__config["n_output_days"]
        self.n_step_days: int = self.__config["n_step_days"]
        self.climatological_window_size: int = self.__config["climatological_window_size"]

        self.detrender_state_directory: pathlib.Path = pathlib.Path(self.__config["detrender_state_directory"])
        self.climatology_state_directory: pathlib.Path = pathlib.Path(self.__config["climatology_state_directory"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "device": self.device,
            "input_vars": self.input_vars,
            "output_vars": self.output_vars,
            "sim_ids": self.sim_ids,
            "years": self.years,
            "n_input_days": self.n_input_days,
            "n_lead_days": self.n_lead_days,
            "n_output_days": self.n_output_days,
            "n_step_days": self.n_step_days,
            "climatological_window_size": self.climatological_window_size,
        }


class CNNConfig(BaseConfig):

    def __init__(self):
        with open("./config.yaml", mode="r") as file:
            self.__config: Dict[str, Any] = yaml.safe_load(file)["cnn"]
        
        self._load()

    def _load(self) -> None:
        self.device: str = self.__config["device"]
        self.in_features: int = self.__config["in_features"]
        self.out_features: int = self.__config["out_features"]
        self.embedding_dim: int = self.__config["embedding_dim"]
        self.n_hidden_layers: int = self.__config["n_hidden_layers"]

        self.learning_rate: float = float(self.__config["learning_rate"])
        self.train_batch_size: int = self.__config["train_batch_size"]
        self.val_batch_size: int = self.__config["val_batch_size"]
        self.n_epochs: int = self.__config["n_epochs"]
        self.patience: float = self.__config["patience"]
        self.tolerance: float = float(self.__config["tolerance"])
        self.save_frequency: int = self.__config["save_frequency"]
        self.from_checkpoint: pathlib.Path | None = pathlib.Path(value) if (value := self.__config["from_checkpoint"]) else None
        self.saved_checkpoint_directory: pathlib.Path = pathlib.Path(self.__config["saved_checkpoint_directory"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
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

    def __init__(self):
        with open("./config.yaml", mode="r") as file:
            self.__config: Dict[str, Any] = yaml.safe_load(file)["unet"]
        
        self._load()

    def _load(self) -> None:
        self.device: str = self.__config["device"]
        self.in_features: int = self.__config["in_features"]
        self.out_features: int = self.__config["out_features"]
        self.embedding_dim: int = self.__config["embedding_dim"]

        self.learning_rate: float = float(self.__config["learning_rate"])
        self.train_batch_size: int = self.__config["train_batch_size"]
        self.val_batch_size: int = self.__config["val_batch_size"]
        self.n_epochs: int = self.__config["n_epochs"]
        self.patience: float = self.__config["patience"]
        self.tolerance: float = float(self.__config["tolerance"])
        self.save_frequency: int = self.__config["save_frequency"]
        self.from_checkpoint: pathlib.Path | None = pathlib.Path(value) if (value := self.__config["from_checkpoint"]) else None
        self.saved_checkpoint_directory: pathlib.Path = pathlib.Path(self.__config["saved_checkpoint_directory"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
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

    def __init__(self):
        with open("./config.yaml", mode="r") as file:
            self.__config: Dict[str, Any] = yaml.safe_load(file)["vit"]
        
        self._load()

    def _load(self) -> None:
        self.device: str = self.__config["device"]
        self.in_features: int = self.__config["in_features"]
        self.out_features: int = self.__config["out_features"]
        self.embedding_dim: int = self.__config["embedding_dim"]
        self.patch_size: int = self.__config["patch_size"]
        self.n_heads: int = self.__config["n_heads"]
        self.n_transformer_layers: int = self.__config["n_transformer_layers"]
        self.dropout: float = self.__config["dropout"]

        self.learning_rate: float = float(self.__config["learning_rate"])
        self.train_batch_size: int = self.__config["train_batch_size"]
        self.val_batch_size: int = self.__config["val_batch_size"]
        self.n_epochs: int = self.__config["n_epochs"]
        self.patience: float = self.__config["patience"]
        self.tolerance: float = float(self.__config["tolerance"])
        self.save_frequency: int = self.__config["save_frequency"]
        self.from_checkpoint: pathlib.Path | None = pathlib.Path(value) if (value := self.__config["from_checkpoint"]) else None
        self.saved_checkpoint_directory: pathlib.Path = pathlib.Path(self.__config["saved_checkpoint_directory"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
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


class VAEContextConfig(BaseConfig):

    def __init__(self):
        with open("./config.yaml", mode="r") as file:
            self.__config: Dict[str, Any] = yaml.safe_load(file)["vae-context"]
        
        self._load()

    def _load(self) -> None:
        self.device: str = self.__config["device"]
        self.latent_dim: int = self.__config["latent_dim"]
        self.hidden_dim: int = self.__config["hidden_dim"]
        self.n_scaling_blocks: int = self.__config["n_scaling_blocks"]
        self.n_convstack_layers: int = self.__config["n_convstack_layers"]
        self.n_convhead_layers: int = self.__config["n_convhead_layers"]

        self.lambda_: float = self.__config["lambda"]
        self.learning_rate: float = float(self.__config["learning_rate"])
        self.train_batch_size: int = self.__config["train_batch_size"]
        self.val_batch_size: int = self.__config["val_batch_size"]
        self.n_epochs: int = self.__config["n_epochs"]
        self.patience: float = self.__config["patience"]
        self.tolerance: float = float(self.__config["tolerance"])
        self.save_frequency: int = self.__config["save_frequency"]
        self.from_checkpoint: pathlib.Path | None = pathlib.Path(value) if (value := self.__config["from_checkpoint"]) else None
        self.saved_checkpoint_directory: pathlib.Path = pathlib.Path(self.__config["saved_checkpoint_directory"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
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
    

class VAETargetConfig(BaseConfig):

    def __init__(self):
        with open("./config.yaml", mode="r") as file:
            self.__config: Dict[str, Any] = yaml.safe_load(file)["vae-target"]
        
        self._load()

    def _load(self) -> None:
        self.device: str = self.__config["device"]
        self.latent_dim: int = self.__config["latent_dim"]
        self.hidden_dim: int = self.__config["hidden_dim"]
        self.n_scaling_blocks: int = self.__config["n_scaling_blocks"]
        self.n_convstack_layers: int = self.__config["n_convstack_layers"]
        self.n_convhead_layers: int = self.__config["n_convhead_layers"]
        
        self.lambda_: float = self.__config["lambda"]
        self.learning_rate: float = float(self.__config["learning_rate"])
        self.train_batch_size: int = self.__config["train_batch_size"]
        self.val_batch_size: int = self.__config["val_batch_size"]
        self.n_epochs: int = self.__config["n_epochs"]
        self.patience: float = self.__config["patience"]
        self.tolerance: float = float(self.__config["tolerance"])
        self.save_frequency: int = self.__config["save_frequency"]
        self.from_checkpoint: pathlib.Path | None = pathlib.Path(value) if (value := self.__config["from_checkpoint"]) else None
        self.saved_checkpoint_directory: pathlib.Path = pathlib.Path(self.__config["saved_checkpoint_directory"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
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


class DDPMConfig(BaseConfig):

    def __init__(self):
        with open("./config.yaml", mode="r") as file:
            self.__config: Dict[str, Any] = yaml.safe_load(file)["ddpm"]
        
        self._load()

    def _load(self) -> None:
        self.device: str = self.__config["device"]
        self.target_in_dim: int = self.__config["target_in_dim"]
        self.condition_in_dim: int = self.__config["condition_in_dim"]
        self.step_in_dim: int = self.__config["step_in_dim"]
        self.down_out_dims: List[int] = self.__config["down_out_dims"]
        self.down_hidden_dims: List[int] = self.__config["down_hidden_dims"]
        self.mid_out_dims: List[int] = self.__config["mid_out_dims"]
        self.mid_hidden_dims: List[int] = self.__config["mid_hidden_dims"]
        self.up_out_dims: List[int] = self.__config["up_out_dims"]
        self.up_hidden_dims: List[int] = self.__config["up_hidden_dims"]
        self.n_layers_per_scaling_block: int = self.__config["n_layers_per_scaling_block"]
        self.n_layers_per_mid_block: int = self.__config["n_layers_per_mid_block"]
        self.n_attention_heads: int = self.__config["n_attention_heads"]
        self.condition_dropout: float = float(self.__config["condition_dropout"])
        self.n_steps: int = self.__config["n_steps"]
        self.noise_scheduler_scheme: Literal["linear", "cosine"] = self.__config["noise_scheduler_scheme"]
        self.beta_min: float = float(self.__config["beta_min"])
        self.beta_max: float = float(self.__config["beta_max"])

        self.learning_rate: float = float(self.__config["learning_rate"])
        self.train_batch_size: int = self.__config["train_batch_size"]
        self.val_batch_size: int = self.__config["val_batch_size"]
        self.n_epochs: int = self.__config["n_epochs"]
        self.patience: int = self.__config["patience"]
        self.tolerance: float = float(self.__config["tolerance"])
        self.save_frequency: int = self.__config["save_frequency"]

        self.target_vae_checkpoint: pathlib.Path = pathlib.Path(self.__config["target_vae_checkpoint"])
        self.context_vae_checkpoint: pathlib.Path = pathlib.Path(self.__config["context_vae_checkpoint"])
        self.from_checkpoint: pathlib.Path | None = pathlib.Path(value) if (value := self.__config["from_checkpoint"]) else None
        self.saved_checkpoint_directory: pathlib.Path = pathlib.Path(self.__config["saved_checkpoint_directory"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "target_in_dim": self.target_in_dim,
            "condition_in_dim": self.condition_in_dim,
            "step_in_dim": self.step_in_dim,
            "down_out_dims": self.down_out_dims,
            "down_hidden_dims": self.down_hidden_dims,
            "mid_out_dims": self.mid_out_dims,
            "mid_hidden_dims": self.mid_hidden_dims,
            "up_out_dims": self.up_out_dims,
            "up_hidden_dims": self.up_hidden_dims,
            "n_layers_per_scaling_block": self.n_layers_per_scaling_block,
            "n_layers_per_mid_block": self.n_layers_per_mid_block,
            "n_attention_heads": self.n_attention_heads,
            "condition_dropout": self.condition_dropout,
            "n_steps": self.n_steps,
            "noise_scheduler_scheme": self.noise_scheduler_scheme,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "learning_rate": self.learning_rate,
            "train_batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "n_epochs": self.n_epochs,
            "patience": self.patience,
            "tolerance": self.tolerance,
            "save_frequency": self.save_frequency,
            "target_vae_checkpoint": self.target_vae_checkpoint,
            "context_vae_checkpoint": self.context_vae_checkpoint,
            "from_checkpoint": self.from_checkpoint,
            "saved_checkpoint_directory": self.saved_checkpoint_directory,
        }
