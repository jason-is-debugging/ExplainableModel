"""Hierarchical configuration using pydantic-settings.

Mirrors the YAML structure exactly:

    model:    -> TransformerModelConfig (architecture + tokenizer)
    train:    -> TrainConfig           (training loop hyperparameters)
    dataset:  -> DatasetConfig         (data paths)
    save:     -> SaveConfig            (checkpoint paths)

Built on :class:`pydantic_settings.BaseSettings`, so it automatically:

  * loads ``config.yaml`` on instantiation
  * validates types (e.g. ``d_model: "abc"`` fails fast)
  * applies environment-variable overrides (``APP_TRAIN__BATCH_SIZE=64``)
  * falls back to dataclass-style defaults when keys are missing

Usage:

    cfg = AppConfig()                 # auto-loads config.yaml
    print(cfg.model.d_model)
    print(cfg.train.learning_rate)

Environment overrides::

    APP_TRAIN__BATCH_SIZE=64 APP_MODEL__D_MODEL=512 python main.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


# ────────────────────────────────────────────────────────────────────────────
#  Default config file location
# ────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ExplainableModel/
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


# ────────────────────────────────────────────────────────────────────────────
#  Nested configs (plain BaseModels — no env-binding, no file loading)
# ────────────────────────────────────────────────────────────────────────────
class TokenizerConfig(BaseModel):
    """Special token IDs shared by the tokenizer and the dataset."""
    model_config = {"populate_by_name": True}

    pad_token_id: int = Field(default=0, alias="pad-token-id")
    bos_token_id: int = Field(default=1, alias="bos-token-id")
    eos_token_id: int = Field(default=2, alias="eos-token-id")


class TransformerModelConfig(BaseModel):
    """Transformer architecture hyperparameters."""
    model_config = {"populate_by_name": True}

    vocab_size: int = Field(default=10000, alias="vocab-size")
    max_seq_len: int = Field(default=256, alias="max-seq-len")

    d_model: int = Field(default=256, alias="d-model")
    nhead: int = Field(default=8, alias="nhead")
    num_decoder_layers: int = Field(default=6, alias="num-decoder-layers")
    dim_feedforward: int = Field(default=1024, alias="dim-feedforward")
    dropout: float = Field(default=0.1, alias="dropout")

    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig, alias="tokenizer")


class TrainConfig(BaseModel):
    """Training-loop hyperparameters."""
    model_config = {"populate_by_name": True}

    batch_size: int = Field(default=32, alias="batch-size")
    learning_rate: float = Field(default=1e-4, alias="learning-rate")
    num_epochs: int = Field(default=10, alias="num-epochs")
    warmup_steps: int = Field(default=500, alias="warmup-steps")
    accumulation_steps: int = Field(default=1, alias="accumulation-steps")


class DatasetConfig(BaseModel):
    """Where to find / place dataset files."""
    model_config = {"populate_by_name": True}

    dir_path: str = Field(default="dataset", alias="dir-path")
    file_name: str = Field(default="dialogue.txt", alias="file-name")
    dataset_name: str = Field(default="cornell", alias="dataset-name")


class SaveConfig(BaseModel):
    """Where to save checkpoints."""
    model_config = {"populate_by_name": True}

    checkpoints_dir: str = Field(default="checkpoints", alias="checkpoints-dir")


# ────────────────────────────────────────────────────────────────────────────
#  Top-level settings (auto-loads YAML + env vars)
# ────────────────────────────────────────────────────────────────────────────
class AppConfig(BaseSettings):
    """Root configuration mirroring the YAML sections.

    Inheriting from :class:`BaseSettings` makes this auto-loadable from
    ``config.yaml`` and overridable via environment variables.
    """
    model:    TransformerModelConfig = Field(default_factory=TransformerModelConfig)
    train:    TrainConfig            = Field(default_factory=TrainConfig)
    dataset:  DatasetConfig          = Field(default_factory=DatasetConfig)
    save:     SaveConfig             = Field(default_factory=SaveConfig)

    model_config = SettingsConfigDict(
        yaml_file=str(_DEFAULT_CONFIG_PATH),
        env_prefix="APP_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",          # unknown YAML keys are silently dropped
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Wire :class:`YamlConfigSettingsSource` into the resolution chain.

        Order (highest priority first):
            init kwargs > env vars > YAML file > defaults
        """
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


# ────────────────────────────────────────────────────────────────────────────
#  Convenience helpers
# ────────────────────────────────────────────────────────────────────────────
def load_config(path: str | Path | None = None) -> AppConfig:
    """Load the configuration and return an :class:`AppConfig` instance.

    Args:
        path: Optional override path to a YAML file. If ``None`` the
            default ``<project_root>/config.yaml`` is used.

    Returns:
        A fully-validated :class:`AppConfig`.
    """
    if path is None:
        return AppConfig()
    return AppConfig(_yaml_file=str(Path(path)))


__all__ = [
    "AppConfig",
    "TokenizerConfig",
    "TransformerModelConfig",
    "TrainConfig",
    "DatasetConfig",
    "SaveConfig",
    "load_config",
]