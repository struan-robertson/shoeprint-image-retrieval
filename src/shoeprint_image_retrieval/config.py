"""Module to handle loading config toml file."""

from pathlib import Path
from typing import TypedDict

import toml

from .customtypes import DatasetTypeType


class DatasetConfig(TypedDict, total=True):
    """Dataset configuration."""

    dir: str
    type: DatasetTypeType
    crop: list[int]
    n_processes: int
    n_clusters: int
    cluster_minimise_tolerance: float


class ModelConfig(TypedDict, total=True):
    """Model configuration."""

    type: str
    clahe_clip_limit: float
    clahe_tile_grid_size: list[int]
    start_block: int
    end_block: int
    skip_blocks: list[int]
    minimum_dim: int
    maximum_dim: int


class ComparisonConfig(TypedDict, total=True):
    """Comparison configuration."""

    n_processes: int
    rotations: list[int] | None
    scales: list[float] | None


class Config(TypedDict, total=True):
    """Config for entire system."""

    dataset: DatasetConfig
    model: ModelConfig
    comparison: ComparisonConfig


def load_config(config_file: Path | str):
    """Load a config .toml file.

    Args:
        config_file: TOML file to load.

    """
    with Path(config_file).open() as file:
        raw_config = toml.load(file)
        if raw_config["comparison"]["rotations"] == "":
            raw_config["comparison"]["rotations"] = None
        if raw_config["comparison"]["scales"] == "":
            raw_config["comparison"]["scales"] = None
        return Config(raw_config)  # pyright: ignore[reportArgumentType]
