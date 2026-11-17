from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SambaConfig:
    """Serializable SAMBA architecture configuration."""

    model_type: str = "samba"
    base_channels: int = 64
    bottleneck_channels: int = 128
    d_state: int = 16
    d_conv: int = 2
    expand: int = 8
    head_dim: int = 16
    cdse_hidden_dim: int = 32
    target_channel_names: list[str] | None = None
    target_positions: list[list[float]] | None = None
    reconstruction_channels: int | None = None
    reconstruction_mode: str = "target_montage"
    mixer_backend: str = "mamba2"
    sampling_rate: float | None = None
    checkpoint_variant: str | None = None

    def __post_init__(self) -> None:
        if self.target_positions is None:
            raise ValueError("target_positions must contain the shared CDSE target montage")
        if self.target_channel_names is not None and len(self.target_channel_names) != len(self.target_positions):
            raise ValueError("target_channel_names and target_positions must have the same length")
        if self.bottleneck_channels % self.head_dim:
            raise ValueError("bottleneck_channels must be divisible by head_dim")
        if self.reconstruction_mode not in {
            "target_montage",
            "fixed_input",
            "input_coordinates",
        }:
            raise ValueError(
                "reconstruction_mode must be target_montage, fixed_input, "
                "or input_coordinates"
            )
        if self.reconstruction_mode == "fixed_input" and not self.reconstruction_channels:
            raise ValueError("fixed_input reconstruction requires reconstruction_channels")

    @property
    def target_channels(self) -> int:
        return len(self.target_positions or [])

    @property
    def output_channels(self) -> int:
        if self.reconstruction_mode == "input_coordinates":
            return self.target_channels
        return self.reconstruction_channels or self.target_channels

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_pretrained(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "SambaConfig":
        return cls(**json.loads((Path(directory) / "config.json").read_text()))
