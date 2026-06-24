from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MechanicalPropertyMLPConfig:
    input_dim: int = 1280
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.1

    def validate(self) -> None:
        if int(self.input_dim) <= 0:
            raise ValueError("input_dim must be positive.")
        if not self.hidden_dims or any(int(dim) <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["hidden_dims"] = list(self.hidden_dims)
        return payload


class MechanicalPropertyMLP(nn.Module):
    """ESM2 pooled embedding -> strength and toughness regression heads."""

    def __init__(self, config: MechanicalPropertyMLPConfig) -> None:
        super().__init__()
        config.validate()
        layers: list[nn.Module] = []
        current_dim = int(config.input_dim)
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.GELU())
            if config.dropout > 0:
                layers.append(nn.Dropout(float(config.dropout)))
            current_dim = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.strength_head = nn.Linear(current_dim, 1)
        self.toughness_head = nn.Linear(current_dim, 1)
        self.config = config

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        features = self.backbone(embeddings)
        strength = self.strength_head(features)
        toughness = self.toughness_head(features)
        return torch.cat([strength, toughness], dim=-1)
