"""
Small MLP surrogate for truss FEM outputs.

Architecture (deliberately tiny — trains in seconds, avoids over-fit):
    input  -> Linear(n_in, H) -> ReLU -> Linear(H, H) -> ReLU -> Linear(H, n_out)

Default hidden width H = 64. Three outputs: log1p(weight),
log1p(max_stress), log1p(max_disp). Log transform keeps the loss well-
conditioned because the raw outputs span several decades.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    n_inputs: int
    n_outputs: int = 3
    hidden: int = 64
    n_hidden_layers: int = 2


class MLP(nn.Module):
    """Generic configurable MLP. 3 outputs by default (weight/stress/disp)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = cfg.n_inputs
        for _ in range(cfg.n_hidden_layers):
            layers.append(nn.Linear(last, cfg.hidden))
            layers.append(nn.ReLU(inplace=True))
            last = cfg.hidden
        layers.append(nn.Linear(last, cfg.n_outputs))
        self.net = nn.Sequential(*layers)
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, n_inputs) -> (B, n_outputs)
        return self.net(x)
