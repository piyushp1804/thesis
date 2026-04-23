"""
Uniform result dataclass for every optimizer (GA, PSO, NSGA-II, RL, LLM).

Having one shape means scripts / plotting / tests don't care *which*
algorithm ran — they only need the standard fields. This is the glue
that makes Chapter 4's cross-algorithm tables apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OptimizationResult:
    """Standardized optimizer output."""

    algorithm: str                    # "ga" | "pso" | "nsga2" | "ppo" | "llm+ga"
    benchmark: str                    # e.g. "10bar"
    seed: int

    best_x: np.ndarray                # shape (n_design_vars,)
    best_weight: float
    feasible: bool
    max_stress: float
    max_displacement: float

    pareto_x: np.ndarray | None = None     # (k, n_design_vars)
    pareto_f: np.ndarray | None = None     # (k, n_obj)

    history: list[dict[str, Any]] = field(default_factory=list)

    wall_time_s: float = 0.0
    n_evals: int = 0

    def to_summary_dict(self) -> dict[str, Any]:
        """Flatten for pandas tables — leaves arrays out."""
        return {
            "algorithm": self.algorithm,
            "benchmark": self.benchmark,
            "seed": self.seed,
            "best_weight": self.best_weight,
            "feasible": self.feasible,
            "max_stress": self.max_stress,
            "max_displacement": self.max_displacement,
            "wall_time_s": self.wall_time_s,
            "n_evals": self.n_evals,
            "pareto_n_points": 0 if self.pareto_f is None else len(self.pareto_f),
        }
