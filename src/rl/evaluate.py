"""
Greedy rollout of a trained PPO policy for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from stable_baselines3 import PPO

from src.benchmarks.base import BenchmarkProblem

from .environment import Evaluator, TrussDesignEnv


@dataclass
class RolloutResult:
    weight: float
    feasible: bool
    max_stress: float
    max_disp: float
    x: np.ndarray


def rollout_policy(
    model: PPO,
    benchmark: BenchmarkProblem,
    evaluator: Evaluator | None = None,
    n_rollouts: int = 32,
    deterministic: bool = True,
    seed: int = 0,
) -> RolloutResult:
    """Run the policy `n_rollouts` times; return best feasible.

    We run one deterministic rollout (the policy's mean action) plus
    `n_rollouts - 1` stochastic rollouts. Bandit PPO policies tend to
    settle near the constraint edge; stochastic sampling explores both
    sides and retrieves feasible designs more reliably.
    """
    env = TrussDesignEnv(benchmark, evaluator=evaluator)

    best = None
    for i in range(n_rollouts):
        obs, _ = env.reset(seed=seed + i)
        det = deterministic and (i == 0)
        action, _ = model.predict(obs, deterministic=det)
        _, _, _, _, info = env.step(action)
        if info["feasible"] and (best is None or info["weight"] < best["weight"]):
            best = info

    if best is None:
        return RolloutResult(
            weight=float("nan"),
            feasible=False,
            max_stress=float("nan"),
            max_disp=float("nan"),
            x=np.full(benchmark.n_design_vars, float("nan")),
        )
    return RolloutResult(
        weight=float(best["weight"]),
        feasible=True,
        max_stress=float(best["max_stress"]),
        max_disp=float(best["max_disp"]),
        x=np.asarray(best["x"], dtype=float),
    )
