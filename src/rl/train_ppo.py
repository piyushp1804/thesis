"""
PPO training driver (Stable-Baselines3) for the truss design env.

Defaults are small so a full training fits in under a minute on CPU
for 10-bar: 50 000 env steps with 8 parallel envs. Increase
`total_timesteps` on bigger benchmarks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.benchmarks.base import BenchmarkProblem

from .environment import Evaluator, TrussDesignEnv


@dataclass
class PPOResult:
    model: PPO
    best_x: np.ndarray | None
    best_weight: float
    best_feasible: bool


def _make_env_factory(
    benchmark: BenchmarkProblem,
    evaluator: Evaluator | None,
    weight_scale: float | None,
) -> Callable[[], TrussDesignEnv]:
    def _thunk():
        return TrussDesignEnv(
            benchmark,
            evaluator=evaluator,
            weight_scale=weight_scale,
        )

    return _thunk


def train_ppo(
    benchmark: BenchmarkProblem,
    evaluator: Evaluator | None = None,
    total_timesteps: int = 50_000,
    n_envs: int = 8,
    seed: int = 0,
    lr: float = 3e-4,
    verbose: int = 0,
) -> PPOResult:
    """Train PPO on `benchmark`. Returns best-seen feasible design."""
    factory = _make_env_factory(benchmark, evaluator, weight_scale=None)
    vec = DummyVecEnv([factory for _ in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        vec,
        learning_rate=lr,
        n_steps=64,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        seed=seed,
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    # Extract the best-ever design seen across every worker env.
    best_weight = float("inf")
    best_x = None
    best_feasible = False
    for env in vec.envs:
        if env.best_feasible and env.best_weight < best_weight:
            best_weight = env.best_weight
            best_x = env.best_x
            best_feasible = True

    return PPOResult(
        model=model,
        best_x=best_x,
        best_weight=float(best_weight) if best_feasible else float("nan"),
        best_feasible=best_feasible,
    )


def save_model(model: PPO, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def load_model(path: str | Path) -> PPO:
    return PPO.load(str(path))
