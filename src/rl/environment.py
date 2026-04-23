"""
Gymnasium environment for RL-based truss sizing.

Design choice: **single-step bandit episode**. Each episode the agent
outputs one design vector (action = normalized areas in [-1, 1]), we
un-normalize to the benchmark bounds, evaluate (via FEM or surrogate),
and return the reward. This keeps the environment trivial to implement
and lets PPO learn a direct policy over designs — which is exactly the
RL formulation used by the thesis.

Reward shaping (cheap and transparent):
    r = - weight / W_scale
        - lambda_stress * max(0, stress/stress_limit - 1) ** 2
        - lambda_disp   * max(0, disp/disp_limit - 1) ** 2
If feasible: r ~ -weight/W_scale. If infeasible, additive quadratic
penalties grow smoothly so PPO still gets useful gradients.

Observation: constant 1-D vector (we don't have state to observe in a
bandit setup). We expose a 1-D observation of the action bounds so
that SB3's policy network has a non-trivial input tensor. PPO's policy
reduces to a state-independent mean over actions, which is what we
want for this formulation.
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np

from src.benchmarks.base import BenchmarkEvaluation, BenchmarkProblem


Evaluator = Callable[[np.ndarray], BenchmarkEvaluation]


class TrussDesignEnv(gym.Env):
    """Single-step design bandit for PPO."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        benchmark: BenchmarkProblem,
        evaluator: Evaluator | None = None,
        weight_scale: float | None = None,
        lambda_stress: float = 100.0,
        lambda_disp: float = 100.0,
    ) -> None:
        super().__init__()
        self.benchmark = benchmark
        self.evaluator: Evaluator = evaluator or benchmark.evaluate

        # Action = normalized areas in [-1, 1].
        n_var = benchmark.n_design_vars
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(n_var,), dtype=np.float32
        )
        # Observation is a fixed scalar 'phase' vector — not meaningful for
        # bandit, but SB3 needs a non-trivial observation.
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        lo, hi = benchmark.area_bounds
        self._lo = float(lo)
        self._hi = float(hi)

        # Weight scale: auto from uniform mid-design weight.
        if weight_scale is None:
            mid = benchmark.initial_uniform_design()
            weight_scale = max(1.0, float(benchmark.evaluate(mid).weight))
        self.weight_scale = float(weight_scale)

        self.lambda_stress = float(lambda_stress)
        self.lambda_disp = float(lambda_disp)

        # Track best observed design for evaluation wrappers.
        self.best_weight: float = float("inf")
        self.best_x: np.ndarray | None = None
        self.best_feasible: bool = False

        self._rng = np.random.default_rng()

    # ---------- gym API ----------

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        # un-normalize from [-1, 1] to [lo, hi]
        x = self._lo + 0.5 * (action + 1.0) * (self._hi - self._lo)
        x = np.clip(x, self._lo, self._hi)

        ev = self.evaluator(x)
        reward, feasible = self._reward(ev)

        if feasible and ev.weight < self.best_weight:
            self.best_weight = float(ev.weight)
            self.best_x = x.copy()
            self.best_feasible = True

        obs = np.array([0.0], dtype=np.float32)
        terminated = True
        truncated = False
        info = {
            "weight": float(ev.weight),
            "max_stress": float(ev.max_abs_stress),
            "max_disp": float(ev.max_abs_displacement),
            "feasible": bool(feasible),
            "x": x,
        }
        return obs, float(reward), terminated, truncated, info

    # ---------- reward ----------

    def _reward(self, ev: BenchmarkEvaluation) -> tuple[float, bool]:
        b = self.benchmark
        stress_limit = max(b.stress_limit_tension, b.stress_limit_compression)
        stress_violation = max(0.0, ev.max_abs_stress / stress_limit - 1.0)
        disp_violation = max(0.0, ev.max_abs_displacement / b.displacement_limit - 1.0)
        feasible = (stress_violation == 0.0) and (disp_violation == 0.0)

        reward = -ev.weight / self.weight_scale
        reward -= self.lambda_stress * (stress_violation ** 2)
        reward -= self.lambda_disp * (disp_violation ** 2)
        return reward, feasible
