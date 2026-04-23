"""
Tests for the RL layer (Phase 3).

Unit tests:
  * env observation / action shapes
  * reward sign (feasible design -> negative small reward, infeasible -> more negative)

Slow integration test:
  * train PPO for a modest budget on 10-bar, rollout must be feasible.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.registry import get_benchmark
from src.rl.environment import TrussDesignEnv


def test_env_spaces():
    b = get_benchmark("10bar")
    env = TrussDesignEnv(b)
    assert env.action_space.shape == (b.n_design_vars,)
    assert env.observation_space.shape == (1,)


def test_env_step_returns_correct_tuple():
    b = get_benchmark("10bar")
    env = TrussDesignEnv(b)
    env.reset(seed=42)
    action = np.zeros(b.n_design_vars, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (1,)
    assert np.isfinite(reward)
    assert terminated is True
    assert truncated is False
    assert "weight" in info
    assert "feasible" in info


def test_feasible_action_reward_is_higher_than_infeasible():
    """A big-area (stiff) design is feasible; a tiny-area design blows up."""
    b = get_benchmark("10bar")
    env = TrussDesignEnv(b)
    env.reset(seed=0)
    # action = +1.0 -> max areas -> definitely feasible, heavy weight
    feasible_action = np.full(b.n_design_vars, 1.0, dtype=np.float32)
    _, r_feasible, _, _, info_f = env.step(feasible_action)
    assert info_f["feasible"] is True

    env.reset(seed=0)
    # action = -1.0 -> minimum areas -> violates both stress & disp
    infeasible_action = np.full(b.n_design_vars, -1.0, dtype=np.float32)
    _, r_infeasible, _, _, info_if = env.step(infeasible_action)
    assert info_if["feasible"] is False
    assert r_feasible > r_infeasible


@pytest.mark.slow
def test_ppo_reaches_feasible_on_10bar():
    """Quick PPO run must return at least one feasible design."""
    from src.rl.train_ppo import train_ppo

    b = get_benchmark("10bar")
    result = train_ppo(b, total_timesteps=8_000, n_envs=4, seed=0)
    # With only 8000 timesteps we don't demand near-optimum, just feasibility.
    assert result.best_feasible is True, (
        "PPO failed to find a feasible design in 8000 timesteps."
    )
