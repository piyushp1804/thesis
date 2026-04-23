"""
Tests for the pymoo `TrussProblem` adapter.

The GA and PSO code only see `TrussProblem`; if it returns the wrong
shape or sign convention, every optimization silently does the wrong
thing. These tests pin the contract.
"""

from __future__ import annotations

import numpy as np

from src.algorithms.problem import TrussProblem
from src.benchmarks.registry import get_benchmark


def _dummy_out():
    return {"F": None, "G": None}


def test_single_objective_shapes():
    b = get_benchmark("10bar")
    p = TrussProblem(b, mode="single")
    assert p.n_var == b.n_design_vars
    assert p.n_obj == 1
    assert p.n_ieq_constr == 2

    x = b.initial_uniform_design()
    out = _dummy_out()
    p._evaluate(x, out)
    assert len(out["F"]) == 1
    assert len(out["G"]) == 2
    assert np.isfinite(out["F"]).all()
    assert np.isfinite(out["G"]).all()


def test_multi_objective_shapes():
    b = get_benchmark("10bar")
    p = TrussProblem(b, mode="multi")
    assert p.n_obj == 2
    x = b.initial_uniform_design()
    out = _dummy_out()
    p._evaluate(x, out)
    assert len(out["F"]) == 2
    # Second objective must be max displacement (non-negative).
    assert out["F"][1] >= 0


def test_constraint_signs_feasible_at_published_optimum():
    """At Sunar's optimum, stress/disp constraints are right at the edge."""
    b = get_benchmark("10bar")
    p = TrussProblem(b, mode="single")
    out = _dummy_out()
    p._evaluate(b.reference_optimum_areas, out)
    # G = ratio - 1; at the active constraint this is ~ 0.
    assert out["G"][0] == pytest.approx(0.0, abs=0.01)   # stress
    assert out["G"][1] == pytest.approx(0.0, abs=0.001)  # displacement


def test_custom_evaluator_is_called():
    """Verify we can swap in a fake evaluator (Phase 2 surrogate plumbing)."""
    b = get_benchmark("10bar")

    calls = {"n": 0}

    from src.benchmarks.base import BenchmarkEvaluation

    def fake_evaluator(x):
        calls["n"] += 1
        return BenchmarkEvaluation(
            weight=42.0,
            max_abs_stress=0.0,
            max_abs_displacement=0.0,
            member_abs_stresses=np.zeros(b.n_bars),
            member_abs_axial_forces=np.zeros(b.n_bars),
            per_bar_areas=np.zeros(b.n_bars),
        )

    p = TrussProblem(b, evaluator=fake_evaluator, mode="single")
    out = _dummy_out()
    p._evaluate(b.initial_uniform_design(), out)
    assert calls["n"] == 1
    assert out["F"][0] == 42.0


# `pytest` is imported lazily for `approx`.
import pytest  # noqa: E402
