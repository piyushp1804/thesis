"""
Tests for the neural surrogate (Phase 2).

Two styles:
  * pure unit tests (model forward shape, dataset LHS shape) — fast.
  * one integration test that trains on a small dataset and asserts
    R^2(weight) > 0.9 — marked `slow`. Full R^2 > 0.98 gate lives in
    `scripts/train_surrogate.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.benchmarks.registry import get_benchmark
from src.ml.dataset import generate_dataset
from src.ml.model import MLP, ModelConfig
from src.ml.surrogate import SurrogateEvaluator
from src.ml.train import train_surrogate


def test_mlp_forward_shape():
    cfg = ModelConfig(n_inputs=10, n_outputs=3, hidden=32)
    model = MLP(cfg)
    x = torch.randn(8, 10)
    y = model(x)
    assert y.shape == (8, 3)


def test_dataset_shapes():
    bench = get_benchmark("10bar")
    ds = generate_dataset(bench, n_samples=20, seed=0, progress=False)
    assert ds.X.shape == (20, bench.n_design_vars)
    assert ds.weight.shape == (20,)
    assert ds.max_stress.shape == (20,)
    assert ds.max_disp.shape == (20,)
    assert ds.member_stress.shape == (20, bench.n_bars)
    assert (ds.weight > 0).all()


def test_surrogate_roundtrip_shape():
    bench = get_benchmark("10bar")
    ds = generate_dataset(bench, n_samples=128, seed=0, progress=False)
    s = train_surrogate(ds, seed=0, epochs=20, verbose=False)
    ev = SurrogateEvaluator(bench, s)
    out = ev(ds.X[0])
    assert np.isfinite(out.weight)
    assert np.isfinite(out.max_abs_stress)
    assert np.isfinite(out.max_abs_displacement)
    assert out.per_bar_areas.shape == (bench.n_bars,)


@pytest.mark.slow
def test_surrogate_trains_on_small_10bar():
    """Integration: 1000 LHS + 150 epochs must reach R^2(weight) > 0.9.

    Weight is a smooth linear function of areas, so the MLP nails it.
    Stress/displacement are near-discontinuous at small areas; we don't
    gate on them here (see docs/ground_truth.md Phase 2 targets — they
    are realistic on larger benchmarks, not 10-bar).
    """
    bench = get_benchmark("10bar")
    ds = generate_dataset(bench, n_samples=1000, seed=0, progress=False)
    s = train_surrogate(ds, seed=0, epochs=150, verbose=False)
    assert s.val_r2["weight"] > 0.9, (
        f"val R^2(weight) was {s.val_r2['weight']:.3f}"
    )
