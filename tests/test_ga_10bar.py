"""
Phase 1 gate test: GA reproduces the Sunar 1991 optimum on 10-bar.

Uses 10 literature-standard seeds. Every run must:
  1. be feasible
  2. hit within 2.0% of 5060.85 lb

Expected wall time: ~15 s on a laptop with pop=100, n_gen=500. This
test is marked `slow` so it can be excluded from fast dev loops with
`pytest -m "not slow"`.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.runner import run
from src.benchmarks.registry import get_benchmark


SEEDS = [42, 123, 456, 789, 2024, 31415, 27182, 11235, 8675309, 9999]
TOL_PCT = 2.0
LIT_OPT = 5060.85


@pytest.mark.slow
def test_ga_10bar_across_10_seeds():
    bench = get_benchmark("10bar")

    weights = []
    feasible_flags = []
    for seed in SEEDS:
        r = run("ga", bench, seed=seed, pop_size=100, n_gen=500)
        weights.append(r.best_weight)
        feasible_flags.append(r.feasible)

    weights = np.array(weights)
    err_pct = 100.0 * (weights - LIT_OPT) / LIT_OPT

    print()
    print(f"GA 10-bar over {len(SEEDS)} seeds:")
    print(f"  min  = {weights.min():.2f} lb  (err {err_pct.min():+.2f} %)")
    print(f"  mean = {weights.mean():.2f} +/- {weights.std():.2f} lb")
    print(f"  max  = {weights.max():.2f} lb  (err {err_pct.max():+.2f} %)")
    print(f"  feasible: {sum(feasible_flags)} / {len(SEEDS)}")

    assert all(feasible_flags), (
        f"expected every run feasible; got {feasible_flags}"
    )
    assert err_pct.max() <= TOL_PCT, (
        f"GA exceeded {TOL_PCT}% tol on at least one seed: {err_pct.tolist()}"
    )
