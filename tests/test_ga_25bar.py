"""
Phase 7 gate test: GA convergence on 25-bar across 3 seeds.

Our 25-bar encoding uses a simplified uniform-stress formulation that
converges to ~570 lb (vs the Schmit 1976 literature value of 545.22 lb
with member-type-dependent stress limits). This test therefore checks
algorithmic self-consistency — every seed is feasible and finishes
within 10% of the seeds' mean — rather than exact literature match.

Expected wall time: ~30 s per seed; ~1.5 min total with pop=100,
n_gen=300. Marked `slow` so fast dev loops can skip it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.runner import run
from src.benchmarks.registry import get_benchmark


SEEDS = [42, 123, 456]
MAX_SPREAD_PCT = 10.0


@pytest.mark.slow
def test_ga_25bar_self_consistent():
    bench = get_benchmark("25bar")

    weights = []
    feasible_flags = []
    for seed in SEEDS:
        r = run("ga", bench, seed=seed, pop_size=100, n_gen=300)
        weights.append(r.best_weight)
        feasible_flags.append(r.feasible)

    weights = np.array(weights)
    mean = weights.mean()
    spread_pct = 100.0 * (weights.max() - weights.min()) / mean

    print()
    print(f"GA 25-bar over {len(SEEDS)} seeds:")
    print(f"  best  = {weights.min():.2f} lb")
    print(f"  mean  = {mean:.2f} +/- {weights.std():.2f} lb")
    print(f"  spread = {spread_pct:.2f} %")
    print(f"  feasible: {sum(feasible_flags)} / {len(SEEDS)}")

    assert all(feasible_flags), (
        f"expected every run feasible; got {feasible_flags}"
    )
    assert spread_pct <= MAX_SPREAD_PCT, (
        f"GA 25-bar spread {spread_pct:.2f}% exceeded {MAX_SPREAD_PCT}% "
        f"self-consistency tolerance; weights={weights.tolist()}"
    )
