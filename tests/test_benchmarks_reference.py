"""
Reproduce every *verified* benchmark's published optimum at FEM level.

This is the single strongest sanity check we have: plug the published
areas into our FEM, recompute the weight, and confirm it matches the
paper. If this drifts, every downstream result is suspect.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.registry import get_benchmark


@pytest.mark.parametrize(
    "name, tol_pct",
    [
        ("10bar", 0.05),  # Sunar 1991 rounds areas to 3 dp; we reproduce within 0.05%.
    ],
)
def test_published_optimum_reproduces_weight(name, tol_pct):
    b = get_benchmark(name)
    assert b.reference_verified, f"{name} is not a verified benchmark"
    assert b.reference_optimum_areas is not None

    ev = b.evaluate(b.reference_optimum_areas)
    target = b.reference_optimum_weight
    err_pct = 100.0 * abs(ev.weight - target) / target
    assert err_pct < tol_pct, (
        f"{name}: reproduced weight {ev.weight:.3f} vs "
        f"literature {target:.3f} -> {err_pct:.3f}% (tol {tol_pct}%)"
    )


def test_10bar_published_optimum_is_near_active_constraints():
    """Sunar's optimum is displacement-active by design."""
    b = get_benchmark("10bar")
    ev = b.evaluate(b.reference_optimum_areas)
    # Displacement limit = 2.0 in, and the published optimum rides the edge.
    assert 1.99 <= ev.max_abs_displacement <= 2.01
    # Stress at the published optimum sits right at the 25 ksi bound
    # (slightly above, per Sunar's 4-digit rounding).
    assert 24_900 <= ev.max_abs_stress <= 25_100
