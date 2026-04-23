"""
NSGA-II produces a meaningful weight-vs-displacement Pareto front on
the 10-bar benchmark.

Pass criteria:
  * at least 20 non-dominated points
  * weight range spans at least 20% (light vs stiff designs)
  * displacement range spans at least 30% (proves trade-off exists)
  * minimum-weight point is within 3% of Sunar 1991 optimum
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.runner import run
from src.benchmarks.registry import get_benchmark


@pytest.mark.slow
def test_nsga2_10bar_pareto_front():
    bench = get_benchmark("10bar")
    r = run("nsga2", bench, seed=42, pop_size=100, n_gen=300)

    assert r.pareto_f is not None
    assert r.pareto_x is not None
    n = len(r.pareto_f)
    assert n >= 20, f"expected >=20 Pareto pts, got {n}"

    w = r.pareto_f[:, 0]
    d = r.pareto_f[:, 1]
    weight_span = (w.max() - w.min()) / w.min()
    disp_span = (d.max() - d.min()) / d.min()
    assert weight_span > 0.2, f"weight range too narrow: {weight_span:.3f}"
    assert disp_span > 0.3, f"disp range too narrow: {disp_span:.3f}"

    # Min-weight Pareto point should be close to Sunar optimum.
    assert abs(w.min() - 5060.85) / 5060.85 <= 0.03
