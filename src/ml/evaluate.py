"""
Hold-out evaluation + FEM-vs-surrogate speedup benchmark.

Use from scripts or notebooks:

    from src.ml.evaluate import r2_report, speedup_vs_fem
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.benchmarks.base import BenchmarkProblem

from .dataset import SurrogateDataset
from .surrogate import SurrogateEvaluator


@dataclass
class R2Report:
    weight: float
    max_stress: float
    max_disp: float


def r2_report(
    benchmark: BenchmarkProblem,
    evaluator: SurrogateEvaluator,
    dataset: SurrogateDataset,
) -> R2Report:
    """Score the surrogate on an arbitrary LHS dataset (typically held-out)."""
    pred_w = np.zeros(len(dataset.weight))
    pred_s = np.zeros(len(dataset.weight))
    pred_d = np.zeros(len(dataset.weight))
    for i, x in enumerate(dataset.X):
        ev = evaluator(x)
        pred_w[i] = ev.weight
        pred_s[i] = ev.max_abs_stress
        pred_d[i] = ev.max_abs_displacement

    def r2(p, t):
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return R2Report(
        weight=r2(pred_w, dataset.weight),
        max_stress=r2(pred_s, dataset.max_stress),
        max_disp=r2(pred_d, dataset.max_disp),
    )


@dataclass
class SpeedupReport:
    fem_total_s: float
    surrogate_total_s: float
    speedup: float
    n_calls: int


def speedup_vs_fem(
    benchmark: BenchmarkProblem,
    evaluator: SurrogateEvaluator,
    n_calls: int = 1000,
    seed: int = 0,
) -> SpeedupReport:
    """Time `n_calls` evaluations via FEM vs surrogate on identical inputs."""
    rng = np.random.default_rng(seed)
    lo, hi = benchmark.area_bounds
    X = rng.uniform(lo, hi, size=(n_calls, benchmark.n_design_vars))

    t0 = time.perf_counter()
    for x in X:
        benchmark.evaluate(x)
    t_fem = time.perf_counter() - t0

    t0 = time.perf_counter()
    for x in X:
        evaluator(x)
    t_sur = time.perf_counter() - t0

    return SpeedupReport(
        fem_total_s=t_fem,
        surrogate_total_s=t_sur,
        speedup=t_fem / t_sur if t_sur > 0 else float("inf"),
        n_calls=n_calls,
    )
