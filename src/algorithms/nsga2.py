"""
NSGA-II driver (pymoo) — multi-objective weight vs max-displacement.

Produces a Pareto front for Chapter 4 trade-off plots. Also extracts
the minimum-weight Pareto point as the single-objective comparison
against GA / PSO results.
"""

from __future__ import annotations

import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

from src.benchmarks.base import BenchmarkProblem

from .problem import Evaluator, TrussProblem
from .result_schema import OptimizationResult


def run_nsga2(
    benchmark: BenchmarkProblem,
    seed: int = 42,
    pop_size: int = 100,
    n_gen: int = 500,
    evaluator: Evaluator | None = None,
    verbose: bool = False,
) -> OptimizationResult:
    """Run NSGA-II, return result with `pareto_x` and `pareto_f` populated."""
    problem = TrussProblem(benchmark, evaluator=evaluator, mode="multi")

    algo = NSGA2(
        pop_size=pop_size,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    t0 = time.perf_counter()
    res = minimize(
        problem,
        algo,
        ("n_gen", n_gen),
        seed=seed,
        save_history=False,
        verbose=verbose,
    )
    wall = time.perf_counter() - t0

    pareto_x = np.asarray(res.X, dtype=float) if res.X is not None else None
    pareto_f = np.asarray(res.F, dtype=float) if res.F is not None else None

    # Pick the minimum-weight Pareto point for the scalar summary.
    if pareto_f is not None and len(pareto_f) > 0:
        k = int(np.argmin(pareto_f[:, 0]))
        best_x = pareto_x[k]
    else:
        best_x = np.full(benchmark.n_design_vars, np.nan)

    evaluator_fn = problem._evaluator
    ev = evaluator_fn(best_x)

    g_max = float(np.max(res.G)) if res.G is not None else 0.0
    feasible = g_max <= 1e-6

    return OptimizationResult(
        algorithm="nsga2",
        benchmark=benchmark.name,
        seed=seed,
        best_x=best_x,
        best_weight=float(ev.weight),
        feasible=feasible,
        max_stress=float(ev.max_abs_stress),
        max_displacement=float(ev.max_abs_displacement),
        pareto_x=pareto_x,
        pareto_f=pareto_f,
        wall_time_s=float(wall),
        n_evals=int(res.algorithm.evaluator.n_eval),
    )
