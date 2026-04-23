"""
Particle Swarm Optimization driver (pymoo).

Defaults mirror Kennedy-Eberhart's common values (w=0.7, c1=c2=1.5) with
pymoo's constriction-factor variant enabled for stability.
"""

from __future__ import annotations

import time

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

from src.benchmarks.base import BenchmarkProblem

from .ga import _package_single
from .problem import Evaluator, TrussProblem
from .result_schema import OptimizationResult


def run_pso(
    benchmark: BenchmarkProblem,
    seed: int = 42,
    pop_size: int = 50,
    n_gen: int = 500,
    evaluator: Evaluator | None = None,
    verbose: bool = False,
) -> OptimizationResult:
    """Run PSO on `benchmark`, return standardized result."""
    problem = TrussProblem(benchmark, evaluator=evaluator, mode="single")

    algo = PSO(
        pop_size=pop_size,
        sampling=LHS(),
        w=0.7,
        c1=1.5,
        c2=1.5,
    )

    t0 = time.perf_counter()
    res = minimize(
        problem,
        algo,
        ("n_gen", n_gen),
        seed=seed,
        save_history=True,
        verbose=verbose,
    )
    wall = time.perf_counter() - t0

    return _package_single(
        benchmark=benchmark,
        res=res,
        seed=seed,
        algorithm="pso",
        wall_time=wall,
        evaluator=problem._evaluator,
    )
