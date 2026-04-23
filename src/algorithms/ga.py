"""
Single-objective GA driver (pymoo).

Defaults chosen to match typical structural-optimization papers so our
numbers are comparable: pop_size=100, n_gen=500, SBX+PM operators,
Latin-Hypercube sampling for diverse starts.
"""

from __future__ import annotations

import time

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

from src.benchmarks.base import BenchmarkProblem

from .problem import Evaluator, TrussProblem
from .result_schema import OptimizationResult


def run_ga(
    benchmark: BenchmarkProblem,
    seed: int = 42,
    pop_size: int = 100,
    n_gen: int = 500,
    evaluator: Evaluator | None = None,
    verbose: bool = False,
    x0: np.ndarray | None = None,
) -> OptimizationResult:
    """Run GA on `benchmark`, return standardized result."""
    problem = TrussProblem(benchmark, evaluator=evaluator, mode="single")

    sampling = LHS() if x0 is None else _seeded_sampling(x0, pop_size, benchmark)

    algo = GA(
        pop_size=pop_size,
        sampling=sampling,
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
        save_history=True,
        verbose=verbose,
    )
    wall = time.perf_counter() - t0

    return _package_single(
        benchmark=benchmark,
        res=res,
        seed=seed,
        algorithm="ga",
        wall_time=wall,
        evaluator=problem._evaluator,
    )


def _seeded_sampling(x0: np.ndarray, pop_size: int, benchmark: BenchmarkProblem):
    """Custom sampler: first row = x0 (LLM warm-start), rest = LHS."""
    from pymoo.core.sampling import Sampling

    class _Seeded(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            pop = LHS()._do(problem, n_samples)
            pop[0] = np.clip(
                np.asarray(x0, dtype=float),
                problem.xl,
                problem.xu,
            )
            return pop

    return _Seeded()


def _package_single(
    *,
    benchmark: BenchmarkProblem,
    res,
    seed: int,
    algorithm: str,
    wall_time: float,
    evaluator: Evaluator,
) -> OptimizationResult:
    """Turn a pymoo Result into our OptimizationResult."""
    best_x = np.asarray(res.X, dtype=float).ravel()
    ev = evaluator(best_x)

    g_max = float(np.max(res.G)) if res.G is not None else 0.0
    feasible = g_max <= 1e-6

    history = []
    if res.history is not None:
        running_best = float("inf")
        for gen_idx, h in enumerate(res.history):
            pop = h.pop
            F = pop.get("F").ravel()
            G = pop.get("G")
            # feasible = all constraints <= 0 (within tiny tolerance)
            feas_mask = (G.max(axis=1) <= 1e-6) if G is not None else np.ones_like(F, dtype=bool)
            if feas_mask.any():
                gen_best_feasible = float(F[feas_mask].min())
                running_best = min(running_best, gen_best_feasible)
            history.append(
                {
                    "gen": gen_idx,
                    "best_weight": running_best if np.isfinite(running_best) else float("nan"),
                    "mean_weight": float(F.mean()),
                }
            )

    return OptimizationResult(
        algorithm=algorithm,
        benchmark=benchmark.name,
        seed=seed,
        best_x=best_x,
        best_weight=float(ev.weight),
        feasible=feasible,
        max_stress=float(ev.max_abs_stress),
        max_displacement=float(ev.max_abs_displacement),
        history=history,
        wall_time_s=float(wall_time),
        n_evals=int(res.algorithm.evaluator.n_eval),
    )
