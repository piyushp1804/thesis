"""
Adapter between our `BenchmarkProblem` and `pymoo`'s problem interface.

Two modes:
  * 'single' — scalar objective (weight). Used by GA and PSO.
  * 'multi'  — two objectives (weight, max_displacement). Used by NSGA-II.

Constraint convention: pymoo treats `G <= 0` as feasible. We express
violation as fractional excess over the benchmark limits so the GA can
meaningfully penalize near-feasible designs.

The `evaluator` parameter lets Phase 2 swap in a neural surrogate with
the same signature as `benchmark.evaluate`. Phase 1 uses the default
FEM-backed `benchmark.evaluate`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from src.benchmarks.base import BenchmarkEvaluation, BenchmarkProblem


Evaluator = Callable[[np.ndarray], BenchmarkEvaluation]


class TrussProblem(ElementwiseProblem):
    """pymoo wrapper around a `BenchmarkProblem`."""

    def __init__(
        self,
        benchmark: BenchmarkProblem,
        evaluator: Evaluator | None = None,
        mode: str = "single",
    ) -> None:
        if mode not in ("single", "multi"):
            raise ValueError("mode must be 'single' or 'multi'.")
        self.benchmark = benchmark
        self.mode = mode
        self._evaluator: Evaluator = evaluator or benchmark.evaluate

        lo, hi = benchmark.area_bounds
        n_var = benchmark.n_design_vars
        n_obj = 1 if mode == "single" else 2
        n_constr = 2

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_constr,
            xl=np.full(n_var, lo, dtype=float),
            xu=np.full(n_var, hi, dtype=float),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        ev = self._evaluator(np.asarray(x, dtype=float))

        bench = self.benchmark
        stress_limit = max(
            bench.stress_limit_tension,
            bench.stress_limit_compression,
        )
        g_stress = ev.max_abs_stress / stress_limit - 1.0
        g_disp = ev.max_abs_displacement / bench.displacement_limit - 1.0

        if self.mode == "single":
            out["F"] = [ev.weight]
        else:
            out["F"] = [ev.weight, ev.max_abs_displacement]
        out["G"] = [g_stress, g_disp]
