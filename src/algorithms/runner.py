"""
Thin dispatcher: name -> optimizer function. Used by CLI scripts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.benchmarks.base import BenchmarkProblem

from .ga import run_ga
from .nsga2 import run_nsga2
from .pso import run_pso
from .result_schema import OptimizationResult

_RUNNERS: dict[str, Callable[..., OptimizationResult]] = {
    "ga": run_ga,
    "pso": run_pso,
    "nsga2": run_nsga2,
}


def available_algorithms() -> list[str]:
    return list(_RUNNERS.keys())


def run(
    algorithm: str,
    benchmark: BenchmarkProblem,
    seed: int = 42,
    **kwargs: Any,
) -> OptimizationResult:
    """Single entry-point used by scripts: pick algorithm by name."""
    if algorithm not in _RUNNERS:
        raise KeyError(
            f"unknown algorithm '{algorithm}'. "
            f"available: {available_algorithms()}"
        )
    return _RUNNERS[algorithm](benchmark, seed=seed, **kwargs)
