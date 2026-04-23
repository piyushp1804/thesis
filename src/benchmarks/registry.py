"""
Benchmark name -> factory registry.

Centralizing dispatch here means scripts like `run_single.py` and test
files only need the benchmark *name* string; they never import the
factory functions directly. Adding a new benchmark = one line here.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import BenchmarkProblem
from .truss_10bar import make_truss_10bar
from .truss_25bar import make_truss_25bar
from .truss_72bar import make_truss_72bar
from .truss_200bar import make_truss_200bar


# name -> zero-arg factory function
_FACTORIES: dict[str, Callable[[], BenchmarkProblem]] = {
    "10bar": make_truss_10bar,
    "25bar": make_truss_25bar,
    "72bar": make_truss_72bar,
    "200bar": make_truss_200bar,
}


def available_benchmarks() -> list[str]:
    """Return every registered benchmark name, in insertion order."""
    return list(_FACTORIES.keys())


def get_benchmark(name: str) -> BenchmarkProblem:
    """Construct a benchmark by name.

    Raises
    ------
    KeyError if `name` is not registered.
    NotImplementedError if the benchmark is scaffolded but not encoded yet.
    """
    if name not in _FACTORIES:
        raise KeyError(
            f"unknown benchmark '{name}'. "
            f"available: {available_benchmarks()}"
        )
    return _FACTORIES[name]()
