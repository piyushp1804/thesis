"""
72-bar spatial tower — Erbatur et al. (2000).

STATUS: placeholder. See docs/ground_truth.md for the spec:
    nodes               : 20
    elements            : 72
    E, rho              : 1.0e7 psi, 0.1 lb/in^3
    load cases          : 2
    area bounds         : [0.1, 4.0] in^2
    symmetry groups     : 16
    stress limits       : +/- 25000 psi
    disp limit          : 0.25 in at top node
    ref optimum weight  : 379.62 lb
"""

from __future__ import annotations

from .base import BenchmarkProblem


def make_truss_72bar() -> BenchmarkProblem:  # pragma: no cover
    raise NotImplementedError(
        "72-bar benchmark is scaffolded but not yet encoded. "
        "See docs/ground_truth.md for geometry spec."
    )
