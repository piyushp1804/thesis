"""
25-bar spatial transmission tower — Schmit & Miura (1976).

STATUS: placeholder. Geometry, load-case table, and symmetry grouping
are documented but not yet encoded. Phase 1 ships the *framework* on
the 10-bar gate; the 25-bar will be encoded and re-validated in Phase
1.5 before any downstream phase uses it.

Spec (for whoever encodes it next — see docs/ground_truth.md):
    nodes               : 10   (N-body spatial tower)
    elements            : 25
    E, rho              : 1.0e7 psi, 0.1 lb/in^3 (aluminium, imperial)
    load cases          : 2 (transverse + vertical)
    area bounds         : [0.01, 3.4] in^2
    symmetry groups     : 8 (design vars = 8)
    stress limits       : member-type dependent (see paper)
    disp limit          : 0.35 in at top nodes
    ref optimum weight  : 545.22 lb
    ref source          : Schmit & Miura (1976)
"""

from __future__ import annotations

from .base import BenchmarkProblem


def make_truss_25bar() -> BenchmarkProblem:  # pragma: no cover
    raise NotImplementedError(
        "25-bar benchmark is scaffolded but not yet encoded. "
        "Phase 1 ships 10-bar only; run `make_truss_10bar()` instead. "
        "See docs/ground_truth.md for the exact geometry spec."
    )
