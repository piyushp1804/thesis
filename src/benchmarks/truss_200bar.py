"""
200-bar planar truss — Kaveh & Talatahari (2010). "The steel one."

STATUS: placeholder. Critical trap: this problem uses REAL STEEL (not
aluminium), SI units (not imperial). See docs/ground_truth.md:
    nodes               : 77
    elements            : 200
    E, rho              : 210 GPa, 7850 kg/m^3  (STEEL, SI)
    load cases          : 3
    area bounds         : [6.5e-5, 2.5e-2] m^2
    symmetry groups     : 29
    stress limits       : +/- 250 MPa
    ref optimum weight  : 25445 kg
"""

from __future__ import annotations

from .base import BenchmarkProblem


def make_truss_200bar() -> BenchmarkProblem:  # pragma: no cover
    raise NotImplementedError(
        "200-bar benchmark is scaffolded but not yet encoded. "
        "REMINDER: this problem is steel/SI, not aluminium/imperial. "
        "See docs/ground_truth.md for geometry spec."
    )
