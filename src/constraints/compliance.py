"""
Benchmark-level IS 800:2007 compliance check.

Given a `BenchmarkProblem` and a design vector, runs FEM (re-doing it
once per load case, like `BenchmarkProblem.evaluate` but keeping signed
axial forces so we can distinguish tension vs compression), then walks
every member through the Clause 3.8 / 6.2 / 6.3 / 7.1 gates defined in
`is800_checks.py`.

Unit discipline:
  * IS 800 formulas are SI.
  * For `benchmark.units == 'imperial'` we convert lengths (in -> m),
    forces (lbf -> N), and moduli (psi -> Pa) on the fly so downstream
    formulas stay clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.benchmarks.base import BenchmarkProblem
from src.fem.truss import Truss

from . import is800_checks as c


# ---------------------------------------------------------------------------
# Unit conversion helpers (imperial -> SI).
# ---------------------------------------------------------------------------

_IN_TO_M = 0.0254
_LBF_TO_N = 4.4482216152605
_PSI_TO_PA = 6894.757293168361


def _to_si_length(x_in: float, units: str) -> float:
    return x_in * _IN_TO_M if units == "imperial" else x_in


def _to_si_force(f: float, units: str) -> float:
    return f * _LBF_TO_N if units == "imperial" else f


def _to_si_modulus(E: float, units: str) -> float:
    return E * _PSI_TO_PA if units == "imperial" else E


def _to_si_area(a: float, units: str) -> float:
    return a * _IN_TO_M**2 if units == "imperial" else a


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class MemberCheck:
    """Per-member IS 800 compliance record."""

    index: int
    length_m: float
    area_m2: float
    worst_tension_N: float        # >= 0, worst tensile axial across load cases
    worst_compression_N: float    # >= 0, worst compressive axial across load cases
    slenderness_ok: bool
    tension_yield_ok: bool
    tension_rupture_ok: bool
    compression_ok: bool
    details: dict = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Whole-structure IS 800 report."""

    overall_ok: bool
    members: list[MemberCheck]
    deflection_ok: bool
    deflection_detail: dict


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def full_is800_check(
    benchmark: BenchmarkProblem,
    x: np.ndarray,
    deflection_span_m: float | None = None,
    deflection_divisor: float = 325.0,
) -> ComplianceReport:
    """Run IS 800 clauses on every member for every load case.

    Parameters
    ----------
    benchmark : BenchmarkProblem
    x         : design-variable vector (one area per symmetry group)
    deflection_span_m :
        Characteristic span for Clause 5.6.1. If `None`, we use the
        longest bar length as a conservative proxy.
    """
    units = benchmark.units
    areas = benchmark.expand_design(x)

    # Accumulate signed worst axials per member across load cases.
    worst_tension = np.zeros(benchmark.n_bars)       # >=0
    worst_compression = np.zeros(benchmark.n_bars)   # >=0 (magnitude)
    worst_disp_native = 0.0

    for lc in benchmark.load_cases:
        truss = Truss(
            nodes=benchmark.nodes,
            bar_connectivity=benchmark.connectivity,
            E=benchmark.E,
            areas=areas,
        )
        for node, dirs in benchmark.supports:
            truss.fix_node(node, dirs)
        for node, force in lc.nodal_forces.items():
            truss.apply_load(node=node, force=force)
        result = truss.solve()

        axials = result.axial_forces.astype(float)
        worst_tension = np.maximum(worst_tension, np.maximum(axials, 0.0))
        worst_compression = np.maximum(
            worst_compression, np.maximum(-axials, 0.0)
        )
        worst_disp_native = max(
            worst_disp_native,
            float(np.max(np.abs(result.displacements))),
        )

    bar_lengths_native = benchmark._bar_lengths
    E_si = _to_si_modulus(benchmark.E, units)

    members: list[MemberCheck] = []
    overall_ok = True

    for i in range(benchmark.n_bars):
        L_si = _to_si_length(float(bar_lengths_native[i]), units)
        A_si = _to_si_area(float(areas[i]), units)
        tN = _to_si_force(float(worst_tension[i]), units)
        cN = _to_si_force(float(worst_compression[i]), units)

        slend_t = c.check_slenderness(L_si, A_si, member_type="tension")
        slend_c = c.check_slenderness(L_si, A_si, member_type="compression")
        ty = c.check_tension_yield(A_si, tN)
        tr = c.check_tension_rupture(A_si, tN)
        cc = c.check_compression(L_si, A_si, cN, E=E_si)

        # Governing slenderness: if the bar sees any compression,
        # enforce the compression limit (stricter sign but looser value).
        slenderness_ok = slend_c["ok"] if cN > 0.0 else slend_t["ok"]

        member_ok = (
            slenderness_ok
            and ty["ok"]
            and tr["ok"]
            and cc["ok"]
        )
        overall_ok = overall_ok and member_ok

        members.append(
            MemberCheck(
                index=i,
                length_m=L_si,
                area_m2=A_si,
                worst_tension_N=tN,
                worst_compression_N=cN,
                slenderness_ok=slenderness_ok,
                tension_yield_ok=ty["ok"],
                tension_rupture_ok=tr["ok"],
                compression_ok=cc["ok"],
                details={
                    "slenderness": slend_c if cN > 0 else slend_t,
                    "tension_yield": ty,
                    "tension_rupture": tr,
                    "compression": cc,
                },
            )
        )

    # Clause 5.6.1 deflection check (serviceability).
    span = (
        deflection_span_m
        if deflection_span_m is not None
        else _to_si_length(float(np.max(bar_lengths_native)), units)
    )
    disp_si = _to_si_length(worst_disp_native, units)
    defl = c.check_deflection(
        disp_si, span_m=span, divisor=deflection_divisor
    )
    overall_ok = overall_ok and defl["ok"]

    return ComplianceReport(
        overall_ok=overall_ok,
        members=members,
        deflection_ok=defl["ok"],
        deflection_detail=defl,
    )
