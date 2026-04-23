"""
IS 800:2007 design-check formulas (pure functions, SI units).

Each function returns a small dict with the calculated capacity and a
boolean `ok` flag. Inputs are SI: lengths in metres, areas in m^2,
forces in newtons, stresses in pascals. Callers that work in imperial
units (10-bar, 25-bar, 72-bar benchmarks) must convert before calling.

Cross-section assumption: circular solid bar. For a solid circle of
area A, the radius of gyration is:

    r_g = sqrt(I/A) = R/2     where  R = sqrt(A/pi)

Truss members are pin-pin so the effective-length factor K = 1.0.

Implemented clauses:
  * 3.8   slenderness limits (KL/r <= 180 tension, <= 250 compression)
  * 6.2   tension strength, gross-section yielding
  * 6.3   tension strength, net-section rupture (no bolt holes => An = Ag)
  * 7.1   compression strength via Perry-Robertson buckling curve
  * 7.5   notes on angles / compound members (absorbed into 3.8 here)

Default steel: Fe 410, fy = 250 MPa, fu = 410 MPa, gamma_m0 = 1.10,
gamma_m1 = 1.25, imperfection factor alpha = 0.34 (curve b, welded I /
general purpose — conservative middle ground).
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Default material constants (Fe 410, SI units)
# ---------------------------------------------------------------------------

FY_DEFAULT = 250.0e6      # Pa
FU_DEFAULT = 410.0e6      # Pa
GAMMA_M0 = 1.10           # partial safety factor for yielding
GAMMA_M1 = 1.25           # partial safety factor for ultimate
ALPHA_DEFAULT = 0.34      # buckling curve 'b' imperfection factor

# Clause 3.8 slenderness limits
LAMBDA_MAX_TENSION = 180.0
LAMBDA_MAX_COMPRESSION = 250.0


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------


def radius_of_gyration_circular(area_m2: float) -> float:
    """r_g for a solid-circular cross-section of given area."""
    if area_m2 <= 0.0:
        raise ValueError("area must be positive.")
    R = math.sqrt(area_m2 / math.pi)
    return 0.5 * R


# ---------------------------------------------------------------------------
# Clause 3.8 — slenderness
# ---------------------------------------------------------------------------


def check_slenderness(
    length_m: float,
    area_m2: float,
    member_type: str = "tension",
    K: float = 1.0,
) -> dict:
    """Return {lambda, limit, ok} for IS 800 Clause 3.8."""
    if member_type not in ("tension", "compression"):
        raise ValueError("member_type must be 'tension' or 'compression'.")
    r_g = radius_of_gyration_circular(area_m2)
    slenderness = K * length_m / r_g
    limit = (
        LAMBDA_MAX_TENSION
        if member_type == "tension"
        else LAMBDA_MAX_COMPRESSION
    )
    return {
        "lambda": slenderness,
        "limit": limit,
        "ok": slenderness <= limit,
    }


# ---------------------------------------------------------------------------
# Clause 6.2 — tension yield
# ---------------------------------------------------------------------------


def check_tension_yield(
    area_m2: float,
    axial_force_N: float,
    fy: float = FY_DEFAULT,
    gamma_m0: float = GAMMA_M0,
) -> dict:
    """Gross-section yielding capacity Tdg = A fy / gamma_m0."""
    Tdg = area_m2 * fy / gamma_m0
    return {
        "Tdg": Tdg,
        "demand": axial_force_N,
        "ratio": axial_force_N / Tdg if Tdg > 0 else float("inf"),
        "ok": axial_force_N <= Tdg,
    }


# ---------------------------------------------------------------------------
# Clause 6.3 — tension rupture
# ---------------------------------------------------------------------------


def check_tension_rupture(
    area_m2: float,
    axial_force_N: float,
    fu: float = FU_DEFAULT,
    gamma_m1: float = GAMMA_M1,
    net_area_ratio: float = 1.0,
) -> dict:
    """Net-section rupture: Tdn = 0.9 An fu / gamma_m1.

    `net_area_ratio` = An / Ag; default 1.0 (no bolt holes, welded ends).
    """
    An = net_area_ratio * area_m2
    Tdn = 0.9 * An * fu / gamma_m1
    return {
        "Tdn": Tdn,
        "demand": axial_force_N,
        "ratio": axial_force_N / Tdn if Tdn > 0 else float("inf"),
        "ok": axial_force_N <= Tdn,
    }


# ---------------------------------------------------------------------------
# Clause 7.1 — compression / buckling (Perry-Robertson)
# ---------------------------------------------------------------------------


def check_compression(
    length_m: float,
    area_m2: float,
    axial_force_N: float,
    E: float,
    K: float = 1.0,
    fy: float = FY_DEFAULT,
    gamma_m0: float = GAMMA_M0,
    alpha: float = ALPHA_DEFAULT,
) -> dict:
    """Design compressive strength Pd = A fcd, with fcd per IS 800 7.1.2.

    `axial_force_N` is the positive magnitude of the compressive demand.
    """
    r_g = radius_of_gyration_circular(area_m2)
    slenderness = K * length_m / r_g                    # KL/r

    # Euler critical stress -> non-dimensional slenderness lambda_bar.
    fcc = math.pi**2 * E / slenderness**2               # Pa
    lambda_bar = math.sqrt(fy / fcc)

    phi = 0.5 * (1.0 + alpha * (lambda_bar - 0.2) + lambda_bar**2)
    chi = 1.0 / (phi + math.sqrt(phi**2 - lambda_bar**2))
    chi = min(chi, 1.0)

    fcd = chi * fy / gamma_m0                           # design stress
    Pd = area_m2 * fcd                                  # design force

    return {
        "Pd": Pd,
        "fcd": fcd,
        "lambda_bar": lambda_bar,
        "chi": chi,
        "demand": axial_force_N,
        "ratio": axial_force_N / Pd if Pd > 0 else float("inf"),
        "ok": axial_force_N <= Pd,
    }


# ---------------------------------------------------------------------------
# Clause 5.6.1 — deflection limit
# ---------------------------------------------------------------------------


def check_deflection(
    max_displacement_m: float,
    span_m: float,
    divisor: float = 325.0,
) -> dict:
    """Serviceability deflection: delta <= span / divisor. IS 800 5.6.1."""
    limit = span_m / divisor
    return {
        "limit": limit,
        "demand": max_displacement_m,
        "ok": max_displacement_m <= limit,
    }
