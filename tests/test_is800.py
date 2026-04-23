"""
Hand-verified IS 800:2007 clause checks.

Every formula is exercised with numbers small enough to verify on a
calculator. If any of these drift the compliance module is broken.
"""

from __future__ import annotations

import math

import pytest

from src.constraints import is800_checks as c


def test_radius_of_gyration_circular():
    # A = pi * R^2, r_g = R/2
    A = math.pi * 0.01**2        # R = 10 mm
    assert c.radius_of_gyration_circular(A) == pytest.approx(0.005, rel=1e-12)


def test_slenderness_tension_pass_fail():
    A = math.pi * 0.05**2        # R = 50 mm, r_g = 25 mm
    ok = c.check_slenderness(4.0, A, "tension")      # KL/r = 160 < 180
    fail = c.check_slenderness(5.0, A, "tension")    # KL/r = 200 > 180
    assert ok["lambda"] == pytest.approx(160.0, rel=1e-12)
    assert ok["ok"] is True
    assert fail["ok"] is False


def test_slenderness_compression_pass_fail():
    A = math.pi * 0.05**2
    ok = c.check_slenderness(6.0, A, "compression")   # 240 < 250
    fail = c.check_slenderness(7.0, A, "compression") # 280 > 250
    assert ok["ok"] is True
    assert fail["ok"] is False


def test_tension_yield_matches_hand_calc():
    A = math.pi * 0.05**2        # 7.854e-3 m^2
    # Tdg = A * 250e6 / 1.10
    out = c.check_tension_yield(A, axial_force_N=1.0e6)
    expected_Tdg = A * 250e6 / 1.10
    assert out["Tdg"] == pytest.approx(expected_Tdg, rel=1e-10)
    assert out["ok"] is True


def test_tension_rupture_formula():
    A = math.pi * 0.05**2
    # Tdn = 0.9 * A * 410e6 / 1.25
    out = c.check_tension_rupture(A, axial_force_N=0.0)
    expected = 0.9 * A * 410e6 / 1.25
    assert out["Tdn"] == pytest.approx(expected, rel=1e-10)


def test_compression_perry_robertson_intermediate_slenderness():
    """Stocky column: chi should be close to 1.0."""
    A = math.pi * 0.05**2
    L = 0.5  # very short -> stocky -> chi ~ 1
    out = c.check_compression(L, A, axial_force_N=0.0, E=210e9)
    assert out["chi"] <= 1.0
    assert out["chi"] > 0.9

    # Slender column: chi should be much < 1.
    L_slender = 5.0
    out2 = c.check_compression(L_slender, A, axial_force_N=0.0, E=210e9)
    assert out2["chi"] < 0.5
    assert out2["Pd"] < out["Pd"]


def test_compression_demand_capacity_flag():
    A = math.pi * 0.05**2
    out = c.check_compression(3.0, A, axial_force_N=100e3, E=210e9)
    # Pd ~ 747 kN from manual check; 100 kN should pass.
    assert out["ok"] is True
    out2 = c.check_compression(3.0, A, axial_force_N=2e6, E=210e9)
    assert out2["ok"] is False


def test_deflection_clause_5_6_1():
    # 5 mm over 3 m span with divisor 325 => limit ~ 9.23 mm, pass.
    ok = c.check_deflection(0.005, 3.0)
    assert ok["ok"] is True
    fail = c.check_deflection(0.015, 3.0)  # > 9.23 mm
    assert fail["ok"] is False
