"""
Unit tests for the element-level math.

Why tests?
  - The 3-bar system test (test_fem_3bar.py) tells us "does the whole
    pipeline give the right answer?". If it fails, we don't know WHERE.
  - These unit tests pin down the SMALL pieces so any regression points us
    at the exact file that broke.

Each test checks ONE property:
  - length, direction cosines, dof_indices mapping
  - stiffness matrix: symmetric, correct norm, rank-1 structure (physics)
  - the 3D path (so Day 2 trusses have coverage)
  - input validation (zero-length bar, bad dimensions)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.fem.truss_element import TrussElement


# ---------------- Geometry ---------------------------------------------------

def test_length_horizontal_bar() -> None:
    el = TrussElement(0, 1, [0.0, 0.0], [3.0, 0.0], E=200e9, A=1e-4)
    assert np.isclose(el.length, 3.0)


def test_length_diagonal_2d() -> None:
    # 3-4-5 triangle: length must come out 5 exactly.
    el = TrussElement(0, 1, [0.0, 0.0], [3.0, 4.0], E=200e9, A=1e-4)
    assert np.isclose(el.length, 5.0)


def test_direction_cosines_2d() -> None:
    # 45-degree bar. cos = sin = 1/sqrt(2).
    el = TrussElement(0, 1, [0.0, 0.0], [1.0, 1.0], E=200e9, A=1e-4)
    dc = el.direction_cosines()
    assert np.allclose(dc, [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_direction_cosines_3d() -> None:
    # Unit diagonal in 3D: all three cosines must be 1/sqrt(3).
    el = TrussElement(
        0, 1, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], E=200e9, A=1e-4
    )
    dc = el.direction_cosines()
    assert np.allclose(dc, [1 / np.sqrt(3)] * 3)


# ---------------- DOF mapping -----------------------------------------------

def test_dof_indices_2d() -> None:
    # Node 0 owns DOFs [0, 1]; node 2 owns DOFs [4, 5] in 2D.
    el = TrussElement(0, 2, [0.0, 0.0], [1.0, 0.0], E=200e9, A=1e-4)
    assert list(el.dof_indices()) == [0, 1, 4, 5]


def test_dof_indices_3d() -> None:
    # In 3D every node has 3 DOFs. Node 1 -> [3,4,5]; node 3 -> [9,10,11].
    el = TrussElement(
        1, 3, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], E=200e9, A=1e-4
    )
    assert list(el.dof_indices()) == [3, 4, 5, 9, 10, 11]


# ---------------- Stiffness matrix properties -------------------------------

def test_stiffness_is_symmetric_2d() -> None:
    # Physics: K = K^T for every elastic element.
    el = TrussElement(0, 1, [0.0, 0.0], [1.0, 2.0], E=200e9, A=1e-4)
    k = el.global_stiffness()
    assert np.allclose(k, k.T)


def test_stiffness_is_symmetric_3d() -> None:
    el = TrussElement(
        0, 1, [0.0, 0.0, 0.0], [2.0, -1.0, 3.0], E=200e9, A=2.5e-4
    )
    k = el.global_stiffness()
    assert np.allclose(k, k.T)


def test_stiffness_rank_is_one() -> None:
    # A single truss bar resists ONLY axial motion. So its 4x4 (or 6x6)
    # stiffness has exactly ONE non-zero eigenvalue — the axial mode.
    # Matrix rank must be 1.
    el = TrussElement(0, 1, [0.0, 0.0], [1.0, 1.0], E=200e9, A=1e-4)
    k = el.global_stiffness()
    # Using a tight tolerance relative to the axial stiffness.
    assert np.linalg.matrix_rank(k, tol=1e-6) == 1


def test_stiffness_horizontal_exact() -> None:
    # Expected for horizontal bar with k_axial = EA/L = 2e7:
    #  [[ k, 0, -k, 0],
    #   [ 0, 0,  0, 0],
    #   [-k, 0,  k, 0],
    #   [ 0, 0,  0, 0]]
    el = TrussElement(0, 1, [0.0, 0.0], [1.0, 0.0], E=200e9, A=1e-4)
    k = 2e7
    expected = np.array(
        [
            [k, 0, -k, 0],
            [0, 0, 0, 0],
            [-k, 0, k, 0],
            [0, 0, 0, 0],
        ]
    )
    assert np.allclose(el.global_stiffness(), expected)


def test_stiffness_45_degrees_exact() -> None:
    # 45-degree bar; c = s = 1/sqrt(2), c^2 = s^2 = cs = 1/2.
    # Each non-zero entry becomes (EA/L) * 0.5.
    L = np.sqrt(2.0)
    EA_L = 200e9 * 1e-4 / L
    el = TrussElement(0, 1, [0.0, 0.0], [1.0, 1.0], E=200e9, A=1e-4)
    k = el.global_stiffness()
    # Just check one diagonal and one coupling term.
    assert np.isclose(k[0, 0], 0.5 * EA_L)
    assert np.isclose(k[0, 1], 0.5 * EA_L)
    assert np.isclose(k[0, 2], -0.5 * EA_L)


# ---------------- Input validation -----------------------------------------

def test_rejects_zero_length_bar() -> None:
    # Two coincident nodes: impossible to normalize the direction vector.
    with pytest.raises(ValueError):
        TrussElement(0, 1, [0.0, 0.0], [0.0, 0.0], E=200e9, A=1e-4)


def test_rejects_mismatched_dims() -> None:
    with pytest.raises(ValueError):
        TrussElement(0, 1, [0.0, 0.0], [1.0, 0.0, 0.0], E=200e9, A=1e-4)


def test_rejects_same_nodes() -> None:
    with pytest.raises(ValueError):
        TrussElement(2, 2, [0.0, 0.0], [1.0, 0.0], E=200e9, A=1e-4)


def test_rejects_nonpositive_material_props() -> None:
    with pytest.raises(ValueError):
        TrussElement(0, 1, [0.0, 0.0], [1.0, 0.0], E=0.0, A=1e-4)
    with pytest.raises(ValueError):
        TrussElement(0, 1, [0.0, 0.0], [1.0, 0.0], E=200e9, A=-1e-4)
