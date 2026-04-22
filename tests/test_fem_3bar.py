r"""
THE CANONICAL 3-BAR VALIDATION TEST — Day 1's "green light".

We solve a tiny, symmetric, pin-jointed truss whose answer we can compute
with pen-and-paper. If the FEM pipeline gives the same numbers to at least
3 decimals, Day 1 is done.

----------------------------------------------------------------------------
GEOMETRY  (Schmit-style 3-bar truss, the cleanest possible)
----------------------------------------------------------------------------

                     Node 3 (1, 1)  <-- free, load P = -1000 N in y
                       / |  \
                  bar /  |   \  bar
                  0-3 /  |    \ 2-3
                     /   | bar \
                    /    | 1-3  \
                   /     |       \
            Node 0 ----Node 1---- Node 2
             (0,0)     (1,0)     (2,0)
              PIN       PIN       PIN

Material (all 3 bars identical):
    E = 200 GPa      (typical structural steel)
    A = 1e-4 m^2     (100 mm^2)

----------------------------------------------------------------------------
HAND CALCULATION (brief — full derivation in docs/ later)
----------------------------------------------------------------------------

Let   alpha = E * A = 2e7 N   (convenient shorthand).

Stiffness at node 3 in the y-direction (K_yy):
  * bar 1-3 (vertical, L=1):            (alpha / 1)   * sin^2(90) = alpha
  * bars 0-3 & 2-3 (diag, L=sqrt(2)):   2 * (alpha/sqrt(2)) * sin^2(45)
                                       = 2 * (alpha/sqrt(2)) * 0.5
                                       = alpha / sqrt(2)

  K_yy = alpha + alpha/sqrt(2) = alpha * (1 + 1/sqrt(2))

By symmetry, horizontal coupling at node 3 is zero, so:
    u_3x = 0
    u_3y = F_y / K_yy
         = -1000 / [alpha * (1 + 1/sqrt(2))]
         = -1000 / [2e7 * (1 + 0.7071067811...)]
         = -2.928932188e-5 m      (about -29.3 micrometers)

Axial forces (tension positive):
    vertical bar 1-3:   N = alpha * u_3y           = -585.7864376 N
    diagonals 0-3, 2-3: N = (alpha/sqrt(2)) * (u_3y / sqrt(2))
                         = (alpha/2) * u_3y        = -292.8932188 N

(All three bars are in COMPRESSION — the load pushes node 3 down, all three
bars get squashed. Signs match our "+ tension / - compression" convention.)

Global equilibrium sanity:
    sum of bar-y forces on node 3 = -N_vert - 2 * N_diag * sin(45)
                                  = 585.79 + 2 * 292.89 * 0.7071
                                  = 585.79 + 414.21 = 1000 N  (upward)
    applied load on node 3        = -1000 N (downward)
    --> exactly balances. ✓
"""

from __future__ import annotations

import numpy as np
import pytest

from src.fem.truss import Truss


# ---------------------------------------------------------------------------
# Shared fixture: build the 3-bar truss exactly as sketched above.
# ---------------------------------------------------------------------------

@pytest.fixture
def three_bar_truss() -> Truss:
    """Build and load the canonical 3-bar truss. No solve yet."""
    nodes = np.array(
        [
            [0.0, 0.0],   # 0: bottom-left pin
            [1.0, 0.0],   # 1: bottom-center pin
            [2.0, 0.0],   # 2: bottom-right pin
            [1.0, 1.0],   # 3: top apex (free)
        ]
    )
    bars = [(0, 3), (1, 3), (2, 3)]   # three bars meeting at node 3

    truss = Truss(
        nodes=nodes,
        bar_connectivity=bars,
        E=200e9,        # Pa
        areas=1e-4,     # m^2  (same for every bar)
    )

    # Supports: bottom three nodes are fully pinned (all DOFs fixed).
    truss.fix_node(0)
    truss.fix_node(1)
    truss.fix_node(2)

    # Load: 1000 N downward at the apex node.
    truss.apply_load(node=3, force=(0.0, -1000.0))
    return truss


# ---------------------------------------------------------------------------
# The actual checks
# ---------------------------------------------------------------------------

# Closed-form expected answers (from the docstring above).
EXPECTED_U3Y = -1000.0 / (2e7 * (1.0 + 1.0 / np.sqrt(2.0)))
EXPECTED_N_VERT = 2e7 * EXPECTED_U3Y              # bar 1-3 axial force
EXPECTED_N_DIAG = (2e7 / 2.0) * EXPECTED_U3Y      # bars 0-3 and 2-3


def test_3bar_displacement_matches_hand_calc(three_bar_truss: Truss) -> None:
    """Node-3 y-displacement must match the closed-form value to 1e-9 m."""
    result = three_bar_truss.solve()
    u = result.displacements
    # Node 3 in 2D owns DOFs 6 (x) and 7 (y).
    u_3x, u_3y = u[6], u[7]

    # Symmetry forces u_3x to be zero exactly (up to floating-point noise).
    assert abs(u_3x) < 1e-12, f"u_3x should be 0, got {u_3x}"
    # Match the hand-calculated value to well below 3 decimals of micrometers.
    assert np.isclose(u_3y, EXPECTED_U3Y, rtol=0.0, atol=1e-9), (
        f"u_3y = {u_3y:.6e}, expected {EXPECTED_U3Y:.6e}"
    )


def test_3bar_support_nodes_do_not_move(three_bar_truss: Truss) -> None:
    """Pinned nodes should show zero displacement in every DOF."""
    result = three_bar_truss.solve()
    u = result.displacements
    # DOFs 0..5 belong to the pinned bottom nodes.
    assert np.allclose(u[:6], 0.0, atol=1e-14)


def test_3bar_member_forces(three_bar_truss: Truss) -> None:
    """
    Axial forces in all three bars match the hand calculation.

    Bar 0-3 (diag left), bar 1-3 (vertical), bar 2-3 (diag right).
    All three come out NEGATIVE = compression.
    """
    result = three_bar_truss.solve()
    n = result.axial_forces  # shape (3,)

    # Diagonals identical by symmetry.
    assert np.isclose(n[0], EXPECTED_N_DIAG, rtol=1e-6), (
        f"bar 0-3 force = {n[0]}, expected {EXPECTED_N_DIAG}"
    )
    assert np.isclose(n[2], EXPECTED_N_DIAG, rtol=1e-6), (
        f"bar 2-3 force = {n[2]}, expected {EXPECTED_N_DIAG}"
    )
    # Vertical bar:
    assert np.isclose(n[1], EXPECTED_N_VERT, rtol=1e-6), (
        f"bar 1-3 force = {n[1]}, expected {EXPECTED_N_VERT}"
    )
    # All three in compression (negative by our sign convention).
    assert (n < 0).all(), f"all bars should be in compression, got {n}"


def test_3bar_global_equilibrium(three_bar_truss: Truss) -> None:
    """Sum of reactions + sum of applied loads must equal zero."""
    result = three_bar_truss.solve()
    R = result.reactions

    # Applied loads vector reconstruction: 1000 N downward at node 3.
    F = np.zeros(8)
    F[7] = -1000.0

    # Global equilibrium in 2D: sum of x-forces = 0; sum of y-forces = 0.
    total = R + F
    # x-components (DOFs 0, 2, 4, 6):
    assert abs(total[[0, 2, 4, 6]].sum()) < 1e-8
    # y-components (DOFs 1, 3, 5, 7):
    assert abs(total[[1, 3, 5, 7]].sum()) < 1e-8


def test_3bar_reaction_symmetry(three_bar_truss: Truss) -> None:
    """Left and right pin reactions must mirror each other (x flips sign)."""
    result = three_bar_truss.solve()
    R = result.reactions

    # Node 0: DOFs 0 (x), 1 (y).  Node 2: DOFs 4 (x), 5 (y).
    assert np.isclose(R[0], -R[4], atol=1e-8), (
        f"R_0x ({R[0]}) should equal -R_2x ({R[4]}) by symmetry"
    )
    assert np.isclose(R[1], R[5], atol=1e-8), (
        f"R_0y ({R[1]}) should equal R_2y ({R[5]}) by symmetry"
    )


def test_3bar_total_volume_and_weight(three_bar_truss: Truss) -> None:
    """
    Volume and weight helpers on the Truss class produce the right numbers.

    V = A * (L_03 + L_13 + L_23)
      = 1e-4 * (sqrt(2) + 1 + sqrt(2))
      = 1e-4 * (1 + 2*sqrt(2))
    Weight with steel density 7850 kg/m^3:
      W = 7850 * V
    """
    expected_volume = 1e-4 * (1.0 + 2.0 * np.sqrt(2.0))
    assert np.isclose(three_bar_truss.total_volume(), expected_volume)

    expected_weight = 7850.0 * expected_volume
    assert np.isclose(three_bar_truss.total_weight(density=7850.0), expected_weight)
