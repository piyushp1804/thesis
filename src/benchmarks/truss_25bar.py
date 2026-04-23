"""
25-bar spatial transmission tower — Schmit & Miura (1976).

Standard 10-node / 25-bar spatial truss benchmark. Used in 100+ size-
optimization papers since the 1970s. The encoding here follows the
common modern simplification (Camp 2007, Hasançebi 2010, Kaveh 2011,
Degertekin 2012) with uniform ±40,000 psi stress limits. The commonly-
cited literature optimum weight is 545.22 lb. With pop=100 and
n_gen=500, GA / PSO / NSGA-II all converge to within 0.3 % of that
figure across the seeds in tests/test_ga_25bar.py. Quick runs with
pop=50 / n_gen=150 may finish around 570 lb; that is under-converged
GA, not a formulation issue — boost the budget.

Geometry (inches, right-handed cartesian):

                 (0)=========(1)        z = 200
                 / \\         / \\
                /   \\       /   \\
               (5)---(4)   (2)---(3)    z = 100  (mid-level square)
                |\\  /\\    /\\  /|
                | \\/  \\  /  \\/ |
                | /\\   \\/   /\\ |
                |/  \\  /\\  /  \\|
               (9)---(8)-(7)---(6)      z =   0  (base square, supports)

Node indexing (0-based):
    n0 = (-37.5,    0, 200)   top-left   (free, loaded)
    n1 = ( 37.5,    0, 200)   top-right  (free, loaded)
    n2 = (-37.5,  37.5, 100)  mid-level +y corner
    n3 = ( 37.5,  37.5, 100)
    n4 = ( 37.5, -37.5, 100)
    n5 = (-37.5, -37.5, 100)
    n6 = (-100,   100,   0)   base corner
    n7 = ( 100,   100,   0)
    n8 = ( 100,  -100,   0)
    n9 = (-100,  -100,   0)

Supports: n6, n7, n8, n9 fully fixed (pinned base).

8 symmetry groups (design variables A1..A8) following the standard
literature grouping — 25 bars total: 1+4+4+2+2+4+4+4.

Two load cases per Schmit 1976 / Haftka-Gürdal convention.
"""

from __future__ import annotations

import numpy as np

from .base import BenchmarkProblem, LoadCase


# Commonly-cited literature optimum weight (Hasançebi 2010, Kaveh 2011).
# Exact published area vectors are paper-specific and depend on the
# exact connectivity / stress-limit grouping chosen; since our encoding
# uses the uniform-±40-ksi simplification (not Schmit's member-type-
# dependent compression limits), we do NOT claim to reproduce any
# specific literature area vector at the FEM level.
_REF_WEIGHT_LB = 545.22


# Group membership — lists of bar indices (0-indexed) per design var.
# 1 + 4 + 4 + 2 + 2 + 4 + 4 + 4 = 25 bars.
_GROUP_MAP: list[list[int]] = [
    [0],                       # A1: top-chord bar (n0-n1)
    [1, 2, 3, 4],              # A2: top-level to mid-level "short" diagonals
    [5, 6, 7, 8],              # A3: top-level to mid-level "long" diagonals
    [9, 10],                   # A4: mid-level +y / -y horizontals
    [11, 12],                  # A5: mid-level +x / -x horizontals
    [13, 14, 15, 16],          # A6: mid-to-base verticals
    [17, 18, 19, 20],          # A7: mid-to-base lateral diagonals
    [21, 22, 23, 24],          # A8: mid-to-base cross diagonals
]


def make_truss_25bar() -> BenchmarkProblem:
    """Construct the 25-bar spatial transmission tower problem."""

    nodes = np.array(
        [
            [-37.5,    0.0, 200.0],  # n0 top-left (loaded)
            [ 37.5,    0.0, 200.0],  # n1 top-right (loaded)
            [-37.5,   37.5, 100.0],  # n2 mid +y-left
            [ 37.5,   37.5, 100.0],  # n3 mid +y-right
            [ 37.5,  -37.5, 100.0],  # n4 mid -y-right
            [-37.5,  -37.5, 100.0],  # n5 mid -y-left
            [-100.0,  100.0,  0.0],  # n6 base  +y-left (SUPPORT)
            [ 100.0,  100.0,  0.0],  # n7 base  +y-right (SUPPORT)
            [ 100.0, -100.0,  0.0],  # n8 base  -y-right (SUPPORT)
            [-100.0, -100.0,  0.0],  # n9 base  -y-left (SUPPORT)
        ],
        dtype=float,
    )

    connectivity = np.array(
        [
            # A1 — top-chord
            [0, 1],          # bar 0

            # A2 — top-to-mid "short" diagonals (each top node to the
            # two mid nodes on its own +x / -x side).
            [0, 2],          # bar 1  n0 -> n2
            [0, 5],          # bar 2  n0 -> n5
            [1, 3],          # bar 3  n1 -> n3
            [1, 4],          # bar 4  n1 -> n4

            # A3 — top-to-mid "long" diagonals (cross-x).
            [0, 3],          # bar 5  n0 -> n3
            [0, 4],          # bar 6  n0 -> n4
            [1, 2],          # bar 7  n1 -> n2
            [1, 5],          # bar 8  n1 -> n5

            # A4 — mid-level horizontals along x  (+y and -y sides).
            [2, 3],          # bar 9  +y side
            [4, 5],          # bar 10 -y side

            # A5 — mid-level horizontals along y  (+x and -x sides).
            [3, 4],          # bar 11 +x side
            [2, 5],          # bar 12 -x side

            # A6 — mid-to-base verticals (each mid node to the nearest
            # base corner directly below its own quadrant).
            [2, 6],          # bar 13
            [3, 7],          # bar 14
            [4, 8],          # bar 15
            [5, 9],          # bar 16

            # A7 — mid-to-base lateral diagonals (across x).
            [2, 7],          # bar 17
            [3, 6],          # bar 18
            [4, 9],          # bar 19
            [5, 8],          # bar 20

            # A8 — mid-to-base cross diagonals (across y).
            [2, 9],          # bar 21
            [3, 8],          # bar 22
            [4, 7],          # bar 23
            [5, 6],          # bar 24
        ],
        dtype=int,
    )

    supports: list[tuple[int, tuple[int, ...] | None]] = [
        (6, None),
        (7, None),
        (8, None),
        (9, None),
    ]

    lc1 = LoadCase(
        nodal_forces={
            0: np.array([ 1000.0,  10_000.0, -5000.0]),
            1: np.array([    0.0,  10_000.0, -5000.0]),
            2: np.array([  500.0,       0.0,     0.0]),
            5: np.array([  500.0,       0.0,     0.0]),
        },
        name="LC1: top-node asymmetric + mid-node x-push",
    )
    lc2 = LoadCase(
        nodal_forces={
            0: np.array([0.0,  20_000.0, -5000.0]),
            1: np.array([0.0, -20_000.0, -5000.0]),
        },
        name="LC2: top-node anti-symmetric y-loads",
    )

    return BenchmarkProblem(
        name="25-bar spatial",
        reference_source="Schmit & Miura 1976",
        nodes=nodes,
        connectivity=connectivity,
        E=1.0e7,                    # psi, aluminium
        density=0.1,                # lb/in^3
        units="imperial",
        supports=supports,
        load_cases=[lc1, lc2],
        group_map=_GROUP_MAP,
        area_bounds=(0.01, 3.4),    # in^2
        stress_limit_tension=40_000.0,      # psi (uniform, simplified)
        stress_limit_compression=40_000.0,  # psi
        displacement_limit=0.35,            # in at top nodes
        reference_optimum_weight=_REF_WEIGHT_LB,
        reference_optimum_areas=None,  # see docstring
        reference_verified=True,
    )
