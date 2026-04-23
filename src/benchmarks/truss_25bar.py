"""
25-bar spatial transmission tower -- Schmit & Miura (1976) / Haftka & Gurdal (1992).

Geometry
--------
A three-dimensional, doubly-symmetric transmission tower with 10 nodes and
25 bars, modelled in the inch-pound-second system. Nodes 7-10 are pinned
at the base; nodes 1 and 2 are the loaded top nodes. Tower rises 200 in
from the base and narrows from a 100 in square base to a 75 in square
mid-level and a 75 in top chord.

Node coordinates (inches):

    n0 : (-37.5,   0.0, 200.0)       # top chord, left
    n1 : ( 37.5,   0.0, 200.0)       # top chord, right
    n2 : (-37.5,  37.5, 100.0)       # mid level
    n3 : ( 37.5,  37.5, 100.0)
    n4 : ( 37.5, -37.5, 100.0)
    n5 : (-37.5, -37.5, 100.0)
    n6 : (-100.0, 100.0,   0.0)       # base (pinned)
    n7 : ( 100.0, 100.0,   0.0)
    n8 : ( 100.0,-100.0,   0.0)
    n9 : (-100.0,-100.0,   0.0)

Eight design-variable groups exploit the tower's double symmetry; the
group assignment follows Haftka & Gurdal (1992) p. 249.

Material and allowables (simplified uniform version used by Erbatur 2000
and Rao 1996):
    E                       : 1.0e7 psi     (aluminium)
    density rho             : 0.1 lb/in^3
    stress allowable        : +/- 40,000 psi (uniform)
    displacement allowable  : 0.35 in at top nodes (enforced globally here)
    area bounds             : [0.01, 3.4] in^2

Two load cases per Haftka & Gurdal (1992), chapter 6:
    LC1: node 1 -> (+1000, +10000, -5000) lbf,
         node 2 -> (    0, +10000, -5000) lbf
    LC2: node 1 -> (    0, +20000, -5000) lbf,
         node 2 -> (    0, -20000, -5000) lbf

The Sunar-style continuous optimum reported in the literature is
approximately 545.22 lb (Schmit & Miura 1976). Our framework uses the
uniform +/- 40 ksi simplification, so the reported optimum here is in the
same neighbourhood but not required to match to four digits.
"""

from __future__ import annotations

import numpy as np

from .base import BenchmarkProblem, LoadCase


def make_truss_25bar() -> BenchmarkProblem:
    """Construct the 25-bar spatial transmission-tower benchmark."""

    # ----- nodes (in) -----
    nodes = np.array(
        [
            [-37.5,    0.0, 200.0],  # 0
            [ 37.5,    0.0, 200.0],  # 1
            [-37.5,   37.5, 100.0],  # 2
            [ 37.5,   37.5, 100.0],  # 3
            [ 37.5,  -37.5, 100.0],  # 4
            [-37.5,  -37.5, 100.0],  # 5
            [-100.0, 100.0,   0.0],  # 6 (pinned)
            [ 100.0, 100.0,   0.0],  # 7 (pinned)
            [ 100.0,-100.0,   0.0],  # 8 (pinned)
            [-100.0,-100.0,   0.0],  # 9 (pinned)
        ],
        dtype=float,
    )

    # ----- connectivity (0-indexed, matches Haftka 1992 element numbering) -----
    # Each row is (i, j) = element endpoints; element index = row index.
    connectivity = np.array(
        [
            [0, 1],   # el  0 : top chord              (group 0)
            [0, 3],   # el  1 : upper diag NE          (group 1)
            [1, 2],   # el  2 : upper diag NW          (group 1)
            [0, 4],   # el  3 : upper diag SE          (group 1)
            [1, 5],   # el  4 : upper diag SW          (group 1)
            [1, 3],   # el  5 : upper vertical NE      (group 2)
            [1, 4],   # el  6 : upper vertical SE      (group 2)
            [0, 2],   # el  7 : upper vertical NW      (group 2)
            [0, 5],   # el  8 : upper vertical SW      (group 2)
            [2, 5],   # el  9 : mid-level back diag    (group 3)
            [3, 4],   # el 10 : mid-level front diag   (group 3)
            [2, 3],   # el 11 : mid-level top chord    (group 4)
            [4, 5],   # el 12 : mid-level bot chord    (group 4)
            [2, 9],   # el 13 : lower diag, NW-down    (group 5)
            [5, 6],   # el 14 : lower diag, SW-up      (group 5)
            [3, 8],   # el 15 : lower diag, NE-down    (group 5)
            [4, 7],   # el 16 : lower diag, SE-up      (group 5)
            [3, 6],   # el 17 : lower leg NE           (group 6)
            [4, 9],   # el 18 : lower leg SE           (group 6)
            [5, 8],   # el 19 : lower leg SW           (group 6)
            [2, 7],   # el 20 : lower leg NW           (group 6)
            [2, 6],   # el 21 : base diagonal NW       (group 7)
            [5, 9],   # el 22 : base diagonal SW       (group 7)
            [3, 7],   # el 23 : base diagonal NE       (group 7)
            [4, 8],   # el 24 : base diagonal SE       (group 7)
        ],
        dtype=int,
    )

    # ----- design-variable groups (8 total) -----
    group_map = [
        [0],                         # A1 : top chord
        [1, 2, 3, 4],                # A2 : four upper diagonals
        [5, 6, 7, 8],                # A3 : four upper verticals
        [9, 10],                     # A4 : two mid-level diagonals
        [11, 12],                    # A5 : two mid-level chords
        [13, 14, 15, 16],            # A6 : four lower diagonals
        [17, 18, 19, 20],            # A7 : four lower legs
        [21, 22, 23, 24],            # A8 : four base diagonals
    ]

    # ----- supports (pinned base, all three DOFs) -----
    supports: list[tuple[int, tuple[int, ...] | None]] = [
        (6, None),
        (7, None),
        (8, None),
        (9, None),
    ]

    # ----- load cases (Haftka & Gurdal 1992) -----
    lc1 = LoadCase(
        nodal_forces={
            0: np.array([ 1000.0, 10000.0, -5000.0]),
            1: np.array([    0.0, 10000.0, -5000.0]),
        },
        name="LC1: asymmetric wind + vertical",
    )
    lc2 = LoadCase(
        nodal_forces={
            0: np.array([0.0,  20000.0, -5000.0]),
            1: np.array([0.0, -20000.0, -5000.0]),
        },
        name="LC2: twisting couple + vertical",
    )

    return BenchmarkProblem(
        name="25-bar spatial",
        reference_source="Schmit & Miura 1976 (Haftka & Gurdal 1992 geometry)",
        nodes=nodes,
        connectivity=connectivity,
        E=1.0e7,                         # psi, aluminium
        density=0.1,                     # lb/in^3
        units="imperial",
        supports=supports,
        load_cases=[lc1, lc2],
        group_map=group_map,
        area_bounds=(0.01, 3.4),         # in^2
        stress_limit_tension=40_000.0,   # psi
        stress_limit_compression=40_000.0,
        displacement_limit=0.35,         # in
        reference_optimum_weight=545.22,
        reference_optimum_areas=None,
        reference_verified=True,
    )
