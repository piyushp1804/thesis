"""
10-bar planar cantilever truss — Sunar & Belegundu (1991).

The canonical sizing-optimization test problem. Every structural-
optimization paper since the early 1990s reports its number here, so
matching published optima to three sig figs is the single most important
FEM / GA validation gate in the thesis.

Geometry (units: inches):

        n0 --(2)-- n2 --(1)-- n4  (supports at n4, n5)
        |\\  (10)  |\\  (8)   |
        | \\       | \\       |
        |  \\      |  \\      |
       (6)  (9)   (5)  (7)   |
        |    \\    |    \\    |
        |     \\   |     \\   |
        |      \\  |      \\  |
        n1 --(4)-- n3 --(3)-- n5

Element list (0-indexed nodes; element number in parens matches Sunar 1991):

    el 0  (paper 1)  n4 - n2   top chord, left
    el 1  (paper 2)  n2 - n0   top chord, right
    el 2  (paper 3)  n5 - n3   bot chord, left
    el 3  (paper 4)  n3 - n1   bot chord, right
    el 4  (paper 5)  n2 - n3   mid vertical
    el 5  (paper 6)  n0 - n1   right vertical
    el 6  (paper 7)  n3 - n4   diagonal 1
    el 7  (paper 8)  n2 - n5   diagonal 2
    el 8  (paper 9)  n1 - n2   diagonal 3
    el 9  (paper 10) n0 - n3   diagonal 4

Loads: -100 kips (i.e. -100000 lbf) in y at nodes 1 and 3 (the two
bottom free nodes).

Supports: nodes 4 and 5 (the left wall) — pinned in x and y.

No symmetry grouping — all 10 areas are independent design variables,
which is how the benchmark is posed in the original paper.
"""

from __future__ import annotations

import numpy as np

from .base import BenchmarkProblem, LoadCase


# Published optimum from Sunar & Belegundu (1991).
# Order matches our element indexing above (paper's 1..10 -> our 0..9).
_REF_AREAS = np.array(
    [30.52, 0.100, 23.20, 15.22, 0.100, 0.551, 7.457, 21.04, 21.53, 0.100],
    dtype=float,
)
_REF_WEIGHT_LB = 5060.85  # lb, per Sunar 1991


def make_truss_10bar() -> BenchmarkProblem:
    """Construct the 10-bar planar truss problem."""

    # ----- nodes (in) -----
    nodes = np.array(
        [
            [720.0, 360.0],  # n0 : top-right (free)
            [720.0,   0.0],  # n1 : bot-right (free, loaded)
            [360.0, 360.0],  # n2 : top-mid  (free)
            [360.0,   0.0],  # n3 : bot-mid  (free, loaded)
            [  0.0, 360.0],  # n4 : top-left (SUPPORT)
            [  0.0,   0.0],  # n5 : bot-left (SUPPORT)
        ],
        dtype=float,
    )

    # ----- elements (0-indexed) -----
    connectivity = np.array(
        [
            [4, 2],  # el 0 — top chord, left
            [2, 0],  # el 1 — top chord, right
            [5, 3],  # el 2 — bot chord, left
            [3, 1],  # el 3 — bot chord, right
            [2, 3],  # el 4 — mid vertical
            [0, 1],  # el 5 — right vertical
            [3, 4],  # el 6 — diagonal
            [2, 5],  # el 7 — diagonal
            [1, 2],  # el 8 — diagonal
            [0, 3],  # el 9 — diagonal
        ],
        dtype=int,
    )

    # ----- supports (pin both DOFs of n4 and n5) -----
    supports: list[tuple[int, tuple[int, ...] | None]] = [
        (4, None),
        (5, None),
    ]

    # ----- single load case: -100 kips at bottom free nodes -----
    load_case = LoadCase(
        nodal_forces={
            1: np.array([0.0, -100_000.0]),  # n1
            3: np.array([0.0, -100_000.0]),  # n3
        },
        name="LC1: 100-kip vertical loads",
    )

    # No grouping — 10 independent variables.
    group_map = [[i] for i in range(10)]

    return BenchmarkProblem(
        name="10-bar planar",
        reference_source="Sunar & Belegundu 1991",
        nodes=nodes,
        connectivity=connectivity,
        E=1.0e7,                   # psi, aluminium
        density=0.1,               # lb/in^3
        units="imperial",
        supports=supports,
        load_cases=[load_case],
        group_map=group_map,
        area_bounds=(0.1, 35.0),   # in^2
        stress_limit_tension=25_000.0,     # psi
        stress_limit_compression=25_000.0, # psi
        displacement_limit=2.0,            # in
        reference_optimum_weight=_REF_WEIGHT_LB,
        reference_optimum_areas=_REF_AREAS,
        reference_verified=True,
    )
