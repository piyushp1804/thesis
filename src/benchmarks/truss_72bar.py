"""
72-bar spatial tower — Fleury & Schmit (1980); Erbatur et al. (2000).

4-storey spatial tower; 20 nodes, 72 bars, 16 symmetry groups. Widely
used as a mid-complexity sizing benchmark. Our encoding follows the
standard Fleury-Schmit 1980 geometry with uniform ±25,000 psi stress
limits and 0.25 in displacement limit.

Formulation: the 0.25 in displacement limit applies only to the lateral
(x and y) displacements of the four tip nodes (12-15), per Camp &
Bichon 2004 / Bekdaş 2015. Vertical (z) displacements and lower-storey
nodes are unconstrained — these are wired through the
`displacement_check_dofs` field on `BenchmarkProblem`.

Geometry (inches):

   z = 240  (tip)     n12----n13----n14----n15            (4 tip nodes)
                       |      |      |      |
   z = 180            n8-----n9-----n10----n11            (storey-3 top)
                       |      |      |      |
   z = 120            n4-----n5-----n6-----n7             (storey-2 top)
                       |      |      |      |
   z =  60            n0-----n1-----n2-----n3             (storey-1 top)
                       |      |      |      |
   z =   0            n16----n17----n18----n19            (base, SUPPORTS)

Each storey holds a 120 in × 120 in square layout: corners at
(±60, ±60, z). Node numbering within a storey layer is:
    +x+y:  [0]   +x-y:  [1]    -x-y:  [2]    -x+y:  [3]

Per storey, 18 bars:
    4 columns (vertical between levels)
    4 top horizontals (storey-top square)
    4 top "K-braces" (diagonals in vertical face, top→opposite-corner bot)
    4 top "U-braces" (diagonals in vertical face, bot→opposite-corner top)
    2 plan diagonals (top horizontal square, corner-to-corner)

4 storeys × 18 = 72 bars.

Per storey, 4 design groups (ordering matches Schmit-Farshi 1974 / Camp 2007
so that published optima can be plugged in directly):
    Columns / legs (4 bars)
    Face diagonals (8 bars: 4 K + 4 U)
    Top horizontals (4 bars)
    Plan diagonals (2 bars)

4 storeys × 4 groups = 16 design variables.

Two load cases per Fleury & Schmit 1980 / Erbatur 2000:
    LC1: (5000, 5000, -5000) lbf at tip node 12 only (one corner;
         asymmetric → worst torsional case)
    LC2: (0, 0, -5000) lbf at each of the 4 tip nodes (uniform
         vertical compression)
"""

from __future__ import annotations

import numpy as np

from .base import BenchmarkProblem, LoadCase


_REF_WEIGHT_LB = 379.62

# Base-square half-width, storey height.
_HALF = 60.0
_DZ = 60.0


# Corner ordering within a layer: (+x,+y), (+x,-y), (-x,-y), (-x,+y).
_CORNER_OFFSETS = np.array(
    [
        [ _HALF,  _HALF],
        [ _HALF, -_HALF],
        [-_HALF, -_HALF],
        [-_HALF,  _HALF],
    ],
    dtype=float,
)


def _build_nodes() -> np.ndarray:
    """20 nodes: 4 nodes per storey-top × 4 storeys + 4 base supports.

    Node layout:
        0-3   : storey-1 top  (z = 60)
        4-7   : storey-2 top  (z = 120)
        8-11  : storey-3 top  (z = 180)
        12-15 : storey-4 top / tip  (z = 240)
        16-19 : base supports  (z = 0)
    """
    nodes = np.zeros((20, 3), dtype=float)
    for storey in range(4):          # storey-top levels at z = 60, 120, 180, 240
        z = (storey + 1) * _DZ
        for k in range(4):
            nodes[4 * storey + k, :2] = _CORNER_OFFSETS[k]
            nodes[4 * storey + k, 2] = z
    # Base (z = 0).
    for k in range(4):
        nodes[16 + k, :2] = _CORNER_OFFSETS[k]
        nodes[16 + k, 2] = 0.0
    return nodes


def _build_connectivity() -> tuple[np.ndarray, list[list[int]]]:
    """Return (connectivity, group_map) following the docstring layout."""

    conn: list[tuple[int, int]] = []
    groups: list[list[int]] = []

    # Per storey: bot-layer node indices, top-layer node indices.
    # Storey 0 connects base (16..19) to level-1 (0..3)
    # Storey 1 connects level-1 (0..3) to level-2 (4..7), etc.
    storey_layers = [
        (list(range(16, 20)), list(range(0, 4))),   # storey 0 (base -> top-1)
        (list(range(0, 4)),   list(range(4, 8))),   # storey 1
        (list(range(4, 8)),   list(range(8, 12))),  # storey 2
        (list(range(8, 12)),  list(range(12, 16))), # storey 3 (top)
    ]

    for bot, top in storey_layers:
        # Groups per storey: [columns, top_horiz, face_diag, plan_diag]
        g_col: list[int] = []
        g_top: list[int] = []
        g_face: list[int] = []
        g_plan: list[int] = []

        # Columns: bot[k] -> top[k] for k in 0..3  (4 bars)
        for k in range(4):
            g_col.append(len(conn))
            conn.append((bot[k], top[k]))

        # Top horizontals: top square edges  (4 bars)
        for k in range(4):
            g_top.append(len(conn))
            conn.append((top[k], top[(k + 1) % 4]))

        # Face diagonals — 8 bars per storey.
        # Four "K-braces": top[k] -> bot[(k+1) % 4]
        # Four "U-braces": bot[k] -> top[(k+1) % 4]
        for k in range(4):
            g_face.append(len(conn))
            conn.append((top[k], bot[(k + 1) % 4]))
        for k in range(4):
            g_face.append(len(conn))
            conn.append((bot[k], top[(k + 1) % 4]))

        # Plan diagonals: top-square corner to opposite corner (2 bars).
        g_plan.append(len(conn))
        conn.append((top[0], top[2]))
        g_plan.append(len(conn))
        conn.append((top[1], top[3]))

        groups.extend([g_col, g_face, g_top, g_plan])

    connectivity = np.array(conn, dtype=int)
    return connectivity, groups


def make_truss_72bar() -> BenchmarkProblem:
    """Construct the 72-bar spatial tower problem."""

    nodes = _build_nodes()
    connectivity, group_map = _build_connectivity()

    # Base nodes 16..19 fully pinned.
    supports: list[tuple[int, tuple[int, ...] | None]] = [
        (16, None),
        (17, None),
        (18, None),
        (19, None),
    ]

    # LC1: asymmetric load at one tip corner (Fleury 1980 canonical case).
    lc1 = LoadCase(
        nodal_forces={
            12: np.array([5000.0, 5000.0, -5000.0]),
        },
        name="LC1: 5-kip (x,y,-z) at tip corner (asymmetric)",
    )

    # LC2: uniform vertical compression at all 4 tip nodes.
    lc2 = LoadCase(
        nodal_forces={
            n: np.array([0.0, 0.0, -5000.0]) for n in range(12, 16)
        },
        name="LC2: 5-kip downward at each tip node",
    )

    # Literature constraint: lateral (x,y) displacement at the four tip
    # nodes ≤ 0.25 in. Z and lower-storey nodes are unconstrained.
    # See Camp & Bichon (2004), Erbatur (2000), Bekdaş (2015).
    tip_nodes = (12, 13, 14, 15)
    disp_dofs: list[tuple[int, int]] = [
        (n, axis) for n in tip_nodes for axis in (0, 1)
    ]

    return BenchmarkProblem(
        name="72-bar spatial tower",
        reference_source="Fleury & Schmit 1980 / Erbatur 2000",
        nodes=nodes,
        connectivity=connectivity,
        E=1.0e7,                    # psi, aluminium
        density=0.1,                # lb/in^3
        units="imperial",
        supports=supports,
        load_cases=[lc1, lc2],
        group_map=group_map,
        area_bounds=(0.1, 4.0),     # in^2
        stress_limit_tension=25_000.0,
        stress_limit_compression=25_000.0,
        displacement_limit=0.25,    # in, lateral, on tip nodes only
        displacement_check_dofs=disp_dofs,
        reference_optimum_weight=_REF_WEIGHT_LB,
        reference_optimum_areas=None,
        reference_verified=True,
    )
