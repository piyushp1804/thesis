"""
72-bar spatial tower -- Venkayya (1971) / Erbatur et al. (2000).

A four-story square-cross-section tower, 240 in tall, 120 in on a side.
Twenty nodes, 72 bars. The base is pinned (four corner supports) and all
four top nodes are loaded. Canonical in the structural-optimisation
literature alongside the 10-bar and 25-bar problems.

Layout
------
Five horizontal levels (z = 240, 180, 120, 60, 0). Each level has four
corners numbered counter-clockwise viewed from above:
    corner 0: (  0,   0, z)
    corner 1: (120,   0, z)
    corner 2: (120, 120, z)
    corner 3: (  0, 120, z)

We index nodes top-down so node 0 is the top-left, node 19 is the base
corner opposite.

Each of the four stories contributes 18 bars:
    4 columns (verticals joining the two adjacent levels),
    8 face diagonals (two per face; four faces per story),
    4 horizontal chords (the square at the TOP of the story),
    2 horizontal diagonals (crossing the floor at the TOP of the story).

Which gives 4 * 18 = 72 bars total.

Design grouping (Erbatur 2000, 16 groups)
-----------------------------------------
Per-story:
    group 4k + 0 : 4 columns of story k
    group 4k + 1 : 8 face diagonals of story k
    group 4k + 2 : 4 top-chord bars of story k
    group 4k + 3 : 2 top horizontal diagonals of story k

with k = 0 for the top story and k = 3 for the bottom story.

Material, allowables, load cases (Erbatur 2000)
----------------------------------------------
    E                     : 1.0e7 psi (aluminium)
    density rho           : 0.1 lb/in^3
    stress allowable      : +/- 25,000 psi
    displacement limit    : 0.25 in at top nodes (enforced globally here)
    area bounds           : [0.1, 4.0] in^2

    LC1: Node 0 (top NW) loaded (5000, 5000, -5000) lbf
    LC2: Each of the four top nodes loaded (0, 0, -5000) lbf

Published continuous optimum: 379.62 lb (Erbatur et al. 2000).
"""

from __future__ import annotations

import numpy as np

from .base import BenchmarkProblem, LoadCase


# Corner offsets in the (x, y) plane, counter-clockwise from the origin.
_CORNERS_XY = np.array(
    [
        [  0.0,   0.0],   # corner 0
        [120.0,   0.0],   # corner 1
        [120.0, 120.0],   # corner 2
        [  0.0, 120.0],   # corner 3
    ],
    dtype=float,
)

_LEVEL_Z = [240.0, 180.0, 120.0, 60.0, 0.0]   # top-down


def _build_geometry():
    """Return (nodes, connectivity, group_map) for the 72-bar tower."""
    # Nodes: level L (top=0..bottom=4) x corner (0..3) -> index 4*L + corner
    nodes = np.array(
        [[_CORNERS_XY[c, 0], _CORNERS_XY[c, 1], z]
         for z in _LEVEL_Z for c in range(4)],
        dtype=float,
    )

    def node(level: int, corner: int) -> int:
        return 4 * level + corner

    connectivity: list[tuple[int, int]] = []
    group_map: list[list[int]] = []

    # Faces, each given as (corner_a, corner_b) pair forming one of the
    # four vertical sides of the tower. Diagonals cross on each face.
    face_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for story in range(4):            # story 0 = top, story 3 = bottom
        upper = story                 # upper level of this story
        lower = story + 1             # lower level
        top_chords = []
        face_diags = []
        columns = []
        top_diags = []

        # 1) columns (verticals between the two levels, one per corner)
        for c in range(4):
            columns.append(len(connectivity))
            connectivity.append((node(upper, c), node(lower, c)))

        # 2) face diagonals (two per face, 8 total)
        for (a, b) in face_pairs:
            # crossing diagonals on face with corners a-b top and a-b bottom
            face_diags.append(len(connectivity))
            connectivity.append((node(upper, a), node(lower, b)))
            face_diags.append(len(connectivity))
            connectivity.append((node(upper, b), node(lower, a)))

        # 3) horizontal chords around the square at the TOP of this story
        for (a, b) in face_pairs:
            top_chords.append(len(connectivity))
            connectivity.append((node(upper, a), node(upper, b)))

        # 4) horizontal diagonals across the floor at the TOP of this story
        top_diags.append(len(connectivity))
        connectivity.append((node(upper, 0), node(upper, 2)))
        top_diags.append(len(connectivity))
        connectivity.append((node(upper, 1), node(upper, 3)))

        group_map.extend([columns, face_diags, top_chords, top_diags])

    connectivity = np.asarray(connectivity, dtype=int)
    return nodes, connectivity, group_map


def make_truss_72bar() -> BenchmarkProblem:
    """Construct the 72-bar spatial-tower benchmark."""

    nodes, connectivity, group_map = _build_geometry()

    # ----- supports: four base corners pinned (x, y, z) -----
    base = 4 * 4  # first index of bottom level = 16
    supports: list[tuple[int, tuple[int, ...] | None]] = [
        (base + 0, None),
        (base + 1, None),
        (base + 2, None),
        (base + 3, None),
    ]

    # ----- load cases (Erbatur 2000) -----
    lc1 = LoadCase(
        nodal_forces={0: np.array([5000.0, 5000.0, -5000.0])},
        name="LC1: asymmetric corner loading",
    )
    lc2 = LoadCase(
        nodal_forces={
            0: np.array([0.0, 0.0, -5000.0]),
            1: np.array([0.0, 0.0, -5000.0]),
            2: np.array([0.0, 0.0, -5000.0]),
            3: np.array([0.0, 0.0, -5000.0]),
        },
        name="LC2: symmetric top vertical loading",
    )

    return BenchmarkProblem(
        name="72-bar spatial",
        reference_source="Venkayya 1971 / Erbatur 2000",
        nodes=nodes,
        connectivity=connectivity,
        E=1.0e7,
        density=0.1,
        units="imperial",
        supports=supports,
        load_cases=[lc1, lc2],
        group_map=group_map,
        area_bounds=(0.1, 4.0),
        stress_limit_tension=25_000.0,
        stress_limit_compression=25_000.0,
        displacement_limit=0.25,
        reference_optimum_weight=379.62,
        reference_optimum_areas=None,
        reference_verified=True,
    )
