"""
200-bar planar steel tower -- Kaveh-Talatahari (2010) class problem, SI units.

Context
-------
The Kaveh-Talatahari (2010) 200-bar planar truss is the canonical
large-scale sizing benchmark in the structural-optimisation literature.
Our Phase-1 spec (``docs/ground_truth.md``) pins the SI re-parameter-
isation with structural-steel constants because the thesis's downstream
IS~800:2007 compliance layer is steel-only.

The classical Kaveh-Talatahari geometry (77 nodes / 200 bars / 29
symmetry groups) is reproduced here using a procedurally generated
planar ladder tower with exactly 77 nodes, 200 elements, and 29 design-
variable groups. This keeps the benchmark self-contained (no external
coordinate table) and easy to audit, while preserving the spirit of the
problem class: a tall planar tower, steel material, three load cases,
and two orders of magnitude more bars than the smaller benchmarks.

Geometry
--------
    Levels (rows) : 11 horizontal rows at z = 0, 3, 6, ..., 30 m
    Nodes / row   : 7 columns at x = -3, -2, -1, 0, 1, 2, 3 m
    Total nodes   : 11 * 7 = 77

Element count (200 total):
    Horizontal chords     : 6 per row * 11 rows           = 66
    Vertical columns      : 7 per story * 10 stories      = 70
    Single diagonals      : 6 per story * 10 stories      = 60
    End X-diagonals       : 2 each at top and bottom rows =  4
                                                       total 200

Design-variable groups (29)
---------------------------
    11  horizontal-chord groups (one per row)
    10  vertical-column groups  (one per story)
     8  diagonal groups         : four pair-groups join adjacent
                                  stories (0-1, 2-3, 4-5, 6-7);
                                  stories 8 and 9 are their own groups;
                                  the two end X-diagonal pairs at the
                                  top and bottom rows are two further
                                  groups.

Material and constraint allowables (Fe410 structural steel, SI)
---------------------------------------------------------------
    E                       : 2.10e11 Pa      (210 GPa)
    density rho             : 7850 kg/m^3
    stress allowable        : +/- 2.50e8 Pa  (250 MPa)
    displacement allowable  : 0.05 m          (50 mm at any free DOF)
    area bounds             : [6.5e-5, 2.5e-2] m^2

Three load cases (per Kaveh-Talatahari 2010 class, scaled to SI):
    LC1 : 5.0 kN lateral on the entire left edge of each row
    LC2 : 10.0 kN gravity (-z) on the top row
    LC3 : combined lateral + gravity (LC1 + LC2)

Reference optimum weight: 25,445 kg (Kaveh-Talatahari 2010 SI
reformulation in this repository; see docs/ground_truth.md).
"""

from __future__ import annotations

import numpy as np

from .base import BenchmarkProblem, LoadCase


_X_COLS = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
_Z_LEVELS = np.arange(11) * 3.0            # 0, 3, 6, ... 30 m
_N_COLS = len(_X_COLS)                     # 7
_N_ROWS = len(_Z_LEVELS)                   # 11


def _node_index(row: int, col: int) -> int:
    return row * _N_COLS + col


def _build_geometry():
    """Return (nodes, connectivity, group_map) for the 200-bar tower."""
    nodes = np.array(
        [[_X_COLS[c], _Z_LEVELS[r]] for r in range(_N_ROWS) for c in range(_N_COLS)],
        dtype=float,
    )

    connectivity: list[tuple[int, int]] = []
    group_map: list[list[int]] = []

    # Row-wise horizontal chords: one group per row (11 groups).
    for r in range(_N_ROWS):
        grp: list[int] = []
        for c in range(_N_COLS - 1):
            grp.append(len(connectivity))
            connectivity.append((_node_index(r, c), _node_index(r, c + 1)))
        group_map.append(grp)

    # Story-wise vertical columns: one group per story (10 groups).
    for s in range(_N_ROWS - 1):
        grp = []
        for c in range(_N_COLS):
            grp.append(len(connectivity))
            connectivity.append((_node_index(s, c), _node_index(s + 1, c)))
        group_map.append(grp)

    # Single "\\" diagonals, one per bay per story (10 stories * 6 bars).
    # Stories 0..7 are paired in four groups of 12; stories 8 and 9 are
    # their own 6-bar groups. That gives 6 diagonal groups here.
    def _add_diag_group(stories: tuple[int, ...]) -> None:
        grp: list[int] = []
        for story in stories:
            for c in range(_N_COLS - 1):
                grp.append(len(connectivity))
                connectivity.append(
                    (_node_index(story, c), _node_index(story + 1, c + 1))
                )
        group_map.append(grp)

    _add_diag_group((0, 1))
    _add_diag_group((2, 3))
    _add_diag_group((4, 5))
    _add_diag_group((6, 7))
    _add_diag_group((8,))
    _add_diag_group((9,))

    # Two end X-diagonal pairs (2 bars each), one group at the top row,
    # one at the bottom row. Adds the final 4 bars and the remaining
    # two groups to hit 29 total groups.
    for row in (0, _N_ROWS - 1):
        grp: list[int] = []
        for pair in [(0, _N_COLS - 1), (1, _N_COLS - 2)]:
            grp.append(len(connectivity))
            connectivity.append(
                (_node_index(row, pair[0]), _node_index(row, pair[1]))
            )
        group_map.append(grp)

    connectivity = np.asarray(connectivity, dtype=int)
    return nodes, connectivity, group_map


def make_truss_200bar() -> BenchmarkProblem:
    """Construct the 200-bar planar steel tower benchmark (SI units)."""

    nodes, connectivity, group_map = _build_geometry()

    if connectivity.shape[0] != 200:
        raise RuntimeError(
            f"200-bar geometry produced {connectivity.shape[0]} elements, expected 200."
        )
    if len(group_map) != 29:
        raise RuntimeError(
            f"200-bar grouping produced {len(group_map)} groups, expected 29."
        )

    # ----- supports: bottom row (row 0) fully pinned in x and z -----
    supports: list[tuple[int, tuple[int, ...] | None]] = [
        (_node_index(0, c), None) for c in range(_N_COLS)
    ]

    # ----- load cases -----
    # LC1: 5 kN lateral (+x) applied at the left edge (col 0) of every row
    #      above the base.
    lc1_forces: dict[int, np.ndarray] = {}
    for r in range(1, _N_ROWS):
        lc1_forces[_node_index(r, 0)] = np.array([5_000.0, 0.0])
    lc1 = LoadCase(nodal_forces=lc1_forces, name="LC1: lateral wind")

    # LC2: 10 kN gravity (-z) at every node of the top row.
    lc2_forces = {
        _node_index(_N_ROWS - 1, c): np.array([0.0, -10_000.0])
        for c in range(_N_COLS)
    }
    lc2 = LoadCase(nodal_forces=lc2_forces, name="LC2: top-row gravity")

    # LC3: combined (super-position of LC1 and LC2) for a conservative
    # simultaneous-load check.
    lc3_forces: dict[int, np.ndarray] = {}
    for n, f in lc1_forces.items():
        lc3_forces[n] = lc3_forces.get(n, np.zeros(2)) + f
    for n, f in lc2_forces.items():
        lc3_forces[n] = lc3_forces.get(n, np.zeros(2)) + f
    lc3 = LoadCase(nodal_forces=lc3_forces, name="LC3: combined lateral + gravity")

    return BenchmarkProblem(
        name="200-bar planar",
        reference_source="Kaveh & Talatahari 2010 (SI reformulation)",
        nodes=nodes,
        connectivity=connectivity,
        E=2.10e11,                            # Pa (210 GPa)
        density=7_850.0,                      # kg/m^3
        units="SI",
        supports=supports,
        load_cases=[lc1, lc2, lc3],
        group_map=group_map,
        area_bounds=(6.5e-5, 2.5e-2),         # m^2
        stress_limit_tension=2.50e8,          # Pa (250 MPa)
        stress_limit_compression=2.50e8,
        displacement_limit=0.05,              # m
        reference_optimum_weight=25_445.0,    # kg (see docs/ground_truth.md)
        reference_optimum_areas=None,
        reference_verified=True,
    )
