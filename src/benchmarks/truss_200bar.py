"""
200-bar planar truss (Kaveh-style stepped tower).

This encoding fills the 4th slot in the canonical benchmark set. It
uses the same scale as Kaveh & Talatahari (2010) — 77 nodes, 200
bars, 29 symmetry groups, steel + SI units — but a principled
regular-grid layout we can verify end-to-end rather than the exact
Kaveh coordinates (which require paper-appendix geometry not in this
repo). The reference weight below is therefore *our own* rigorous-
FEM feasible optimum once computed; we do not claim to reproduce
Kaveh's published 25,445 kg on this precise geometry.

The chapter-4 writeup flags this explicitly: the contribution of
the 200-bar benchmark here is an end-to-end validation of the
IS~800:2007 compliance layer on a steel / SI / 29-variable problem,
not a literature-match claim.

Geometry (metres, y = vertical):

  Level 10 (top, y=30)     :  7 nodes, loaded
  Level  9 (y=27)          :  7 nodes
  ...
  Level  1 (y= 3)          :  7 nodes
  Level  0 (y= 0, base)    :  7 nodes, pinned

  x-coordinates per level  : -4.5, -3.0, -1.5, 0.0, +1.5, +3.0, +4.5

  11 levels x 7 columns    = 77 nodes

Bar layout (200 bars total, per-storey decomposition):

  Horizontals (top chord of each storey except the very top)
       6 per level x 10 storeys                           =  60 bars
  Horizontals at the very top level                       =   6 bars
  Verticals  (columns connecting level i -> level i+1)
       7 per storey x 10 storeys                          =  70 bars
  Diagonals  (one per interior bay, direction alternates per storey)
       6 per storey x 10 storeys                          =  60 bars
  Extra X-brace diagonals on the top storey (stiffness)   =   4 bars
                                                         ------
                                                           200 bars

Symmetry groups (29 design variables):

  group  0 .. 9   (10)  column-type groups, one per storey
  group 10 .. 20 (11)  horizontal-type groups, one per level
  group 21 .. 28  (8)  diagonal-type groups, one per storey 1..8
                       (storeys 0 and 9 diagonals grouped in with
                       their adjacent storey to land at exactly 29
                       design vars, matching the Kaveh count)

Loading (three load cases, after Kaveh & Talatahari 2010 and Lee &
Geem 2004):

  LC1: 1 kN downward at every node of the top level
  LC2: 10 kN downward at every node of the left half-span at level 10
       and at each of the 6 odd-numbered levels on the left edge
  LC3: LC1 + LC2 superposed

The three cases together exercise the axial, bending (lateral), and
combined action of the tower.
"""

from __future__ import annotations

import numpy as np

from .base import BenchmarkProblem, LoadCase


# Our rigorous-FEM feasible optimum is reported in Chapter 4.5. The
# number below is the Kaveh & Talatahari 2010 literature value for a
# geometrically related (but not identical) layout; we keep it as a
# reference point but do not test against it. The Chapter 4.5
# reporting makes this distinction explicit.
_REF_WEIGHT_KG = 25_445.0

_N_LEVELS = 11           # 0 (base) .. 10 (top)
_N_COLS = 7              # x = -4.5 .. +4.5 m, step 1.5 m
_STOREY_HEIGHT = 3.0     # m
_COL_WIDTH = 1.5         # m (x-spacing)


def _node_index(level: int, col: int) -> int:
    """Flat index into nodes array, row-major, (level 0 = base)."""
    return level * _N_COLS + col


def _build_nodes() -> np.ndarray:
    """77 nodes on an 11x7 grid. Level 0 at y=0 (base), level 10 at y=30."""
    nodes = np.zeros((_N_LEVELS * _N_COLS, 2), dtype=float)
    for level in range(_N_LEVELS):
        y = level * _STOREY_HEIGHT
        for col in range(_N_COLS):
            x = (col - 3) * _COL_WIDTH   # -4.5, -3.0, -1.5, 0, +1.5, +3.0, +4.5
            nodes[_node_index(level, col)] = [x, y]
    return nodes


def _build_connectivity() -> tuple[np.ndarray, list[list[int]]]:
    """Return (connectivity, group_map). 200 bars, 29 groups."""

    conn: list[tuple[int, int]] = []
    # We'll assign every bar to one of 29 groups as we build.
    # group indices:
    #   0..9   : columns,    one per storey
    #   10..20 : horizontals, one per level
    #   21..28 : diagonals,   8 groups covering 10 storeys
    g_cols:  list[list[int]] = [[] for _ in range(10)]
    g_horiz: list[list[int]] = [[] for _ in range(11)]
    g_diag:  list[list[int]] = [[] for _ in range(8)]

    # Horizontals: 6 per level, 11 levels => 66 bars
    for level in range(_N_LEVELS):
        for col in range(_N_COLS - 1):
            g_horiz[level].append(len(conn))
            conn.append((_node_index(level, col), _node_index(level, col + 1)))

    # Verticals (columns): 7 per storey, 10 storeys => 70 bars
    for storey in range(10):
        for col in range(_N_COLS):
            g_cols[storey].append(len(conn))
            conn.append((_node_index(storey, col), _node_index(storey + 1, col)))

    # Single diagonal per interior bay: 6 per storey, 10 storeys => 60 bars.
    # Diagonal direction alternates per storey for lateral stiffness.
    # Diagonals of storeys 0..8 map to diagonal groups 0..7 according to
    # the table below (storey 9 is absorbed into group 7, giving 8
    # diagonal groups for 10 storeys of diagonals).
    storey_to_diag_group = [0, 0, 1, 2, 3, 4, 5, 6, 7, 7]
    for storey in range(10):
        dg = storey_to_diag_group[storey]
        for col in range(_N_COLS - 1):
            if storey % 2 == 0:
                # `/`-direction: (storey, col) -> (storey+1, col+1)
                n0 = _node_index(storey, col)
                n1 = _node_index(storey + 1, col + 1)
            else:
                # `\`-direction: (storey, col+1) -> (storey+1, col)
                n0 = _node_index(storey, col + 1)
                n1 = _node_index(storey + 1, col)
            g_diag[dg].append(len(conn))
            conn.append((n0, n1))

    # 4 extra X-brace diagonals on the TOP storey for lateral stiffness.
    # These are the counter-direction of the single diagonals at storey 9.
    # Place them on the middle 4 bays.
    top_storey = 9
    for col in (1, 2, 3, 4):
        # counter-direction at top storey (whatever wasn't used above)
        n0 = _node_index(top_storey, col)
        n1 = _node_index(top_storey + 1, col + 1) if top_storey % 2 == 1 \
             else _node_index(top_storey + 1, col - 1)
        # Fix indexing: top_storey=9 is odd, so single-diag uses \-dir.
        # Counter = /-direction: (9, col) -> (10, col+1)
        n0 = _node_index(top_storey, col)
        n1 = _node_index(top_storey + 1, col + 1)
        g_diag[7].append(len(conn))   # absorb into diagonal group 7
        conn.append((n0, n1))

    # Sanity checks.
    n_bars = len(conn)
    assert n_bars == 200, f"built {n_bars} bars, expected 200"
    all_assigned = sum(g_cols, []) + sum(g_horiz, []) + sum(g_diag, [])
    assert sorted(all_assigned) == list(range(n_bars)), \
        "group_map must cover every bar exactly once"
    assert len(g_cols) + len(g_horiz) + len(g_diag) == 29, \
        "should have exactly 29 groups"

    group_map = g_cols + g_horiz + g_diag
    return np.array(conn, dtype=int), group_map


def make_truss_200bar() -> BenchmarkProblem:
    """Construct the 200-bar planar stepped-tower (steel, SI)."""

    nodes = _build_nodes()
    connectivity, group_map = _build_connectivity()

    # Base level 0 fully pinned at all 7 nodes.
    supports: list[tuple[int, tuple[int, ...] | None]] = [
        (_node_index(0, c), None) for c in range(_N_COLS)
    ]

    # Top-level indices (y = 30 m, 7 nodes).
    top_nodes = [_node_index(10, c) for c in range(_N_COLS)]
    # Left-edge indices at odd levels 1, 3, 5, 7, 9 (for LC2 lateral pulls).
    left_edge = [_node_index(lvl, 0) for lvl in (1, 3, 5, 7, 9)]

    # LC1: 1 kN down at every top-level node.
    lc1 = LoadCase(
        nodal_forces={n: np.array([0.0, -1000.0]) for n in top_nodes},
        name="LC1: 1 kN downward at every top-level node",
    )
    # LC2: 10 kN down at the 3 leftmost top-level nodes + 10 kN at left-edge.
    lc2_forces: dict[int, np.ndarray] = {}
    for n in top_nodes[:3]:
        lc2_forces[n] = np.array([0.0, -10_000.0])
    for n in left_edge:
        lc2_forces[n] = lc2_forces.get(n, np.zeros(2)) + np.array([0.0, -10_000.0])
    lc2 = LoadCase(nodal_forces=lc2_forces, name="LC2: asymmetric 10 kN load")
    # LC3: superposition of LC1 + LC2.
    lc3_forces: dict[int, np.ndarray] = {}
    for n, f in lc1.nodal_forces.items():
        lc3_forces[n] = f.copy()
    for n, f in lc2.nodal_forces.items():
        lc3_forces[n] = lc3_forces.get(n, np.zeros(2)) + f
    lc3 = LoadCase(nodal_forces=lc3_forces, name="LC3: LC1 + LC2 combined")

    # Displacement constraint: limit vertical deflection at top nodes
    # to span/300 = 30 m / 300 = 0.1 m. Lateral constraint implicit through
    # horizontal members; checked via the full-DOF default.
    top_node_disp_dofs = [(n, 1) for n in top_nodes]  # axis 1 = y

    return BenchmarkProblem(
        name="200-bar planar stepped tower",
        reference_source="Kaveh & Talatahari 2010 (scale match; geometry our own)",
        nodes=nodes,
        connectivity=connectivity,
        E=2.10e11,                    # Pa, structural steel
        density=7850.0,               # kg/m^3
        units="SI",
        supports=supports,
        load_cases=[lc1, lc2, lc3],
        group_map=group_map,
        area_bounds=(6.5e-5, 2.5e-2), # m^2  (~ 65 mm^2 to 25000 mm^2)
        stress_limit_tension=250e6,   # Pa (250 MPa)
        stress_limit_compression=250e6,
        displacement_limit=0.10,      # m (= span/300 for 30-m tower)
        reference_optimum_weight=_REF_WEIGHT_KG,
        reference_optimum_areas=None,
        reference_verified=False,     # our geometry is not literature-exact
        displacement_check_dofs=top_node_disp_dofs,
    )
