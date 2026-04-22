"""
End-to-end demo of the Day 1 FEM pipeline.

Run:
    python scripts/fem_demo.py

What it does (top to bottom):
  1. Builds the canonical 3-bar truss (same one the test suite validates).
  2. Solves it.
  3. Prints displacements, reactions, and member forces in a readable table.
  4. Prints the hand-calculated "expected" values side-by-side for a quick
     visual sanity check.

This script is not used anywhere else — it's purely a demo / smoke test that
lets you eyeball the numbers to confirm the FEM layer is healthy before
moving on to Day 2 (benchmark trusses).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the project root to sys.path so `src.fem.*` imports work when this
# script is launched directly (e.g. `python scripts/fem_demo.py`).
# pytest uses pyproject.toml's pythonpath and doesn't need this shim.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.fem.truss import Truss


def build_three_bar_truss() -> Truss:
    """Same geometry as tests/test_fem_3bar.py. See that file for the picture."""
    nodes = np.array(
        [
            [0.0, 0.0],   # 0
            [1.0, 0.0],   # 1
            [2.0, 0.0],   # 2
            [1.0, 1.0],   # 3  (apex, loaded)
        ]
    )
    bars = [(0, 3), (1, 3), (2, 3)]
    truss = Truss(nodes=nodes, bar_connectivity=bars, E=200e9, areas=1e-4)
    truss.fix_node(0)
    truss.fix_node(1)
    truss.fix_node(2)
    truss.apply_load(node=3, force=(0.0, -1000.0))
    return truss


def main() -> None:
    truss = build_three_bar_truss()
    result = truss.solve()

    # Closed-form expectations (see test_fem_3bar.py docstring).
    expected_u3y = -1000.0 / (2e7 * (1.0 + 1.0 / np.sqrt(2.0)))
    expected_n_vert = 2e7 * expected_u3y
    expected_n_diag = (2e7 / 2.0) * expected_u3y

    print("=" * 68)
    print("Day 1 FEM Demo  —  Canonical 3-Bar Truss")
    print("=" * 68)

    # -- Geometry summary --------------------------------------------------
    print(f"\nNodes         : {truss.num_nodes}   Bars : {truss.n_bars}   "
          f"DOFs : {truss.ndof}")
    print(f"Total volume  : {truss.total_volume():.4e} m^3")
    print(f"Total weight  : {truss.total_weight(7850.0):.2f} N  (steel)")

    # -- Node displacements ------------------------------------------------
    print("\n--- Node displacements (m) -----------------------------------")
    u = result.displacements.reshape(-1, truss.ndim)
    for i, disp in enumerate(u):
        print(f"  node {i}: u = {disp}")
    print(f"\n  node 3 u_y computed  : {u[3, 1]:.6e} m")
    print(f"  node 3 u_y expected  : {expected_u3y:.6e} m")
    print(f"  absolute error       : {abs(u[3, 1] - expected_u3y):.2e}")

    # -- Member (axial) forces --------------------------------------------
    print("\n--- Member axial forces (N, + = tension, - = compression) ----")
    labels = ["bar 0-3 (diag L)", "bar 1-3 (vertical)", "bar 2-3 (diag R)"]
    expected = [expected_n_diag, expected_n_vert, expected_n_diag]
    for name, n_got, n_exp in zip(labels, result.axial_forces, expected):
        print(f"  {name:<20}  got = {n_got:>10.3f}   expected = {n_exp:>10.3f}")

    # -- Support reactions -------------------------------------------------
    print("\n--- Support reactions (N) ------------------------------------")
    R = result.reactions.reshape(-1, truss.ndim)
    for i, r in enumerate(R):
        # Skip the free node's "reaction" (it's zero by construction).
        if np.allclose(r, 0.0):
            continue
        print(f"  node {i}: R = {r}")
    # Quick equilibrium check: sum of reactions + applied load ≈ 0.
    applied = np.array([0.0, -1000.0])
    sum_R = R.sum(axis=0)
    print(f"\n  sum(R) + applied = {sum_R + applied}  (should be ~ 0)")

    print("\n" + "=" * 68)
    print("If the numbers above match, the FEM layer is good. Day 1 done.")
    print("=" * 68)


if __name__ == "__main__":
    main()
