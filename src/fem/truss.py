"""
High-level `Truss` convenience class.

All the heavy lifting lives in `truss_element.py`, `assembly.py`,
`solver.py`, and `post_process.py`. This file is just a thin wrapper that
lets Day 2 / Day 3 code build and solve a truss in ~5 lines instead of 30:

    truss = Truss(nodes=nodes, bar_connectivity=bars, E=E, areas=A)
    truss.apply_load(node=3, force=(0.0, -1000.0))
    truss.fix_node(0); truss.fix_node(1)
    result = truss.solve()
    print(result.displacements, result.axial_forces)

Design choices:
  - Immutable geometry & connectivity (set at construction). This keeps the
    GA/PSO inner loop simple: we build the truss once, change only the
    areas each iteration, and re-solve.
  - Loads and supports added via explicit setters. Clearer than juggling
    raw DOF indices in client code.
  - Supports 2D and 3D via the ndim the coordinates declare.

Keep this file small and boring. Anything complex belongs one layer below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .assembly import assemble_global_stiffness
from .post_process import MemberResult, compute_member_results
from .solver import solve_system
from .truss_element import TrussElement


@dataclass
class TrussResult:
    """Everything the FEM solve produced — grouped for easy return."""

    displacements: np.ndarray            # shape (ndof,)
    reactions: np.ndarray                # shape (ndof,)  (non-zero at supports)
    member_results: list[MemberResult]   # per-bar bundle
    axial_forces: np.ndarray             # shape (n_bars,) — common shortcut


class Truss:
    """
    Assemble, solve, and post-process a pin-jointed truss.

    Parameters
    ----------
    nodes : array_like of shape (num_nodes, ndim)
        Coordinates of every node. ndim = 2 for planar, 3 for spatial trusses.
    bar_connectivity : sequence of (int, int)
        Each tuple (i, j) defines a bar between node i and node j.
    E : float or array_like of shape (n_bars,)
        Young's modulus. Scalar = same material for every bar, array = per-bar.
    areas : float or array_like of shape (n_bars,)
        Cross-section area. Scalar = uniform, array = per-bar. This is the
        design variable that GA/PSO will change during optimization.
    """

    # ---------- construction ------------------------------------------------

    def __init__(
        self,
        nodes,
        bar_connectivity: Sequence[tuple[int, int]],
        E,
        areas,
    ) -> None:
        self.nodes = np.asarray(nodes, dtype=float)
        if self.nodes.ndim != 2 or self.nodes.shape[1] not in (2, 3):
            raise ValueError(
                f"`nodes` must be (num_nodes, 2 or 3); got {self.nodes.shape}."
            )
        self.num_nodes = self.nodes.shape[0]
        self.ndim = self.nodes.shape[1]
        self.ndof = self.num_nodes * self.ndim

        # Normalize connectivity into an int array of shape (n_bars, 2).
        self.connectivity = np.asarray(bar_connectivity, dtype=int)
        if self.connectivity.ndim != 2 or self.connectivity.shape[1] != 2:
            raise ValueError(
                f"`bar_connectivity` must be (n_bars, 2); "
                f"got {self.connectivity.shape}."
            )
        self.n_bars = self.connectivity.shape[0]

        # Broadcast scalars up to per-bar arrays. Lets us treat both cases
        # uniformly from here on.
        self.E = np.broadcast_to(np.asarray(E, dtype=float), (self.n_bars,)).copy()
        self.areas = np.broadcast_to(
            np.asarray(areas, dtype=float), (self.n_bars,)
        ).copy()

        # Build the TrussElement list once (cheap; we reuse on every solve).
        self._build_elements()

        # Loads & supports start empty — user must add them.
        self._F = np.zeros(self.ndof, dtype=float)
        self._fixed: set[int] = set()

    def _build_elements(self) -> None:
        """Re-create the TrussElement list from current connectivity/areas."""
        self.elements: list[TrussElement] = []
        for k, (i, j) in enumerate(self.connectivity):
            self.elements.append(
                TrussElement(
                    node_i=int(i),
                    node_j=int(j),
                    coord_i=self.nodes[int(i)],
                    coord_j=self.nodes[int(j)],
                    E=float(self.E[k]),
                    A=float(self.areas[k]),
                )
            )

    # ---------- user-facing setters -----------------------------------------

    def set_areas(self, new_areas) -> None:
        """
        Update every bar's area and rebuild the element list.

        This is THE call the optimizer (GA/PSO) makes every iteration:
        areas = design variable, everything else stays fixed. Cheap because
        nodes/connectivity don't change — only the numbers inside the
        TrussElement objects.
        """
        new_areas = np.asarray(new_areas, dtype=float)
        if new_areas.shape != (self.n_bars,):
            raise ValueError(
                f"`new_areas` must have shape ({self.n_bars},); "
                f"got {new_areas.shape}."
            )
        self.areas = new_areas
        self._build_elements()

    def apply_load(self, node: int, force) -> None:
        """
        Add an external force vector at a node.

        Forces from repeated calls on the same node ACCUMULATE — useful when
        multiple load cases get summed, or when a node has both dead and
        live load contributions.
        """
        force = np.asarray(force, dtype=float)
        if force.shape != (self.ndim,):
            raise ValueError(
                f"`force` must have shape ({self.ndim},); got {force.shape}."
            )
        dof_start = int(node) * self.ndim
        self._F[dof_start:dof_start + self.ndim] += force

    def fix_node(self, node: int, directions: Sequence[int] | None = None) -> None:
        """
        Pin a node.

        By default, all DOFs of the node are fixed (a "pin" in 2D, a "ball
        joint" support in 3D). Pass `directions=[0]` to fix only u_x (i.e. a
        roller that permits vertical motion), or any subset [0, 1, 2].
        """
        nd = self.ndim
        if directions is None:
            directions = range(nd)
        dof_start = int(node) * nd
        for d in directions:
            if d < 0 or d >= nd:
                raise ValueError(
                    f"direction {d} invalid for ndim={nd}. Use 0..{nd - 1}."
                )
            self._fixed.add(dof_start + d)

    # ---------- main solve --------------------------------------------------

    def solve(self) -> TrussResult:
        """
        Assemble K, apply BCs, solve K u = F, and post-process member forces.

        Returns a TrussResult bundle.
        """
        K = assemble_global_stiffness(
            num_nodes=self.num_nodes,
            elements=self.elements,
            ndim=self.ndim,
        )
        u, reactions = solve_system(K, self._F, fixed_dofs=sorted(self._fixed))
        member = compute_member_results(self.elements, u)
        axial = np.array([m.axial_force for m in member], dtype=float)
        return TrussResult(
            displacements=u,
            reactions=reactions,
            member_results=member,
            axial_forces=axial,
        )

    # ---------- useful derived quantities -----------------------------------

    def bar_lengths(self) -> np.ndarray:
        """Length of every bar (cached in each element; grab them as an array)."""
        return np.array([el.length for el in self.elements], dtype=float)

    def total_volume(self) -> float:
        """Sum of area*length over all bars — proxy for material usage."""
        return float(np.sum(self.areas * self.bar_lengths()))

    def total_weight(self, density: float) -> float:
        """
        Structural weight = density * volume.

        For steel, pass density in consistent units (e.g. 7850 kg/m^3 in SI,
        or 0.283 lb/in^3 in imperial for the published 10-bar benchmark).
        """
        if density <= 0:
            raise ValueError("density must be positive.")
        return density * self.total_volume()
