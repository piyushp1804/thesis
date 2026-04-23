"""
Benchmark problem base classes + FEM-backed evaluator.

A `BenchmarkProblem` is a pure data bundle: nodes, bars, loads, supports,
material constants, symmetry grouping, constraint limits, and the
published reference optimum from the literature. Every algorithm
(GA/PSO/NSGA-II/surrogate/RL/LLM) consumes benchmarks through this one
interface — which is what makes the comparison in Chapter 4 apples-to-
apples.

`evaluate(x)` is the single hot path: it takes a design-variable vector
`x` (one area per symmetry group), expands it to per-bar areas, runs the
FEM solver once per load case, and returns the worst-case weight /
stress / displacement bundle the constraint layer needs.

Kept intentionally small and boring. Optimizer-specific massaging (fix-
deterministic seeding, penalty shaping, surrogate swap-in) lives in
`src/algorithms/problem.py`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.fem.truss import Truss


# ---------------------------------------------------------------------------
# Light-weight data carriers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadCase:
    """One load-case description.

    Attributes
    ----------
    nodal_forces :
        Mapping {node_index -> force vector of shape (ndim,)}.
    name :
        Human-readable label (only used for logs & plots).
    """

    nodal_forces: dict[int, np.ndarray]
    name: str = ""


@dataclass
class BenchmarkEvaluation:
    """What `BenchmarkProblem.evaluate` returns.

    `weight` is the objective; the rest feed into the constraint layer.
    """

    weight: float
    max_abs_stress: float
    max_abs_displacement: float
    member_abs_stresses: np.ndarray      # shape (n_bars,), worst across load cases
    member_abs_axial_forces: np.ndarray  # same
    per_bar_areas: np.ndarray            # shape (n_bars,), after group expansion


@dataclass
class BenchmarkProblem:
    """Full problem definition: geometry + loads + constraints + reference."""

    # identity
    name: str
    reference_source: str

    # geometry & topology
    nodes: np.ndarray                               # (num_nodes, ndim)
    connectivity: np.ndarray                        # (n_bars, 2) int

    # material
    E: float
    density: float
    units: str                                      # "imperial" or "SI"

    # boundary conditions
    # Each entry pins one node. `directions=None` means fix all DOFs of
    # that node; otherwise a tuple of direction indices (0=x, 1=y, 2=z).
    supports: list[tuple[int, tuple[int, ...] | None]]
    load_cases: list[LoadCase]

    # optimization encoding
    # group_map[i] = list of element indices that share design variable i.
    # For a benchmark with no symmetry, this is [[0], [1], ..., [n-1]].
    group_map: list[list[int]]
    area_bounds: tuple[float, float]

    # constraints
    stress_limit_tension: float       # positive; tensile stress < this
    stress_limit_compression: float   # positive; |compressive stress| < this
    displacement_limit: float         # max |displacement| at any free DOF

    # literature reference
    reference_optimum_weight: float
    reference_optimum_areas: np.ndarray | None = None  # per-bar, optional
    reference_verified: bool = False  # True => Phase-1 gate tests enforce

    # pre-computed derived quantities (filled in __post_init__)
    _bar_lengths: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Defensive copies + array normalization.
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.connectivity = np.asarray(self.connectivity, dtype=int)
        if self.nodes.ndim != 2 or self.nodes.shape[1] not in (2, 3):
            raise ValueError(f"nodes must be (N, 2|3); got {self.nodes.shape}")
        if self.connectivity.ndim != 2 or self.connectivity.shape[1] != 2:
            raise ValueError(
                f"connectivity must be (n_bars, 2); got {self.connectivity.shape}"
            )
        # Cache bar lengths once — they never change during optimization.
        d = self.nodes[self.connectivity[:, 1]] - self.nodes[self.connectivity[:, 0]]
        self._bar_lengths = np.linalg.norm(d, axis=1)
        # Sanity: every bar index mentioned in group_map must be valid.
        all_idx = [i for grp in self.group_map for i in grp]
        if sorted(all_idx) != list(range(self.n_bars)):
            raise ValueError(
                f"group_map must cover every element exactly once; "
                f"got {len(all_idx)} entries for {self.n_bars} bars."
            )

    # ----- convenience properties -----

    @property
    def n_bars(self) -> int:
        return self.connectivity.shape[0]

    @property
    def n_design_vars(self) -> int:
        return len(self.group_map)

    @property
    def ndim(self) -> int:
        return self.nodes.shape[1]

    # ----- encoding helpers -----

    def expand_design(self, x: np.ndarray) -> np.ndarray:
        """Map design vector `x` (n_design_vars,) -> per-bar areas (n_bars,)."""
        x = np.asarray(x, dtype=float).ravel()
        if x.shape != (self.n_design_vars,):
            raise ValueError(
                f"expected design vector of length {self.n_design_vars}, "
                f"got shape {x.shape}."
            )
        areas = np.empty(self.n_bars, dtype=float)
        for g, elems in enumerate(self.group_map):
            areas[elems] = x[g]
        return areas

    def initial_uniform_design(self, area: float | None = None) -> np.ndarray:
        """Return a feasible starting point — mid-of-bounds by default."""
        if area is None:
            lo, hi = self.area_bounds
            area = 0.5 * (lo + hi)
        return np.full(self.n_design_vars, float(area))

    # ----- the hot-path evaluator -----

    def evaluate(self, x: np.ndarray) -> BenchmarkEvaluation:
        """Run FEM once per load case, return the weight + worst-case state."""
        areas = self.expand_design(x)

        worst_stress = 0.0
        worst_disp = 0.0
        member_abs_stresses = np.zeros(self.n_bars, dtype=float)
        member_abs_axial = np.zeros(self.n_bars, dtype=float)

        for lc in self.load_cases:
            truss = Truss(
                nodes=self.nodes,
                bar_connectivity=self.connectivity,
                E=self.E,
                areas=areas,
            )
            for node, dirs in self.supports:
                truss.fix_node(node, dirs)
            for node, force in lc.nodal_forces.items():
                truss.apply_load(node=node, force=force)
            result = truss.solve()

            s = np.array(
                [m.stress for m in result.member_results], dtype=float
            )
            member_abs_stresses = np.maximum(member_abs_stresses, np.abs(s))
            member_abs_axial = np.maximum(
                member_abs_axial, np.abs(result.axial_forces)
            )
            worst_stress = max(worst_stress, float(np.max(np.abs(s))))
            worst_disp = max(
                worst_disp, float(np.max(np.abs(result.displacements)))
            )

        weight = float(self.density * np.sum(areas * self._bar_lengths))

        return BenchmarkEvaluation(
            weight=weight,
            max_abs_stress=worst_stress,
            max_abs_displacement=worst_disp,
            member_abs_stresses=member_abs_stresses,
            member_abs_axial_forces=member_abs_axial,
            per_bar_areas=areas,
        )
