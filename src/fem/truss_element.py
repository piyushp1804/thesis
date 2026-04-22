"""
Truss element (one steel bar) for the finite element method.

A truss bar is the simplest structural element:
  - it carries ONLY axial force (pure tension or pure compression)
  - it does NOT resist bending or shear
  - it connects exactly two nodes (pin-jointed at both ends)

This module computes, for one bar, the stiffness matrix that tells us:
  "given small displacements of the two end nodes, what forces appear at the nodes?"

Design choice: this class is dimension-agnostic — works for 2D (planar) and
3D (spatial) trusses identically. That matters because Day 1 validates a 2D
3-bar truss, but Day 2 will need the 25-bar and 72-bar 3D spatial trusses.
Reusing one clean formulation avoids duplicate code and duplicate bugs.

Theory (Hooke's Law applied to a bar):
    axial_force  = (E * A / L) * elongation
    elongation   = (u_j - u_i) . n_hat         (displacement along bar axis)
    n_hat        = (x_j - x_i) / L             (unit vector from node i to j)

The element stiffness matrix in global coordinates comes out as:
    k_e = (E * A / L) * B^T B
where B = [-n_hat, +n_hat]  (a row of length 2*ndim).
Size of k_e: (2*ndim, 2*ndim) — so 4x4 in 2D, 6x6 in 3D.
"""

from __future__ import annotations

import numpy as np


class TrussElement:
    """
    One truss bar connecting two nodes.

    Parameters
    ----------
    node_i, node_j : int
        Global node indices this bar connects. Must be different.
    coord_i, coord_j : array_like of shape (ndim,)
        Global coordinates of the two end nodes. ndim = 2 or 3.
    E : float
        Young's modulus of the bar material (Pa).
    A : float
        Cross-sectional area of the bar (m^2).

    Attributes
    ----------
    ndim : int          -- spatial dimension inferred from the coordinates
    length : float      -- bar length, cached
    direction : ndarray -- unit vector from node_i to node_j, cached
    """

    def __init__(
        self,
        node_i: int,
        node_j: int,
        coord_i,
        coord_j,
        E: float,
        A: float,
    ) -> None:
        # Store node indices — used later to map this element's DOFs into the
        # global stiffness matrix.
        if node_i == node_j:
            raise ValueError("A truss bar needs two DIFFERENT nodes.")
        self.node_i = int(node_i)
        self.node_j = int(node_j)

        # Store coordinates as float arrays so numpy math is clean.
        self.coord_i = np.asarray(coord_i, dtype=float)
        self.coord_j = np.asarray(coord_j, dtype=float)

        # Both coordinate arrays must have the same dimension (2D or 3D).
        if self.coord_i.shape != self.coord_j.shape:
            raise ValueError(
                f"coord_i shape {self.coord_i.shape} != coord_j shape "
                f"{self.coord_j.shape}. Use the same dimension."
            )
        if self.coord_i.ndim != 1 or self.coord_i.shape[0] not in (2, 3):
            raise ValueError("Coordinates must be 1-D arrays of length 2 or 3.")

        # Material & section properties. Guard against silent nonsense.
        if E <= 0:
            raise ValueError("Young's modulus E must be positive.")
        if A <= 0:
            raise ValueError("Cross-section area A must be positive.")
        self.E = float(E)
        self.A = float(A)

        # Spatial dimension (2 for planar truss, 3 for spatial truss).
        self.ndim = self.coord_i.shape[0]

        # Compute & cache length and unit direction vector.
        # These are used by every other method, so doing this once is cheaper
        # and also lets us fail fast if two nodes are at the same position.
        delta = self.coord_j - self.coord_i
        self.length = float(np.linalg.norm(delta))
        if self.length == 0.0:
            raise ValueError(
                f"Bar connects nodes at identical positions {self.coord_i}. "
                "Check your geometry."
            )
        # Unit vector pointing from node i to node j. This is what the
        # textbook calls the "direction cosines" of the bar.
        self.direction = delta / self.length

    # ------------------------------------------------------------------
    # Geometric helpers (useful for plotting and debugging)
    # ------------------------------------------------------------------

    def direction_cosines(self) -> np.ndarray:
        """
        Return the unit direction vector (a.k.a. direction cosines) of the bar.

        In 2D this is [cos(theta), sin(theta)].
        In 3D this is [l, m, n] — the standard direction cosines.
        """
        return self.direction.copy()

    # ------------------------------------------------------------------
    # Stiffness matrix (the core FEM quantity)
    # ------------------------------------------------------------------

    def global_stiffness(self) -> np.ndarray:
        """
        Element stiffness matrix k_e in GLOBAL coordinates.

        Formula:  k_e = (E * A / L) * B^T B
        where     B  = [-n_hat , +n_hat]   (1 x 2*ndim row vector)

        Shape: (2*ndim, 2*ndim) — that is 4x4 for 2D, 6x6 for 3D.

        DOF ordering within the returned matrix follows the element's local
        convention:
            [ u_i_x, u_i_y, (u_i_z),  u_j_x, u_j_y, (u_j_z) ]
        """
        n = self.direction  # shape (ndim,)

        # Build B = [-n, n] as a 1-row matrix of length 2*ndim.
        # np.concatenate keeps both 2D and 3D cases identical.
        B = np.concatenate([-n, n]).reshape(1, -1)  # (1, 2*ndim)

        # Axial stiffness scalar, k = E * A / L
        # (This is the classic 1-D Hooke's-law stiffness of a bar.)
        k_axial = self.E * self.A / self.length

        # The global-coordinate stiffness is (EA/L) * B^T B.
        # B^T B is rank-1: it has one non-zero eigenvalue (the axial mode)
        # which is physically correct — a bar only resists axial stretch.
        k_global = k_axial * (B.T @ B)
        return k_global

    # ------------------------------------------------------------------
    # DOF indices (how this element plugs into the global system)
    # ------------------------------------------------------------------

    def dof_indices(self) -> np.ndarray:
        """
        Return the global DOF indices this element contributes to.

        Convention: node k owns DOFs [k*ndim, k*ndim+1, ..., k*ndim+ndim-1].
        So this element touches 2*ndim DOFs in total — the DOFs of node_i
        followed by the DOFs of node_j.

        Example (2D, node_i=0, node_j=2): returns [0, 1, 4, 5].
        """
        nd = self.ndim
        # DOFs owned by node_i: nd consecutive integers starting at node_i * nd
        dofs_i = np.arange(self.node_i * nd, self.node_i * nd + nd)
        dofs_j = np.arange(self.node_j * nd, self.node_j * nd + nd)
        return np.concatenate([dofs_i, dofs_j])


# ----------------------------------------------------------------------
# Quick sanity demo — run `python -m src.fem.truss_element`
# Prints the 4x4 stiffness of a horizontal bar; easy to check by eye.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Horizontal bar, length 1 m, E = 200 GPa, A = 1e-4 m^2
    #   k_axial = E*A/L = 200e9 * 1e-4 / 1 = 2e7 N/m
    # Expected 4x4 stiffness (horizontal bar, sin=0, cos=1):
    #   [[ 2e7, 0, -2e7, 0],
    #    [   0, 0,    0, 0],
    #    [-2e7, 0,  2e7, 0],
    #    [   0, 0,    0, 0]]
    elem = TrussElement(0, 1, [0.0, 0.0], [1.0, 0.0], E=200e9, A=1e-4)
    with np.printoptions(precision=3, suppress=True):
        print("Length:", elem.length)
        print("Direction cosines:", elem.direction_cosines())
        print("DOF indices:", elem.dof_indices())
        print("Global stiffness matrix (N/m):")
        print(elem.global_stiffness())
