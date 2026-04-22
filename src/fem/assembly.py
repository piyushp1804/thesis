"""
Global stiffness assembly for a pin-jointed truss.

Given a list of TrussElement objects (each one a bar with its own small
stiffness matrix), we need to combine them into ONE big matrix K that
describes the whole structure. This process is called "assembly".

The idea in one picture:

                        (global DOFs)
                        0   1   2   3   4   5   ...
                      +---+---+---+---+---+---+
      bar 1 (0-1):    | + | + | + | + |   |   |    k_e1 goes into rows/cols 0,1,2,3
      bar 2 (1-2):    |   |   | + | + | + | + |    k_e2 goes into rows/cols 2,3,4,5
                      +---+---+---+---+---+---+
                      =  one summed global K (ndof x ndof)

We literally sum element contributions into the right slots of K.
That's all assembly is. Python slicing with `np.ix_` does the slot-picking
in one line.

The resulting K is:
  - symmetric (physics: reciprocity of forces and displacements)
  - sparse (each bar only touches 2 nodes)
  - SINGULAR before we apply boundary conditions — the structure can still
    translate/rotate rigidly. Fixing some DOFs later removes the singularity.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .truss_element import TrussElement


def assemble_global_stiffness(
    num_nodes: int,
    elements: Sequence[TrussElement],
    ndim: int | None = None,
) -> np.ndarray:
    """
    Build the global stiffness matrix K by summing element contributions.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the truss. Defines the size of K.
    elements : sequence of TrussElement
        All bars in the truss. All elements must share the same ndim.
    ndim : int, optional
        Spatial dimension (2 or 3). If None, it's inferred from the first
        element. Passing it explicitly lets us validate empty element lists
        and catch mixed-dimension mistakes.

    Returns
    -------
    K : ndarray of shape (num_nodes * ndim, num_nodes * ndim)
        The assembled global stiffness matrix (before boundary conditions).
        Symmetric; singular until BCs are applied.

    Raises
    ------
    ValueError
        If elements disagree on ndim, or node indices exceed num_nodes.
    """
    # ------ input validation -------------------------------------------------
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be > 0, got {num_nodes}")
    if len(elements) == 0 and ndim is None:
        raise ValueError(
            "Cannot infer ndim from an empty element list. "
            "Pass ndim=2 or ndim=3 explicitly."
        )

    # Decide spatial dimension. If the caller didn't specify, use the first
    # element's dimension. Then demand that every other element matches.
    if ndim is None:
        ndim = elements[0].ndim
    if ndim not in (2, 3):
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    for idx, el in enumerate(elements):
        if el.ndim != ndim:
            raise ValueError(
                f"Element {idx} has ndim={el.ndim} but assembly expected "
                f"ndim={ndim}. All elements must share the same dimension."
            )
        # Guard against out-of-range node indices that would silently write
        # into bogus slots and produce weird results later.
        for n in (el.node_i, el.node_j):
            if n < 0 or n >= num_nodes:
                raise ValueError(
                    f"Element {idx} references node index {n}, but "
                    f"num_nodes = {num_nodes}. Valid range: 0..{num_nodes - 1}."
                )

    # ------ allocate global K ------------------------------------------------
    # Size: total DOFs = num_nodes * ndim (each node has ndim translation DOFs)
    ndof = num_nodes * ndim
    K = np.zeros((ndof, ndof), dtype=float)

    # ------ scatter-add each element's stiffness into K ----------------------
    # This is the whole assembly process. For each bar:
    #   1. get its small (2*ndim) x (2*ndim) stiffness matrix
    #   2. get the global DOF indices it occupies
    #   3. ADD the small matrix into the right slots of K (note: "add", not
    #      "overwrite" — two bars sharing a node both contribute to the same
    #      diagonal block, and those contributions must SUM).
    for el in elements:
        k_e = el.global_stiffness()          # (2*ndim, 2*ndim)
        dofs = el.dof_indices()              # length 2*ndim
        # np.ix_ builds a row/column index grid so K[ix] addresses the exact
        # (2*ndim)x(2*ndim) submatrix to add into. Equivalent to a double
        # Python loop but vectorized and far faster.
        K[np.ix_(dofs, dofs)] += k_e

    return K


# ----------------------------------------------------------------------
# Quick sanity demo — two bars sharing a node.
# Run with:  python -m src.fem.assembly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Two horizontal bars in series along the x-axis:
    #   node 0 ---- node 1 ---- node 2
    #   each bar identical: E = 200 GPa, A = 1e-4 m^2, L = 1 m
    #   so each bar's axial stiffness is k = EA/L = 2e7 N/m
    #
    # We expect the global K (4 DOFs in 2D: [u0x,u0y,u1x,u1y,u2x,u2y]) to
    # have the classical "spring in series" pattern in the u_x rows:
    #   K_xx =  [[ k, -k, 0 ],
    #            [-k, 2k, -k],
    #            [ 0, -k,  k]]
    # because node 1 is shared by both bars and so its diagonal sums.
    e1 = TrussElement(0, 1, [0.0, 0.0], [1.0, 0.0], E=200e9, A=1e-4)
    e2 = TrussElement(1, 2, [1.0, 0.0], [2.0, 0.0], E=200e9, A=1e-4)

    K = assemble_global_stiffness(num_nodes=3, elements=[e1, e2])
    with np.printoptions(precision=3, suppress=True):
        print(f"Global K shape: {K.shape}  (expected 6x6 = 3 nodes * 2 dof)")
        print(K)
        # Middle-node x-diagonal should be 2k = 4e7:
        print(f"\nK[2,2] (u1_x diagonal, should be 2k = 4e7): {K[2, 2]:.3e}")
