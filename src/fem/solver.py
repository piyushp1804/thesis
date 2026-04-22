"""
Linear FEM solver: apply boundary conditions, then solve  K u = F.

After assembly, the global stiffness matrix K is SINGULAR. That's not a bug —
it's physics. An unrestrained structure can drift (rigid-body motion), so
infinite displacement solutions exist for a given load. We have to "pin" some
DOFs (boundary conditions) to make the problem have a unique answer.

Strategy used here — the PARTITION method:
  1. Split all DOFs into "free" (we solve for these) and "fixed" (we know
     their displacement, usually zero for a support).
  2. Solve the reduced system K_ff u_f = F_f  (only the free rows/cols).
  3. Write the known zeros back into the full displacement vector.

Why partition vs. the "penalty" or "big number on diagonal" trick?
  - No numerical conditioning games — clean, reproducible.
  - Easy to extend to prescribed non-zero supports later (settlement, etc.).
  - Small problems: fast and stable. Big problems: still fine for our sizes
    (largest benchmark is 200 bars → ~400 DOFs — a speck for numpy).

We also compute support REACTIONS for free, using the rows of K that
correspond to the fixed DOFs:   R_fixed = K_fixed,all * u_all  -  F_fixed
This is a cheap sanity check and handy for post-processing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def solve_system(
    K: np.ndarray,
    F: np.ndarray,
    fixed_dofs: Sequence[int],
    prescribed_values: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the linear static equilibrium  K u = F  with boundary conditions.

    Parameters
    ----------
    K : ndarray of shape (ndof, ndof)
        Assembled global stiffness matrix.
    F : ndarray of shape (ndof,)
        Global force vector (external loads on every DOF).
    fixed_dofs : sequence of int
        Global DOF indices where displacement is PRESCRIBED (usually
        supports, where displacement is 0). Order doesn't matter; duplicates
        are removed.
    prescribed_values : sequence of float, optional
        The prescribed displacement at each of `fixed_dofs` (same order).
        Defaults to all zeros — the typical "pinned/roller support" case.

    Returns
    -------
    u : ndarray of shape (ndof,)
        Full displacement vector (includes the prescribed values).
    reactions : ndarray of shape (ndof,)
        Support reactions — non-zero only at the fixed DOFs; zero elsewhere.

    Raises
    ------
    ValueError
        On shape mismatches or unconstrained structures.
    numpy.linalg.LinAlgError
        If the reduced K_ff is singular (under-constrained structure, a
        mechanism, or duplicate/coincident nodes).
    """
    # ---- shape checks — catch upstream bugs early --------------------------
    K = np.asarray(K, dtype=float)
    F = np.asarray(F, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape {K.shape}.")
    if F.shape != (K.shape[0],):
        raise ValueError(
            f"F shape {F.shape} must match K size ({K.shape[0]},)."
        )

    ndof = K.shape[0]

    # ---- build the "free" / "fixed" index sets -----------------------------
    # np.unique also sorts them, which keeps the reduced system deterministic.
    fixed = np.unique(np.asarray(fixed_dofs, dtype=int))
    if fixed.size == 0:
        raise ValueError(
            "No fixed DOFs supplied. The structure would translate freely "
            "and K is singular. Pin at least enough DOFs to prevent rigid "
            "body motion (ndim for a 2D truss, or (ndim*(ndim+1))/2 in 3D)."
        )
    if fixed.min() < 0 or fixed.max() >= ndof:
        raise ValueError(
            f"fixed_dofs out of range [0, {ndof - 1}]: {fixed}."
        )

    # All DOFs not in `fixed` are free.
    all_dofs = np.arange(ndof)
    free = np.setdiff1d(all_dofs, fixed, assume_unique=False)

    # Prescribed displacement values (default: zeros)
    if prescribed_values is None:
        u_fixed = np.zeros(fixed.size, dtype=float)
    else:
        u_fixed = np.asarray(prescribed_values, dtype=float)
        if u_fixed.shape != (fixed.size,):
            raise ValueError(
                f"prescribed_values length {u_fixed.size} != fixed_dofs "
                f"length {fixed.size} (after de-duplication)."
            )

    # ---- partition K and F --------------------------------------------------
    # We pick out sub-blocks using fancy indexing:
    #   K_ff = K[free, free]    (free-free block)
    #   K_fs = K[free, fixed]   (coupling: free rows, fixed cols)
    K_ff = K[np.ix_(free, free)]
    K_fs = K[np.ix_(free, fixed)]
    F_f = F[free]

    # ---- solve the reduced system ------------------------------------------
    # Equilibrium at free DOFs:   K_ff u_f + K_fs u_s = F_f
    #   → K_ff u_f = F_f - K_fs u_s
    # For the common case u_s = 0, this simplifies to  K_ff u_f = F_f.
    rhs = F_f - K_fs @ u_fixed
    # np.linalg.solve uses LU; raises LinAlgError if K_ff is singular, which
    # tells us the structure is under-constrained (a "mechanism").
    u_free = np.linalg.solve(K_ff, rhs)

    # ---- reassemble full displacement vector -------------------------------
    u = np.zeros(ndof, dtype=float)
    u[free] = u_free
    u[fixed] = u_fixed

    # ---- compute reactions at supports -------------------------------------
    # Newton's 3rd law at supports:  R = K_all u - F_applied
    # Non-zero only at fixed DOFs (free rows re-give F_f by construction).
    reactions = np.zeros(ndof, dtype=float)
    reactions[fixed] = K[np.ix_(fixed, all_dofs)] @ u - F[fixed]

    return u, reactions


# ----------------------------------------------------------------------
# Quick sanity demo — a single horizontal bar pulled by 1 kN.
# Run with:  python -m src.fem.solver
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from .truss_element import TrussElement
    from .assembly import assemble_global_stiffness

    # One horizontal bar, length 1 m, E=200 GPa, A=1e-4 m^2.
    # Node 0 is pinned (u_0 = 0). Node 1 is pulled by F = +1000 N in x.
    # Expected answer (1-D spring):
    #     k = EA/L = 2e7 N/m
    #     u_1x = F / k = 1000 / 2e7 = 5e-5 m  (50 micrometers)
    #     Reaction at node 0, x-direction: R = -1000 N
    elem = TrussElement(0, 1, [0.0, 0.0], [1.0, 0.0], E=200e9, A=1e-4)
    K = assemble_global_stiffness(num_nodes=2, elements=[elem])

    F = np.zeros(4)
    F[2] = 1000.0  # force on node 1 in x-direction

    # Pin node 0 entirely (DOFs 0 and 1), and pin node 1's y-DOF (DOF 3) to
    # kill the rigid-body y-translation (the bar has no y-stiffness).
    u, R = solve_system(K, F, fixed_dofs=[0, 1, 3])

    with np.printoptions(precision=6, suppress=True):
        print("Displacements (m):", u)
        print(f"u_1x should be 5e-5 m. Got: {u[2]:.3e}")
        print("Reactions (N):    ", R)
        print(f"R_0x should be -1000 N. Got: {R[0]:.3e}")
