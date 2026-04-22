"""
Post-processing: turn the raw displacement vector `u` into quantities that
engineers actually look at — member elongations, strains, stresses, and
axial forces in each bar.

For a truss bar connecting node i -> node j, once we know `u`:

    elongation  = (u_j - u_i) . n_hat                 [m]
    strain      = elongation / L                      [dimensionless]
    stress      = E * strain                          [Pa]
    axial_force = stress * A  =  (EA/L) * elongation  [N]

Sign convention used everywhere:
    positive  ->  TENSION   (bar is stretched, pulls the nodes together)
    negative  ->  COMPRESSION (bar is squished, pushes the nodes apart)

This convention matches pymoo's constraint formulation (g(x) <= 0) later:
when we add IS-800 checks on Day 6, tension yield and compression buckling
both compare the *signed* force to its allowable limit in the correct sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .truss_element import TrussElement


@dataclass
class MemberResult:
    """Per-bar result bundle. Small dataclass keeps the test code readable."""

    elongation: float   # change in bar length (m). + = longer (tension).
    strain: float       # dimensionless strain (elongation / L).
    stress: float       # axial stress (Pa). + = tension, - = compression.
    axial_force: float  # axial force (N).   + = tension, - = compression.


def compute_member_results(
    elements: Sequence[TrussElement],
    displacements: np.ndarray,
) -> list[MemberResult]:
    """
    Compute elongation, strain, stress, and axial force for every bar.

    Parameters
    ----------
    elements : sequence of TrussElement
        The same list used during assembly.
    displacements : ndarray of shape (ndof,)
        Full nodal displacement vector from `solver.solve_system`.

    Returns
    -------
    results : list of MemberResult
        One entry per element, in the same order as `elements`.
    """
    u = np.asarray(displacements, dtype=float)
    if u.ndim != 1:
        raise ValueError(f"displacements must be 1-D, got shape {u.shape}.")

    results: list[MemberResult] = []
    for idx, el in enumerate(elements):
        # Pick out the nodal displacements of THIS element's two endpoints.
        # Using el.dof_indices() keeps the node -> DOF mapping in one place.
        dofs = el.dof_indices()                    # length 2*ndim
        if dofs.max() >= u.size:
            raise ValueError(
                f"Element {idx} references DOF {dofs.max()} but "
                f"displacement vector has only {u.size} entries."
            )
        u_elem = u[dofs]                           # [u_i, u_j] stacked

        # Split into node-i and node-j displacement sub-vectors (each ndim).
        nd = el.ndim
        u_i = u_elem[:nd]
        u_j = u_elem[nd:]

        # Elongation = (u_j - u_i) . n_hat.
        # Only the axial component of the relative displacement matters —
        # transverse components don't stretch the bar (trusses have no
        # bending stiffness).
        delta_u = u_j - u_i
        elongation = float(np.dot(delta_u, el.direction))

        strain = elongation / el.length
        stress = el.E * strain
        # Axial force = stress * area = (E A / L) * elongation.
        # Equivalent to the 1-D spring law  F = k * x.
        axial_force = stress * el.A

        results.append(
            MemberResult(
                elongation=elongation,
                strain=strain,
                stress=stress,
                axial_force=axial_force,
            )
        )

    return results


def member_forces(
    elements: Sequence[TrussElement],
    displacements: np.ndarray,
) -> np.ndarray:
    """
    Convenience wrapper: return only the axial forces as a 1-D numpy array.

    Handy because optimization constraints usually care about forces, not
    the full MemberResult bundle. Keeps the hot inner loop of the GA/PSO
    calls free from dataclass overhead.
    """
    return np.array(
        [r.axial_force for r in compute_member_results(elements, displacements)],
        dtype=float,
    )


# ----------------------------------------------------------------------
# Quick sanity demo — same single-bar case as solver.py.
# Run with:  python -m src.fem.post_process
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from .assembly import assemble_global_stiffness
    from .solver import solve_system

    # Same setup as solver demo: horizontal bar pulled with 1000 N at node 1.
    # Expected: bar is in TENSION, axial force = +1000 N, strain = 5e-5,
    # stress = E * strain = 200e9 * 5e-5 = 1e7 Pa (= 10 MPa).
    elem = TrussElement(0, 1, [0.0, 0.0], [1.0, 0.0], E=200e9, A=1e-4)
    K = assemble_global_stiffness(num_nodes=2, elements=[elem])
    F = np.zeros(4)
    F[2] = 1000.0
    u, _ = solve_system(K, F, fixed_dofs=[0, 1, 3])

    (r,) = compute_member_results([elem], u)
    print(f"Elongation : {r.elongation:.3e} m   (expect  5.0e-05)")
    print(f"Strain     : {r.strain:.3e}         (expect  5.0e-05)")
    print(f"Stress     : {r.stress:.3e} Pa      (expect  1.0e+07)")
    print(f"Axial force: {r.axial_force:.3e} N  (expect  1.0e+03, TENSION)")
