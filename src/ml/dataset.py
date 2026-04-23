"""
Latin-Hypercube dataset generator for a benchmark's design space.

Given a `BenchmarkProblem`, draws `n_samples` design vectors uniformly
from `area_bounds` via Latin-Hypercube Sampling, evaluates each with
the FEM, and returns aligned arrays:

    X : (n_samples, n_design_vars)  -- design variables
    y : dict of (n_samples,) targets:
        "weight"        -- structural weight
        "max_stress"    -- worst absolute member stress across LCs
        "max_disp"      -- worst absolute displacement across LCs

Plus `member_stress` (n_samples, n_bars) and
`member_axial`   (n_samples, n_bars) for downstream feasibility work.

Saves to .npz. Progress bar via tqdm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

from src.benchmarks.base import BenchmarkProblem


@dataclass
class SurrogateDataset:
    X: np.ndarray                  # (N, n_design_vars)
    weight: np.ndarray             # (N,)
    max_stress: np.ndarray         # (N,)
    max_disp: np.ndarray           # (N,)
    member_stress: np.ndarray      # (N, n_bars) abs values
    member_axial: np.ndarray       # (N, n_bars) abs values
    bounds: tuple[float, float]
    benchmark_name: str

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            X=self.X,
            weight=self.weight,
            max_stress=self.max_stress,
            max_disp=self.max_disp,
            member_stress=self.member_stress,
            member_axial=self.member_axial,
            bounds=np.asarray(self.bounds),
            benchmark_name=np.asarray(self.benchmark_name),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SurrogateDataset":
        data = np.load(path, allow_pickle=False)
        return cls(
            X=data["X"],
            weight=data["weight"],
            max_stress=data["max_stress"],
            max_disp=data["max_disp"],
            member_stress=data["member_stress"],
            member_axial=data["member_axial"],
            bounds=tuple(data["bounds"].tolist()),
            benchmark_name=str(data["benchmark_name"]),
        )


def generate_dataset(
    benchmark: BenchmarkProblem,
    n_samples: int,
    seed: int = 0,
    progress: bool = True,
) -> SurrogateDataset:
    """Build an LHS dataset for `benchmark`."""
    lo, hi = benchmark.area_bounds
    rng = np.random.default_rng(seed)

    sampler = qmc.LatinHypercube(d=benchmark.n_design_vars, seed=rng)
    unit = sampler.random(n=n_samples)
    X = lo + unit * (hi - lo)

    weights = np.zeros(n_samples)
    stresses = np.zeros(n_samples)
    disps = np.zeros(n_samples)
    mem_stress = np.zeros((n_samples, benchmark.n_bars))
    mem_axial = np.zeros((n_samples, benchmark.n_bars))

    it = tqdm(range(n_samples), desc="LHS-FEM") if progress else range(n_samples)
    for i in it:
        ev = benchmark.evaluate(X[i])
        weights[i] = ev.weight
        stresses[i] = ev.max_abs_stress
        disps[i] = ev.max_abs_displacement
        mem_stress[i] = ev.member_abs_stresses
        mem_axial[i] = ev.member_abs_axial_forces

    return SurrogateDataset(
        X=X,
        weight=weights,
        max_stress=stresses,
        max_disp=disps,
        member_stress=mem_stress,
        member_axial=mem_axial,
        bounds=(lo, hi),
        benchmark_name=benchmark.name,
    )
