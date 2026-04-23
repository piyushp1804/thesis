"""
`SurrogateEvaluator` — a drop-in replacement for `BenchmarkProblem.evaluate`.

The GA / PSO / NSGA-II already call an evaluator through `TrussProblem`.
Swap one in:

    s = load_surrogate("results/10bar_surrogate.pt")
    evaluator = SurrogateEvaluator(benchmark, s)
    run("ga", benchmark, evaluator=evaluator, ...)

The surrogate returns weight, max_stress, max_disp — enough for the
optimizer. Per-bar stresses/axial-forces are NOT predicted by the
default model; we return zeros there, which is fine because the
optimizer only uses the scalar bounds.
"""

from __future__ import annotations

import numpy as np
import torch

from src.benchmarks.base import BenchmarkEvaluation, BenchmarkProblem

from .train import NormStats, TrainedSurrogate, _apply


class SurrogateEvaluator:
    """Callable returning `BenchmarkEvaluation` using the MLP."""

    def __init__(
        self,
        benchmark: BenchmarkProblem,
        surrogate: TrainedSurrogate,
        device: str = "cpu",
    ) -> None:
        self.benchmark = benchmark
        self.s = surrogate
        self.device = device
        self.s.model.to(device)
        self.s.model.eval()
        self._zeros_bars = np.zeros(benchmark.n_bars, dtype=float)

    def __call__(self, x: np.ndarray) -> BenchmarkEvaluation:
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        x_std = _apply(x, self.s.x_stats).astype(np.float32)
        with torch.no_grad():
            pred_std = (
                self.s.model(torch.from_numpy(x_std).to(self.device))
                .cpu()
                .numpy()[0]
            )
        pred_log = pred_std * self.s.y_stats.std + self.s.y_stats.mean
        pred = np.expm1(pred_log)
        pred = np.clip(pred, 0.0, None)  # guard against rare negatives

        # expand to per-bar areas for downstream reporting only
        areas = self.benchmark.expand_design(x.ravel())

        return BenchmarkEvaluation(
            weight=float(pred[0]),
            max_abs_stress=float(pred[1]),
            max_abs_displacement=float(pred[2]),
            member_abs_stresses=self._zeros_bars.copy(),
            member_abs_axial_forces=self._zeros_bars.copy(),
            per_bar_areas=areas,
        )
