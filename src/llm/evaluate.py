"""
LLM-warmstart vs random-init GA experiment.

Phase 4 thesis claim:
    'LLM-initialized GA reaches the literature optimum in fewer
    generations than random-init GA across seeds.'

We measure 'generations to 90%-convergence', where we define 90%-
convergence as the first generation at which the best-so-far weight
descends below (W_rand_final + 0.1 * (W_init_mean - W_rand_final)).
Reported reduction is mean(gens_rand - gens_llm) / mean(gens_rand).

We also run a paired Wilcoxon signed-rank test on the per-seed gen
counts and report the p-value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.algorithms.runner import run
from src.benchmarks.base import BenchmarkProblem

from .designer import suggest_initial_design


@dataclass
class WarmstartReport:
    seeds: list[int]
    gens_random: list[int]
    gens_llm: list[int]
    final_random: list[float]
    final_llm: list[float]
    pct_reduction: float
    p_value: float


def _gens_to_converge(history: list[dict], target: float) -> int:
    """First generation whose best-so-far <= target.

    If the condition is already satisfied at gen 0 (e.g., a lucky LHS
    sample), we return 0. If never satisfied, we return the last gen
    index as a ceiling.
    """
    for h in history:
        if h["best_weight"] <= target:
            return int(h["gen"])
    return int(history[-1]["gen"]) if history else 0


def compare_warmstart_vs_random(
    benchmark: BenchmarkProblem,
    seeds: list[int],
    pop_size: int = 100,
    n_gen: int = 500,
    target_pct_over_lit: float = 2.0,
) -> WarmstartReport:
    """Run GA with and without LLM warm-start across `seeds`.

    Convergence target: `literature_optimum * (1 + target_pct_over_lit/100)`.
    This is a problem-anchored, fair metric — both arms must close the
    same absolute gap.
    """
    suggestion = suggest_initial_design(benchmark)
    x0 = suggestion.x

    target = benchmark.reference_optimum_weight * (
        1.0 + target_pct_over_lit / 100.0
    )

    gens_rand: list[int] = []
    gens_llm: list[int] = []
    final_rand: list[float] = []
    final_llm: list[float] = []

    for seed in seeds:
        r_rand = run("ga", benchmark, seed=seed, pop_size=pop_size, n_gen=n_gen)
        r_llm = run(
            "ga", benchmark, seed=seed, pop_size=pop_size, n_gen=n_gen, x0=x0
        )
        final_rand.append(r_rand.best_weight)
        final_llm.append(r_llm.best_weight)
        gens_rand.append(_gens_to_converge(r_rand.history, target))
        gens_llm.append(_gens_to_converge(r_llm.history, target))

    gens_rand_arr = np.asarray(gens_rand, dtype=float)
    gens_llm_arr = np.asarray(gens_llm, dtype=float)
    pct_reduction = 100.0 * (gens_rand_arr.mean() - gens_llm_arr.mean()) / gens_rand_arr.mean()

    if len(seeds) >= 2 and not np.all(gens_rand_arr == gens_llm_arr):
        try:
            _, p = stats.wilcoxon(gens_rand_arr, gens_llm_arr)
        except ValueError:
            p = float("nan")
    else:
        p = float("nan")

    return WarmstartReport(
        seeds=list(seeds),
        gens_random=list(gens_rand),
        gens_llm=list(gens_llm),
        final_random=final_rand,
        final_llm=final_llm,
        pct_reduction=float(pct_reduction),
        p_value=float(p),
    )
