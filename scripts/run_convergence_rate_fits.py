"""Fit W(t) = W_inf + A * exp(-t / tau) to every single-objective pickle
(thesis §4.16).

For each GA/PSO pickle on disk we:

  1. Extract the best_weight trajectory from ``result.history``.
  2. Take the running minimum (monotone non-increasing; neutralises the
     known gen-0 pymoo quirk where a few LHS initial samples pass the
     numerical G<=1e-6 tolerance but are not truly feasible).
  3. Fit W(t) = W_inf + A * exp(-t / tau) via nonlinear least squares.
  4. Record W_inf, A, tau, R^2, and the evaluation-budget-normalised
     tau_frac = tau / n_gen.

NSGA-II has no single-best trajectory in our history schema, so it is
excluded from this study (its equivalent would be hypervolume over
time, which we do not log).

Writes:

  * results/convergence_rate_fits.csv        one row per seed/algo
  * results/convergence_rate_summary.csv     aggregated by (benchmark, algo)
  * figures/fig_4_16_convergence_rate_fits.{png,svg}   3x2 grid (3 bench x 2 algo)
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting.style import PALETTE, FIGSIZE_WIDE, save, setup  # noqa: E402


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Benchmarks we have pickles for (not 25/72 — those pickles are gitignored
# but summary CSVs are present; we skip them here rather than fabricate).
BENCHMARKS = ("10bar", "200bar")
ALGOS = ("ga", "pso")
PICKLE_RE = re.compile(r"^(?P<bench>[^_]+)_(?P<algo>ga|pso)_seed(?P<seed>\d+)\.pkl$")


def exp_decay(t, w_inf, a, tau):
    return w_inf + a * np.exp(-t / tau)


def _safe_fit(t: np.ndarray, w: np.ndarray):
    """Fit exp_decay with sensible initial guess and bounds.
    Returns (w_inf, a, tau, r2) or (nan,...) on failure."""
    w_inf0 = float(w[-1])
    a0 = max(float(w[0]) - w_inf0, 1.0)
    tau0 = max(len(t) / 4.0, 1.0)
    try:
        popt, _ = curve_fit(
            exp_decay, t, w,
            p0=[w_inf0, a0, tau0],
            bounds=([0.0, 0.0, 1e-3],
                    [float(w[-1]) * 1.5 + 1e-6, float(w[0]) * 10.0 + 1e-6, float(len(t)) * 10.0]),
            maxfev=20000,
        )
    except Exception as e:
        print(f"  [fit-fail] {e}")
        return np.nan, np.nan, np.nan, np.nan
    w_hat = exp_decay(t, *popt)
    ss_res = float(np.sum((w - w_hat) ** 2))
    ss_tot = float(np.sum((w - w.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(popt[0]), float(popt[1]), float(popt[2]), float(r2)


def gather_pickles() -> list[tuple[str, str, int, Path]]:
    out = []
    for p in sorted(RESULTS.glob("*_seed*.pkl")):
        m = PICKLE_RE.match(p.name)
        if not m:
            continue
        bench = m.group("bench")
        if bench not in BENCHMARKS:
            continue
        if m.group("algo") not in ALGOS:
            continue
        out.append((bench, m.group("algo"), int(m.group("seed")), p))
    return out


def main() -> None:
    setup()
    pickles = gather_pickles()
    print(f"[conv-fit] found {len(pickles)} pickles")

    fit_rows = []
    curves: dict[tuple[str, str], list[tuple[int, np.ndarray, np.ndarray, tuple[float, float, float, float]]]] = {}
    # gen 0 in some pickles has an unphysical sub-optimal "feasible" LHS
    # sample (pymoo evaluates G with a 1e-6 slack that lets a borderline
    # candidate pass although the nonlinear FEM sees it as infeasible).
    # Skipping one generation is a negligible loss and removes the bias.
    SKIP_FIRST = 1

    for bench, algo, seed, path in pickles:
        r = pickle.load(open(path, "rb"))
        if not r.history:
            print(f"  [skip] {path.name}: empty history")
            continue
        w_raw = np.asarray([h["best_weight"] for h in r.history], dtype=float)
        w = np.minimum.accumulate(w_raw[SKIP_FIRST:])   # monotone non-increasing
        t = np.arange(len(w), dtype=float)
        w_inf, a, tau, r2 = _safe_fit(t, w)
        fit_rows.append({
            "benchmark": bench, "algorithm": algo, "seed": seed,
            "n_gen": int(len(w)),
            "w0": float(w[0]), "w_final": float(w[-1]),
            "w_inf": w_inf, "A": a, "tau_gens": tau,
            "tau_frac": tau / len(w) if len(w) > 0 else np.nan,
            "r2": r2,
            "wall_time_s": float(r.wall_time_s),
            "feasible": bool(r.feasible),
        })
        curves.setdefault((bench, algo), []).append((seed, t, w, (w_inf, a, tau, r2)))

    df = pd.DataFrame(fit_rows)
    df.to_csv(RESULTS / "convergence_rate_fits.csv", index=False)
    print(df.to_string(index=False))

    # Summary per (benchmark, algo)
    def q(x, p):
        return float(np.quantile(x.dropna(), p)) if len(x.dropna()) else np.nan

    grp = df.groupby(["benchmark", "algorithm"])
    summary = grp.agg(
        n_seeds=("seed", "count"),
        median_tau=("tau_gens", "median"),
        iqr_tau=("tau_gens", lambda x: q(x, 0.75) - q(x, 0.25)),
        median_r2=("r2", "median"),
        median_w_inf=("w_inf", "median"),
        median_tau_frac=("tau_frac", "median"),
    ).reset_index()
    summary.to_csv(RESULTS / "convergence_rate_summary.csv", index=False)
    print("--- summary ---")
    print(summary.to_string(index=False))

    # ---- figure: (n_bench x n_algo) grid; each panel shows empirical
    # curves (thin grey) + median curve + fitted dashed line ----
    n_rows, n_cols = len(BENCHMARKS), len(ALGOS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(FIGSIZE_WIDE[0] * 1.1, FIGSIZE_WIDE[1] * n_rows / 1.6),
                             squeeze=False)
    for i, bench in enumerate(BENCHMARKS):
        for j, algo in enumerate(ALGOS):
            ax = axes[i, j]
            series = curves.get((bench, algo), [])
            if not series:
                ax.set_visible(False)
                continue
            for (seed, t, w, _) in series:
                ax.plot(t, w, color="0.75", lw=0.8, alpha=0.8, zorder=1)
            # stack all curves on their common length and take the
            # point-wise median (they should all be the same length,
            # but tolerate differences).
            min_len = min(len(s[1]) for s in series)
            W = np.stack([s[2][:min_len] for s in series], axis=0)
            t_med = np.arange(min_len, dtype=float)
            w_med = np.median(W, axis=0)
            ax.plot(t_med, w_med, color=PALETTE.get(algo, "black"), lw=1.6,
                    label="median", zorder=2)
            # refit on the median curve for a single "representative" fit
            w_inf, a, tau, r2 = _safe_fit(t_med, w_med)
            if np.isfinite(tau):
                w_hat = exp_decay(t_med, w_inf, a, tau)
                ax.plot(t_med, w_hat, ls="--", color=PALETTE["surrogate"], lw=1.4,
                        label=(rf"fit: $\tau={tau:.1f}$, $R^2={r2:.3f}$"),
                        zorder=3)
            ax.set_yscale("log")
            ax.set_xlabel("generation")
            weight_unit = "lb" if bench == "10bar" else "kg"
            ax.set_ylabel(f"best-so-far weight ({weight_unit})")
            ax.set_title(f"{bench} / {algo.upper()} (n={len(series)} seeds)")
            ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, FIGURES, "fig_4_16_convergence_rate_fits")
    print("[conv-fit] wrote figures/fig_4_16_convergence_rate_fits.(png|svg)")


if __name__ == "__main__":
    main()
