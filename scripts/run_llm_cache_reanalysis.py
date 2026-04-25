"""LLM warm-start cache re-analysis (thesis §4.15).

Reads the three cached Claude Opus 4.7 responses in
``results/llm_cache/*.json`` and the per-seed warm-start CSVs, and
produces a retrospective breakdown of:

  (i)  reported LLM confidence vs realised speedup (gens-to-converge
       ratio random / llm) and vs realised weight gain;
  (ii) LLM reasoning text: length + frequency of domain keywords
       (load path, buckling, tension, compression, symmetry);
  (iii) for 10-bar (the only benchmark with pickles committed to disk),
        parity between LLM-suggested areas and the optimizer-converged
        median area per design variable.

Writes:

  * results/llm_cache_reanalysis.csv    one row per benchmark
  * results/llm_cache_keywords.csv      one row per (benchmark, keyword)
  * figures/fig_4_15_llm_cache_reanalysis.{png,svg}   2x2 composite
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting.style import PALETTE, FIGSIZE_WIDE, save, setup  # noqa: E402


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# n_areas in each cache -> our benchmark label.
BENCH_BY_NAREAS = {10: "10bar", 8: "25bar", 16: "72bar"}

DOMAIN_KEYWORDS = [
    "load path",
    "symmetry",
    "buckling",
    "tension",
    "compression",
    "stress",
    "displacement",
    "chord",
    "diagonal",
    "vertical",
    "horizontal",
]


def load_caches() -> dict[str, dict]:
    """Return {benchmark: parsed_dict} keyed by our benchmark label."""
    out: dict[str, dict] = {}
    for f in sorted((RESULTS / "llm_cache").glob("*.json")):
        d = json.load(open(f))
        parsed = d["parsed"]
        n = len(parsed["areas"])
        bench = BENCH_BY_NAREAS.get(n)
        if bench is None:
            print(f"  [warn] skipping {f.name} (n_areas={n})")
            continue
        out[bench] = parsed
    return out


def keyword_counts(text: str) -> dict[str, int]:
    """Case-insensitive substring count; returns a dict keyword -> count."""
    lc = text.lower()
    return {kw: len(re.findall(re.escape(kw), lc)) for kw in DOMAIN_KEYWORDS}


def realised_stats(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    speedup = df["gens_random"] / df["gens_llm"].replace(0, np.nan)
    weight_gain = df["final_random"] - df["final_llm"]
    pct_weight = 100.0 * weight_gain / df["final_random"]
    return {
        "n_seeds": int(len(df)),
        "median_speedup": float(speedup.median()),
        "iqr_speedup": float(speedup.quantile(0.75) - speedup.quantile(0.25)),
        "median_weight_gain_abs": float(weight_gain.median()),
        "median_weight_gain_pct": float(pct_weight.median()),
    }


def achieved_10bar_median_areas() -> np.ndarray | None:
    """Median component-wise best_x across 10-bar GA seeds, or None."""
    xs = []
    for p in sorted(RESULTS.glob("10bar_ga_seed*.pkl")):
        try:
            r = pickle.load(open(p, "rb"))
            if r.feasible:
                xs.append(np.asarray(r.best_x, dtype=float))
        except Exception:
            pass
    if not xs:
        return None
    X = np.stack(xs, axis=0)
    return np.median(X, axis=0)


def main() -> None:
    setup()
    caches = load_caches()
    rows = []
    keyword_rows = []

    for bench in ("10bar", "25bar", "72bar"):
        if bench not in caches:
            print(f"  [warn] no cache for {bench}")
            continue
        p = caches[bench]
        reasoning = p.get("reasoning", "") or ""
        confidence = float(p.get("confidence", np.nan))

        stats = realised_stats(RESULTS / f"{bench}_llm_warmstart.csv")

        rows.append({
            "benchmark": bench,
            "llm_confidence": confidence,
            "reasoning_chars": len(reasoning),
            "reasoning_words": len(reasoning.split()),
            **stats,
        })

        for kw, c in keyword_counts(reasoning).items():
            keyword_rows.append({"benchmark": bench, "keyword": kw, "count": c})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "llm_cache_reanalysis.csv", index=False)
    print(df.to_string(index=False))

    kw_df = pd.DataFrame(keyword_rows)
    kw_df.to_csv(RESULTS / "llm_cache_keywords.csv", index=False)

    # Parity of LLM suggestion vs achieved optimum (10-bar only).
    achieved = achieved_10bar_median_areas()
    llm_areas_10 = np.asarray(caches["10bar"]["areas"], dtype=float) if "10bar" in caches else None

    # ---- figure: 2x2 composite ----
    fig, axes = plt.subplots(2, 2, figsize=(FIGSIZE_WIDE[0] * 1.25, FIGSIZE_WIDE[1] * 2.0))

    # (a) Confidence vs median speedup
    ax = axes[0, 0]
    ax.scatter(df["llm_confidence"], df["median_speedup"],
               s=70, color=PALETTE["llm"], edgecolor="black", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["benchmark"], (r["llm_confidence"], r["median_speedup"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("LLM self-reported confidence")
    ax.set_ylabel("median speedup ($\\mathrm{gens_{rand}}/\\mathrm{gens_{llm}}$)")
    ax.set_title("(a) Stated confidence vs realised speedup")
    ax.axhline(1.0, ls="--", color="0.6", lw=1.0, label="no speedup")
    ax.legend(frameon=False, loc="lower left", fontsize=8)

    # (b) Confidence vs median weight gain percentage
    ax = axes[0, 1]
    ax.scatter(df["llm_confidence"], df["median_weight_gain_pct"],
               s=70, color=PALETTE["llm"], edgecolor="black", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["benchmark"], (r["llm_confidence"], r["median_weight_gain_pct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("LLM self-reported confidence")
    ax.set_ylabel("median final-weight reduction (\\%)")
    ax.set_title("(b) Stated confidence vs realised weight gain")
    ax.axhline(0.0, ls="--", color="0.6", lw=1.0)

    # (c) Keyword frequency heatmap
    ax = axes[1, 0]
    pivot = kw_df.pivot(index="keyword", columns="benchmark", values="count").fillna(0).astype(int)
    pivot = pivot.loc[DOMAIN_KEYWORDS]                       # preserve declared order
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = int(pivot.values[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > 1 else "black", fontsize=9)
    ax.set_title("(c) Domain-keyword occurrences in reasoning")
    fig.colorbar(im, ax=ax, shrink=0.7, label="count")

    # (d) 10-bar LLM suggestion vs achieved median
    ax = axes[1, 1]
    if llm_areas_10 is not None and achieved is not None and len(llm_areas_10) == len(achieved):
        ax.scatter(llm_areas_10, achieved, s=70,
                   color=PALETTE["ga"], edgecolor="black", zorder=3)
        for i in range(len(achieved)):
            ax.annotate(str(i + 1), (llm_areas_10[i], achieved[i]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        mx = max(float(llm_areas_10.max()), float(achieved.max())) * 1.05
        ax.plot([0, mx], [0, mx], ls="--", color="0.5", lw=1.0, label="y = x")
        ax.set_xlabel("LLM-suggested area (in$^2$)")
        ax.set_ylabel("GA-achieved median area across 10 seeds (in$^2$)")
        ax.set_title("(d) 10-bar: LLM suggestion vs converged optimum")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    else:
        ax.text(0.5, 0.5, "10-bar pickles not available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title("(d) 10-bar parity (skipped)")

    fig.tight_layout()
    save(fig, FIGURES, "fig_4_15_llm_cache_reanalysis")
    print("[llm-cache] wrote figures/fig_4_15_llm_cache_reanalysis.(png|svg)")


if __name__ == "__main__":
    main()
