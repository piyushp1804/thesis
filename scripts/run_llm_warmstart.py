"""
Reproduce Phase 4's headline table: LLM-warmstart vs random-init GA on
the 10-bar benchmark.

Usage:
    ./venv/bin/python scripts/run_llm_warmstart.py --bench 10bar --seeds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.benchmarks.registry import available_benchmarks, get_benchmark
from src.llm.designer import suggest_initial_design
from src.llm.evaluate import compare_warmstart_vs_random


DEFAULT_SEEDS = [42, 123, 456, 789, 2024]


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM warmstart comparison.")
    ap.add_argument("--bench", required=True, choices=available_benchmarks())
    ap.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=DEFAULT_SEEDS,
    )
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--n-gen", type=int, default=500)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    bench = get_benchmark(args.bench)
    suggestion = suggest_initial_design(bench)
    print(f"[LLM] source   : {suggestion.source}")
    print(f"[LLM] reasoning: {suggestion.reasoning}")
    print(f"[LLM] confidence: {suggestion.confidence}")
    print(f"[LLM] x0        : {suggestion.x}")

    print(f"\nrunning GA x2 (random vs LLM warmstart) on {len(args.seeds)} seeds")
    report = compare_warmstart_vs_random(
        bench,
        seeds=args.seeds,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
    )

    df = pd.DataFrame(
        {
            "seed": report.seeds,
            "gens_random": report.gens_random,
            "gens_llm": report.gens_llm,
            "final_random": report.final_random,
            "final_llm": report.final_llm,
        }
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.bench}_llm_warmstart.csv"
    df.to_csv(path, index=False)

    print()
    print(df.to_string(index=False))
    print()
    print(f"mean gens (random) : {sum(report.gens_random)/len(report.gens_random):.1f}")
    print(f"mean gens (llm)    : {sum(report.gens_llm)/len(report.gens_llm):.1f}")
    print(f"reduction          : {report.pct_reduction:+.1f}%")
    print(f"p-value (Wilcoxon) : {report.p_value:.4f}")

    if report.pct_reduction >= 15.0 and report.p_value < 0.05:
        print("\n[PASS] Phase 4 gate: >=15% reduction, p<0.05")
    elif report.pct_reduction >= 15.0:
        print(f"\n[PARTIAL] {report.pct_reduction:.1f}% reduction but p={report.p_value:.3f}")
    else:
        print(f"\n[FAIL] Phase 4 gate: {report.pct_reduction:.1f}% reduction")


if __name__ == "__main__":
    main()
