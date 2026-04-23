"""
Walk results/ and produce one big scoreboard CSV: (benchmark, algorithm,
n_seeds, best, mean +/- std, feasible_fraction, lit_opt, pct_error).

This is the Chapter-4 headline table.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Shim for direct execution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.benchmarks.registry import available_benchmarks, get_benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Phase-1 results.")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/scoreboard.csv")
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    if not res_dir.is_dir():
        sys.exit(f"no such dir: {res_dir}")

    records: list[dict] = []
    for pkl in sorted(res_dir.glob("*_seed*.pkl")):
        with pkl.open("rb") as f:
            r = pickle.load(f)
        records.append(r.to_summary_dict())

    if not records:
        sys.exit("no result pickles found — run scripts/run_batch.py first.")

    df = pd.DataFrame(records)

    lit_by_bench = {}
    for name in available_benchmarks():
        try:
            b = get_benchmark(name)
            lit_by_bench[b.name] = b.reference_optimum_weight
        except NotImplementedError:
            pass

    grp = (
        df.groupby(["benchmark", "algorithm"])
        .agg(
            n_seeds=("best_weight", "size"),
            best=("best_weight", "min"),
            mean=("best_weight", "mean"),
            std=("best_weight", "std"),
            feasible_frac=("feasible", "mean"),
            mean_evals=("n_evals", "mean"),
            mean_time_s=("wall_time_s", "mean"),
        )
        .reset_index()
    )
    grp["lit_opt"] = grp["benchmark"].map(lit_by_bench)
    grp["pct_err_best"] = (
        100.0 * (grp["best"] - grp["lit_opt"]) / grp["lit_opt"]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grp.to_csv(out_path, index=False)
    print(grp.to_string(index=False))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
