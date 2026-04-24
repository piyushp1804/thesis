"""
Run one algorithm on one benchmark across multiple seeds. Saves each
result as pickle + writes a combined CSV summary.

Used by Phase 1 gate tests and Chapter 4 "mean +/- std over 10 seeds"
tables.

Usage:
    ./venv/bin/python scripts/run_batch.py --algo ga --bench 10bar \\
        --seeds 42 123 456 789 2024
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Shim for direct execution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from src.algorithms.runner import available_algorithms, run
from src.benchmarks.registry import available_benchmarks, get_benchmark


DEFAULT_SEEDS = (42, 123, 456, 789, 2024, 31415, 27182, 11235, 8675309, 9999)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-run seeds.")
    ap.add_argument("--algo", required=True, choices=available_algorithms())
    ap.add_argument("--bench", required=True, choices=available_benchmarks())
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="list of integer seeds (default: 10 literature-standard seeds)",
    )
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--n-gen", type=int, default=500)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    bench = get_benchmark(args.bench)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    summary_path = out_dir / f"{args.bench}_{args.algo}_summary.csv"
    for seed in tqdm(args.seeds, desc=f"{args.algo}/{args.bench}"):
        pkl = out_dir / f"{args.bench}_{args.algo}_seed{seed}.pkl"
        # Resume: if pickle already exists and loads cleanly, reuse it.
        if pkl.exists():
            try:
                with pkl.open("rb") as f:
                    result = pickle.load(f)
                print(f"[skip] {args.algo}/{args.bench} seed={seed} (cached)", flush=True)
                rows.append(result.to_summary_dict())
                continue
            except Exception:
                print(f"[warn] {pkl.name} corrupt, re-running", flush=True)
        print(f"[start] {args.algo}/{args.bench} seed={seed}", flush=True)
        result = run(
            args.algo,
            bench,
            seed=seed,
            pop_size=args.pop_size,
            n_gen=args.n_gen,
            verbose=False,
        )
        # Atomic write so partial failures never leave a half-pickle.
        tmp = pkl.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(result, f)
        tmp.replace(pkl)
        rows.append(result.to_summary_dict())
        # Write incremental CSV after every seed so crashes don't lose CSV.
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"[done]  {args.algo}/{args.bench} seed={seed} W={result.to_summary_dict().get('best_weight'):.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)
    print()
    print(df.to_string(index=False))
    print()
    print("weight stats: mean={:.2f} std={:.2f} min={:.2f} max={:.2f}".format(
        df.best_weight.mean(),
        df.best_weight.std(),
        df.best_weight.min(),
        df.best_weight.max(),
    ))
    print(f"feasible fraction: {df.feasible.mean():.0%}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
