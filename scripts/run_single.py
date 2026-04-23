"""
Run one optimizer on one benchmark, print summary, save pickle.

Usage:
    ./venv/bin/python scripts/run_single.py --algo ga --bench 10bar --seed 42
    ./venv/bin/python scripts/run_single.py --algo nsga2 --bench 10bar --seed 7 --n-gen 300
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Shim so `python scripts/run_single.py` works without a package install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.algorithms.runner import available_algorithms, run
from src.benchmarks.registry import available_benchmarks, get_benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a single optimization.")
    ap.add_argument("--algo", required=True, choices=available_algorithms())
    ap.add_argument("--bench", required=True, choices=available_benchmarks())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--n-gen", type=int, default=500)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    bench = get_benchmark(args.bench)
    result = run(
        args.algo,
        bench,
        seed=args.seed,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        verbose=args.verbose,
    )

    print()
    print(f"=== {args.algo.upper()} on {bench.name} (seed={args.seed}) ===")
    print(f"  best weight  : {result.best_weight:.3f}")
    print(f"  feasible     : {result.feasible}")
    print(f"  max stress   : {result.max_stress:.3f}")
    print(f"  max disp     : {result.max_displacement:.4f}")
    print(f"  wall time    : {result.wall_time_s:.2f} s")
    print(f"  FEM evals    : {result.n_evals}")
    if result.pareto_f is not None:
        print(f"  Pareto pts   : {len(result.pareto_f)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir / f"{args.bench}_{args.algo}_seed{args.seed}.pkl"
    )
    with out_path.open("wb") as f:
        pickle.dump(result, f)
    print(f"  saved to     : {out_path}")


if __name__ == "__main__":
    main()
