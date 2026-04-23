"""
End-to-end surrogate training driver.

Usage:
    ./venv/bin/python scripts/train_surrogate.py --bench 10bar --n-samples 5000

Steps performed:
  1. LHS-sample `n_samples` designs, FEM each one.
  2. Train MLP.
  3. Print R^2 on a fresh held-out LHS test set (n_test samples).
  4. Report wall-clock speedup surrogate vs FEM.
  5. Save dataset (.npz) and trained model (.pt) to `--out-dir`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks.registry import available_benchmarks, get_benchmark
from src.ml.dataset import generate_dataset
from src.ml.evaluate import r2_report, speedup_vs_fem
from src.ml.surrogate import SurrogateEvaluator
from src.ml.train import save_surrogate, train_surrogate


def main() -> None:
    ap = argparse.ArgumentParser(description="Train neural surrogate.")
    ap.add_argument("--bench", required=True, choices=available_benchmarks())
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-hidden-layers", type=int, default=2)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    bench = get_benchmark(args.bench)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] generating training dataset: {args.n_samples} LHS samples")
    train_ds = generate_dataset(bench, args.n_samples, seed=args.seed)
    train_ds.save(out_dir / f"{args.bench}_dataset_train.npz")

    print(f"[2/4] training MLP ({args.epochs} epochs)")
    surrogate = train_surrogate(
        train_ds,
        seed=args.seed,
        epochs=args.epochs,
        hidden=args.hidden,
        n_hidden_layers=args.n_hidden_layers,
        verbose=True,
    )
    save_surrogate(surrogate, out_dir / f"{args.bench}_surrogate.pt")
    print(f"  val R^2 (held-out 20% of train): {surrogate.val_r2}")

    print(f"[3/4] generating fresh held-out test set ({args.n_test} samples)")
    test_ds = generate_dataset(bench, args.n_test, seed=args.seed + 9999)
    test_ds.save(out_dir / f"{args.bench}_dataset_test.npz")

    evaluator = SurrogateEvaluator(bench, surrogate)
    r2 = r2_report(bench, evaluator, test_ds)
    print(f"  test R^2: weight={r2.weight:.4f}  stress={r2.max_stress:.4f}  disp={r2.max_disp:.4f}")

    print(f"[4/4] speedup benchmark (1000 calls)")
    sp = speedup_vs_fem(bench, evaluator, n_calls=1000)
    print(
        f"  FEM       : {sp.fem_total_s:.3f} s\n"
        f"  surrogate : {sp.surrogate_total_s:.3f} s\n"
        f"  speedup   : {sp.speedup:.1f}x"
    )

    if r2.weight > 0.98 and sp.speedup > 50:
        print("\n[PASS] Phase 2 gate: R^2(weight) > 0.98 AND speedup > 50x")
    else:
        print("\n[FAIL] Phase 2 gate not met; see metrics above.")


if __name__ == "__main__":
    main()
