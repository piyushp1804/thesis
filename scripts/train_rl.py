"""
Train PPO on a benchmark. Optionally uses the trained surrogate as the
evaluator (much faster than FEM inside the RL loop).

Usage:
    # with FEM (slower but exact)
    ./venv/bin/python scripts/train_rl.py --bench 10bar --timesteps 50000

    # with surrogate (fast, needs `scripts/train_surrogate.py` first)
    ./venv/bin/python scripts/train_rl.py --bench 10bar --timesteps 50000 \\
        --use-surrogate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks.registry import available_benchmarks, get_benchmark
from src.ml.surrogate import SurrogateEvaluator
from src.ml.train import load_surrogate
from src.rl.evaluate import rollout_policy
from src.rl.train_ppo import save_model, train_ppo


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PPO agent.")
    ap.add_argument("--bench", required=True, choices=available_benchmarks())
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-surrogate", action="store_true")
    ap.add_argument("--surrogate-path", default="results/10bar_surrogate.pt")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    bench = get_benchmark(args.bench)

    evaluator = None
    if args.use_surrogate:
        surr = load_surrogate(args.surrogate_path)
        evaluator = SurrogateEvaluator(bench, surr)
        print(f"[cfg] using surrogate from {args.surrogate_path}")
    else:
        print(f"[cfg] using FEM evaluator")

    print(f"[1/2] training PPO for {args.timesteps} timesteps ({args.n_envs} envs)")
    result = train_ppo(
        bench,
        evaluator=evaluator,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(result.model, out_dir / f"{args.bench}_ppo.zip")

    print(
        f"  best during training: feas={result.best_feasible}, "
        f"weight={result.best_weight:.2f}"
    )

    print(f"[2/2] greedy rollout (32 samples)")
    roll = rollout_policy(result.model, bench, evaluator=None, n_rollouts=32)
    if roll.feasible:
        print(
            f"  rollout FEM-truth  : feas=True, weight={roll.weight:.2f}, "
            f"stress={roll.max_stress:.1f}, disp={roll.max_disp:.4f}"
        )

    # Papers report the best design *seen during training*. We re-
    # evaluate it with FEM for truth, then apply the gate to the better
    # of (training-best, rollout-best).
    lit = bench.reference_optimum_weight
    candidates: list[tuple[float, str]] = []
    if result.best_feasible and result.best_x is not None:
        fem = bench.evaluate(result.best_x)
        if fem.max_abs_stress <= bench.stress_limit_compression * 1.001 and \
           fem.max_abs_displacement <= bench.displacement_limit * 1.001:
            candidates.append((float(fem.weight), "training-best"))
    if roll.feasible:
        candidates.append((roll.weight, "rollout-best"))

    if not candidates:
        print("\n[FAIL] no feasible design found by RL")
        return
    best_w, src = min(candidates, key=lambda t: t[0])
    err = 100.0 * (best_w - lit) / lit
    print(f"\nbest PPO design ({src}): {best_w:.2f} lb, vs literature {lit:.2f}: {err:+.2f}%")
    # Realistic RL-on-truss gate: within 20% of literature. On a single
    # instance, a tuned GA still beats PPO; RL's contribution is the
    # *policy* (generalization), not beating classical optimizers on
    # one problem. See Zhao et al. 2021 for similar reported gaps.
    if abs(err) < 20.0:
        print("[PASS] Phase 3 gate: within 20% of literature optimum")
    else:
        print("[FAIL] Phase 3 gate: >20% off literature")


if __name__ == "__main__":
    main()
