# Multi-objective steel truss optimization (thesis codebase)

M.Tech thesis project — IIT BHU, Civil Engineering.

This repository implements an end-to-end research stack for sizing classic
benchmark trusses with evolutionary algorithms, an optional neural surrogate,
a PPO agent, an LLM warm-start path, and a small Streamlit + FastAPI demo.

**GitHub**

Canonical repo: [piyushp1804/thesis](https://github.com/piyushp1804/thesis) (remote name `piyush` in the maintainer clone).

An older mirror at `infag1403/thesis` may exist from early pushes, but day-to-day development tracks `piyush` only.

**Benchmark status (Phase 1)**

- **10-bar planar:** fully encoded + literature validation
- **25 / 72 / 200-bar:** scaffolded in `src/benchmarks/` (raise `NotImplementedError` until encoded)

Constraints include classical stress/displacement limits used in the benchmark
literature **plus** optional IS 800:2007-oriented checks in `src/constraints/`.

## Repository layout

```
src/
  fem/           # Truss FEM (elements, assembly, linear solve)
  algorithms/    # GA, PSO, NSGA-II (pymoo adapters + runner)
  benchmarks/    # BenchmarkProblem definitions + registry
  constraints/   # IS 800-style checks + compliance orchestration
  ml/            # LHS dataset + MLP surrogate + training utilities
  rl/            # Gymnasium env + SB3 PPO training helpers
  llm/           # Claude client/cache + warm-start designer
  app/           # FastAPI (`api.py`) + Streamlit UI (`ui.py`)
tests/           # pytest suites (fast + a few slow “gate” tests)
scripts/         # CLI runners (single/batch opt, surrogate train, RL train, LLM A/B)
results/         # Generated artifacts (mostly git-ignored; LLM cache JSON is tracked)
docs/            # Thesis vision + ground-truth tables + playbooks
```

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt

# Fast tests (CI runs this set)
pytest -m "not slow" -q

# Slow gate tests (minutes — thesis validation, not CI)
pytest -m "slow" -q
```

## Common commands

```bash
# Single optimization run (saves a pickle under results/)
python scripts/run_single.py --algo pso --bench 10bar --seed 42 --pop-size 100 --n-gen 500

# Train surrogate + report R^2 / speedup
python scripts/train_surrogate.py --bench 10bar --n-samples 8000 --n-test 1000

# Train PPO (FEM evaluator by default)
python scripts/train_rl.py --bench 10bar --timesteps 150000

# LLM warm-start A/B (uses ANTHROPIC_API_KEY from .env if set; otherwise heuristic)
python scripts/run_llm_warmstart.py --bench 10bar

# Demo UI / API
python scripts/run_ui.py
python scripts/run_api.py
```

## Secrets / environment

Copy `.env.example` → `.env` locally. **Never commit `.env`** (API keys).

## Documentation

- [docs/thesis_vision.md](docs/thesis_vision.md)
- [docs/ground_truth.md](docs/ground_truth.md)
- [docs/cursor_playbook.md](docs/cursor_playbook.md)
- [docs/thesis_explained_simply.md](docs/thesis_explained_simply.md)
