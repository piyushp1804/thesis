# Multi-Objective Truss Optimization

M.Tech thesis project — IIT BHU, Civil Engineering.

Optimizes steel truss cross-sections using three evolutionary algorithms
(GA, PSO, NSGA-II) subject to IS 800:2007 Indian steel code constraints.
Validated on four classic benchmarks: 10-bar, 25-bar, 72-bar, and 200-bar trusses.

## Project layout

```
src/
  fem/           # Finite Element Method solver (truss elements, assembly, linear solver)
  algorithms/    # GA, PSO, NSGA-II wrappers around pymoo
  benchmarks/    # 10-bar, 25-bar, 72-bar, 200-bar truss definitions
  constraints/   # IS 800:2007 strength / stability / deflection checks
  plotting/      # Publication-quality figure generators
  utils/         # Shared helpers
tests/           # pytest test suites
scripts/         # Run scripts (single runs, batch orchestrator)
results/         # Optimization outputs (.pkl files, generated — git-ignored)
figures/         # Plots (.png/.svg, generated — git-ignored)
docs/            # Planning docs (thesis explainer, Cursor playbook)
thesis_writeup/  # Chapter drafts (Days 8-10)
```

## Quick start

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests (once Day 1 code exists)
pytest tests/ -v
```

## 10-day build plan

See [docs/cursor_playbook.md](docs/cursor_playbook.md) for the day-by-day prompt sequence.
See [docs/thesis_explained_simply.md](docs/thesis_explained_simply.md) for a plain-English thesis overview.

## First git commit

After verifying the scaffold:

```bash
git init
git add .
git commit -m 'Day 0: initial scaffold'
```
