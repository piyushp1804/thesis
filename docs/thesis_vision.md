Thesis Vision - For Cursor Plan Mode 
 
 
THESIS VISION DOCUMENT 
For Cursor Plan Mode 
———————————————————————— 
An AI-Powered Framework for Multi-Objective 
Steel Truss Optimization: Integrating Evolutionary 
Algorithms, Neural Surrogate Models, and 
LLM-Assisted Design 
This document describes WHAT to build and WHY. 
It does not describe WHEN. 
Feed this entire doc into Cursor's plan mode 
and Cursor will propose its own build sequence. 
For: Aryan | IDD Civil Y5 | IIT BHU 
Guide: Dr. Krishna Kant Pathak 
Workspace: /Users/rajnishkumar/thesis/ 
How to Use This Document in Cursor Plan Mode 
This document is the single source of truth for what the thesis is. Cursor can read it once and understand the entire scope, architecture, data flow, validation gates, and deliverables. 
Recommended workflow 
1. Save this document to docs/thesis_vision.md in your workspace (Cursor can import Word into markdown automatically). 
2. Open Cursor. Enable Plan Mode (not Chat mode, not Agent mode — specifically Plan). 
3. Point Cursor at docs/thesis_vision.md and say: 'Read this thesis vision document in full. Understand the current repository state (Day 1 complete). Propose a multi-step implementation plan to complete the remaining components. Prioritize by dependency order.' 
4. Review Cursor's proposed plan. Edit or approve. Cursor will then execute step by step, pausing for your approval between major milestones. 
5. Keep this document version-controlled. Update it only if scope changes (new component, removed component) — NOT for daily tweaks. 
What's in this document 
* Section 1: Current state — what Day 1 has produced (FEM foundation) 
* Section 2: Vision — the five-layer AI system, described completely 
* Section 3: Components — each layer's responsibilities, interfaces, dependencies 
* Section 4: Data flow — how information moves through the system 
* Section 5: Validation gates — what 'done' looks like for each component 
* Section 6: Deliverables — what the repository ships at the end 
* Section 7: External references — which papers, codes, datasets to consult 
* Section 8: Constraints and non-goals — what NOT to build 
CURSOR PLAN MODE TIP 
When Cursor proposes its plan, insist it references SPECIFIC file paths (e.g., src/algorithms/ga.py) and specific functions. If the plan is vague ('implement optimization'), reject it and re-prompt with: 'Re-plan with file-level granularity.' 
 Section 1 — Current State (Day 1 Complete) 
The FEM foundation layer is complete and validated to machine precision. This section documents what exists, so Cursor knows what NOT to rewrite. 
1.1 Repository layout as of Day 1 
thesis/ 
├── src/ 
│ ├── __init__.py 
│ ├── fem/ 
│ │ ├── __init__.py 
│ │ ├── truss_element.py (194 lines) ✓ COMPLETE 
│ │ ├── assembly.py (146 lines) ✓ COMPLETE 
│ │ ├── solver.py (171 lines) ✓ COMPLETE 
│ │ ├── post_process.py (145 lines) ✓ COMPLETE 
│ │ └── truss.py (218 lines) ✓ COMPLETE 
│ ├── algorithms/ (empty — to be built) 
│ ├── benchmarks/ (empty — to be built) 
│ ├── constraints/ (empty — to be built) 
│ ├── plotting/ (empty — to be built) 
│ ├── utils/ (empty — to be built) 
│ ├── ml/ (empty — to be built) 
│ ├── rl/ (empty — to be built) 
│ └── llm/ (empty — to be built) 
├── tests/ 
│ ├── test_truss_element.py (151 lines) ✓ 15 tests passing 
│ └── test_fem_3bar.py (209 lines) ✓ 6 tests passing 
├── scripts/ 
│ └── fem_demo.py (109 lines) ✓ COMPLETE 
├── results/ (empty) 
├── figures/ (empty) 
├── docs/ 
│ ├── thesis_explained_simply.md 
│ └── cursor_playbook.md 
├── thesis_writeup/ (empty) 
├── requirements.txt 
├── pyproject.toml 
└── README.md 
1.2 Validated behavior (do not regress) 
FEM engine — validated to machine precision 
3-bar canonical problem: node 3 u_y computed -2.928932e-05 m, expected -2.928932e-05 m, error 6.78e-21. Equilibrium residual 2.27e-13 (effectively zero). All 21 pytest cases green. 
 1.3 Locked-in design decisions 
These decisions from Day 1 are load-bearing for subsequent layers. Cursor should not revisit them without explicit instruction. 
* Dimension-agnostic element: one code path handles 2D and 3D bars. Day 2 benchmarks (25-bar and 72-bar spatial) slot in without modification. 
* Partition BC method (not penalty): clean, numerically stable, ready for non-zero prescribed displacements if needed later. 
* Sign convention: + = tension, - = compression. Locked everywhere. Matches IS 800 constraint formulation as g(x) <= 0. 
* Areas are the only mutable state on Truss (via set_areas). Designed for the GA/PSO inner loop: build once, mutate areas, re-solve, repeat. 
* Python 3.14 with numpy 2.4.4, scipy 1.17.1, pymoo 0.6.1.6. No known deprecations. 
1.4 Truss class public interface (what other layers will call) 
# The Truss class in src/fem/truss.py is the primary entrypoint. 
# Other layers (algorithms, benchmarks, ml, rl) will all use this interface. 
 
truss = Truss( 
 nodes=[(x1, y1), (x2, y2), ...], # 2D or 3D tuples 
 connectivity=[(i, j), ...], # 0-indexed pairs 
 E=2.0e5, # Young's modulus 
 rho=7850.0, # density 
) 
 
truss.set_boundary_conditions(fixed_dofs=[0, 1, 2, ...]) 
truss.set_loads({(node_idx): (fx, fy)}) # or (fx, fy, fz) 
truss.set_areas(areas_array) # numpy array, one per element 
 
result = truss.solve() 
# result contains: 
# - displacements: (n_nodes, ndim) array 
# - reactions: at fixed DOFs 
# - element_forces: axial forces (+ tension, - compression) 
# - element_stresses: axial stresses 
# - weight: total structural weight 
1.5 What Day 1 does NOT yet cover (explicitly out of scope for FEM layer) 
* No optimization loop — that's Layer 2 
* No benchmark definitions — that's Layer 2 prerequisite 
* No IS 800 compliance checks — that's Layer 6 
* No ML training, no RL agent, no LLM integration — Layers 3/4/5 
* No plotting, no UI — Layers 7/8 
Section 2 — Vision: The Five-Layer AI System 
This thesis builds a complete computational framework around the Day 1 FEM engine. The framework has five AI/optimization layers stacked on top of the FEM foundation, plus two supporting layers (constraints + plotting) and an integration layer (UI + API). 
2.1 Layer stack overview 
┌─────────────────────────────────────────────────────────────┐ 
│ LAYER 8: UI + API (Streamlit + FastAPI) │ 
├─────────────────────────────────────────────────────────────┤ 
│ LAYER 7: Plotting & Visualization (matplotlib + seaborn) │ 
├─────────────────────────────────────────────────────────────┤ 
│ LAYER 6: IS 800:2007 Compliance Checker │ 
├─────────────────────────────────────────────────────────────┤ 
│ LAYER 5: LLM Designer (Claude / Anthropic API) │ 
├─────────────────────────────────────────────────────────────┤ 
│ LAYER 4: Reinforcement Learning Agent (PPO) │ 
├─────────────────────────────────────────────────────────────┤ 
│ LAYER 3: Neural Surrogate Model (PyTorch MLP) │ 
├─────────────────────────────────────────────────────────────┤ 
│ LAYER 2: Classical Optimizers (GA / PSO / NSGA-II) │ 
├─────────────────────────────────────────────────────────────┤ 
│ LAYER 1: FEM Engine ✓ DONE │ 
├─────────────────────────────────────────────────────────────┤ 
│ BENCHMARKS: 10-bar, 25-bar, 72-bar, 200-bar │ 
└─────────────────────────────────────────────────────────────┘ 
2.2 Core idea, in one paragraph 
Classical evolutionary optimization of steel trusses is well-understood but slow: every candidate design requires a full FEM solve, and thousands of candidates are evaluated per run. This thesis accelerates that process using modern AI techniques. A neural network is trained to mimic the FEM solver with 100x speedup. A reinforcement learning agent learns to propose near-optimal designs directly, bypassing the evolutionary loop. A large language model translates natural-language design requirements into initial design vectors, giving the optimizer a warm start. All four components are compared on four classical benchmark trusses (10-bar, 25-bar, 72-bar, 200-bar) under Indian Standard IS 800:2007 design constraints. 
2.3 Why this is a thesis and not just a project 
* Integrative novelty: no prior published work combines GA+PSO+NSGA-II + neural surrogate + RL + LLM + IS 800 in one reproducible framework. 
* Quantitative comparison: you produce benchmark numbers (convergence times, accuracy vs literature, surrogate speedup) that can be tabulated against existing papers. 
* Reproducibility contribution: all code open-source Python with pinned versions; all results generated from documented scripts. 
* Indian code compliance: IS 800:2007 integration is the civil-engineering relevance anchor (Pathak's lab alignment). 
2.4 What success looks like 
* Single best weights for each benchmark within 1% of published literature values (validation) 
* Neural surrogate achieves R² > 0.98 on held-out test set (ML rigor) 
* Surrogate-accelerated optimization shows 50-150x speedup over pure FEM (ML contribution) 
* RL agent converges to within 5% of literature optimum (RL feasibility demonstration) 
* LLM-initialized optimization converges 20-30% faster than random initialization (LLM contribution) 
* All results satisfy IS 800:2007 slenderness, stress, and deflection clauses (code compliance) 
* Streamlit demo runs end-to-end in under 10 seconds on a laptop (usability) 
Section 3 — Component Specifications 
Each subsection below is a self-contained component specification. Cursor should treat these as independent modules with clear interfaces and test requirements. 
3.0 Benchmarks (pre-requisite for Layers 2-5) 
Location: src/benchmarks/ 
Responsibility: Encode the four canonical benchmark trusses with exact geometry, loading, materials, design variable bounds, and constraint limits from the original literature. 
3.0.1 Public interface 
Each benchmark class exposes the same interface so optimizers are benchmark-agnostic: 
class Truss10Bar: 
 def build_truss(self) -> Truss # returns Day-1 Truss object 
 def get_bounds(self) -> (lower, upper) # area bounds, shape (n_vars,) 
 def get_constraints(self) -> dict # stress, displacement, slenderness limits 
 def get_n_design_vars(self) -> int # 10, 8 (grouped), 16, 29 for the four benchmarks 
 def areas_to_elements(self, x: array) -> array # maps design vars to per-element areas 
 def evaluate(self, x: array) -> dict # runs FEM, returns weight + stresses + displacements 
 def reference_optimum(self) -> dict # published best weight + source citation 
3.0.2 Four benchmarks, from literature 
10-bar planar truss (Sunar & Belegundu 1991): 
* 6 nodes, 10 elements, 2 DOF per node = 12 DOF system 
* Design variables: 10 areas, bounds [0.1, 35.0] in² 
* E = 10⁷ psi, rho = 0.1 lb/in³ 
* Loads: 100 kip downward at nodes 2 and 4 
* Stress limit: ±25 ksi all members 
* Displacement limit: 2.0 in at all free nodes 
* Literature optimum: ~5060.85 lb 
25-bar spatial truss (Venkayya 1971): 
* 10 nodes, 25 elements, 3 DOF per node 
* Design variables: 8 grouped areas (symmetry grouping per paper) 
* E = 10⁷ psi, rho = 0.1 lb/in³ 
* Two load conditions (after symmetry reduction) 
* Literature optimum: ~545 lb 
72-bar spatial truss (Fleury & Schmit 1980): 
* 20 nodes, 72 elements, 4-storey tower 
* Design variables: 16 grouped areas 
* Literature optimum: ~379.6 lb 
200-bar planar truss: 
* 77 nodes, 200 elements — stress-tests scalability 
* Design variables: 29 grouped areas 
* Literature optimum: ~25,445 lb 
VALIDATION GATE — BENCHMARKS 
For each benchmark, substituting the published literature optimum area vector should reproduce the reported weight within 0.5% AND satisfy all reported constraints. This is the Day-2 acceptance test. If this fails, the benchmark is encoded wrong, not the FEM. 
 3.1 Layer 2 — Classical Optimizers 
Location: src/algorithms/ 
Responsibility: Wrap pymoo's GA, PSO, and NSGA-II implementations with a consistent interface that works on any benchmark. 
3.1.1 Modules 
* src/algorithms/problem.py — pymoo Problem subclass. Takes a benchmark, exposes _evaluate() that runs FEM and returns objectives + constraints. 
* src/algorithms/ga.py — Single-objective Genetic Algorithm wrapper. 
* src/algorithms/pso.py — Particle Swarm Optimization wrapper. 
* src/algorithms/nsga2.py — Multi-objective NSGA-II for Pareto fronts. 
* src/algorithms/runner.py — Unified runner: accepts (benchmark, algo_name, seed, hyperparams) and returns a standardized result object. 
3.1.2 Constraint handling 
Use penalty-based constraint handling in the Problem class. g_i(x) <= 0 convention. Violated constraints add large penalty to objective. See Coello 2002 for theoretical grounding. 
3.1.3 Objectives 
* Single-objective mode: minimize weight 
* Multi-objective mode (NSGA-II): minimize (weight, max_displacement) simultaneously 
3.1.4 Hyperparameters (defaults) 
GA/PSO/NSGA-II defaults: 
 pop_size = 100 
 n_gen = 500 
 crossover_prob = 0.9 
 mutation_prob = 1/n_design_vars (polynomial mutation) 
 eta_crossover = 15 (SBX) 
 eta_mutation = 20 (polynomial) 
 seed = configurable (default 42) 
3.1.5 Result object schema 
{ 
 "algorithm": "GA" | "PSO" | "NSGA-II", 
 "benchmark": "10-bar" | ..., 
 "seed": int, 
 "best_x": array of areas (single-obj) or Pareto set (multi-obj), 
 "best_f": scalar weight (single-obj) or Pareto front (multi-obj), 
 "convergence_history": list of (gen, best_f_so_far), 
 "n_evaluations": int, 
 "wall_time_seconds": float, 
 "feasible": bool, 
 "constraint_violations": array 
} 
VALIDATION GATE — LAYER 2 
For the 10-bar benchmark, GA and PSO must independently converge to within 2% of the literature optimum of 5060.85 lb across 10 random seeds. NSGA-II must produce a Pareto front with at least 20 non-dominated points. If any of these fails, the problem formulation or constraint handling is incorrect. 
 3.2 Layer 3 — Neural Surrogate Model 
Location: src/ml/ 
Responsibility: Train a neural network to predict truss behavior (weight, max displacement, max stress, max slenderness, feasibility) from design variables, bypassing the expensive FEM solve. 
3.2.1 Why this is the AI star of the thesis 
The FEM solver takes ~50 ms per evaluation. A 500-generation GA with pop_size=100 makes 50,000 evaluations = 40 minutes of wall time. Replacing FEM with a neural surrogate that evaluates in 0.5 ms makes the same run complete in 25 seconds. This is a 100x speedup, genuinely useful, and fundamentally ML engineering — YOUR strength. 
3.2.2 Modules 
* src/ml/dataset.py — Generate training data: sample N random design vectors per benchmark, evaluate via FEM, store as (X, Y) pairs. 
* src/ml/model.py — PyTorch MLP architecture. Multi-head output (weight, displacement, stress, slenderness). 
* src/ml/train.py — Training loop: Adam optimizer, MSE loss, early stopping, wandb or tensorboard logging. 
* src/ml/surrogate.py — Trained surrogate wrapper that drops into the pymoo Problem class in place of FEM calls. 
* src/ml/evaluate.py — Held-out test set evaluation: R², MAE, calibration curves. 
3.2.3 Model architecture (default) 
Input: design vector x of length n_design_vars (10 for 10-bar, 8 for 25-bar, etc.) 
Hidden: [256, 128, 64] with ReLU activation and dropout p=0.1 
Output: [weight, max_displacement, max_stress, max_slenderness, feasibility_logit] 
 
Loss: weighted MSE on regression heads + BCE on feasibility logit 
Optimizer: Adam, lr=1e-3, weight_decay=1e-5 
Batch size: 128 
Epochs: 200 with early stopping on validation loss 
3.2.4 Training data generation strategy 
* Latin hypercube sampling over design variable bounds 
* N = 10,000 samples per benchmark (adjustable) 
* 80/10/10 train/val/test split with fixed seed 
* Save as results/surrogate_data_.npz for reproducibility 
* Data generation is embarrassingly parallel — use multiprocessing if available 
3.2.5 Serving 
Surrogate wraps a trained .pth checkpoint. Pymoo Problem class has a use_surrogate flag — True uses surrogate, False uses FEM. This lets you run A/B comparisons between surrogate-accelerated and pure-FEM optimization for the thesis results chapter. 
VALIDATION GATE — LAYER 3 
Neural surrogate must achieve R² > 0.98 on held-out test set for weight prediction, and R² > 0.90 for displacement/stress. Surrogate-accelerated GA on 10-bar must converge to within 3% of literature optimum AND achieve at least 50x speedup over pure-FEM GA. 
 3.3 Layer 4 — Reinforcement Learning Agent 
Location: src/rl/ 
Responsibility: Train a PPO agent that learns to propose near-optimal truss designs directly from a state representation, providing a policy-based alternative to evolutionary search. 
3.3.1 Framework choice 
* Stable-Baselines3 for PPO implementation (well-tested, stable, minimal boilerplate) 
* Gymnasium (successor to OpenAI Gym) for environment API 
3.3.2 Environment design 
class TrussDesignEnv(gymnasium.Env): 
 """ 
 State (observation): 
 - Current design vector (areas normalized to [0, 1]) 
 - Remaining budget (e.g., generations left, or step count) 
 
 Action: 
 - Continuous adjustments to design variables: delta_x in [-0.1, 0.1]^n_design_vars 
 - Clipped to keep x within bounds 
 
 Reward per step: 
 r = -normalized_weight 
 - constraint_violation_penalty 
 + feasibility_bonus (if design satisfies all constraints) 
 
 Episode termination: 
 - Max steps reached (e.g., 50 steps) 
 - Or constraint-satisfied design found with weight below threshold 
 """ 
3.3.3 Modules 
* src/rl/environment.py — TrussDesignEnv class, Gym-compatible 
* src/rl/train_ppo.py — Training script using Stable-Baselines3 PPO 
* src/rl/evaluate.py — Load trained agent, run inference on all benchmarks 
* src/rl/reward_shaping.py — Reward function variants for ablation studies 
3.3.4 Training configuration (default) 
PPO hyperparameters: 
 total_timesteps = 100,000 (per benchmark) 
 learning_rate = 3e-4 
 n_steps = 2048 
 batch_size = 64 
 n_epochs = 10 
 gamma = 0.99 
 gae_lambda = 0.95 
 clip_range = 0.2 
 
Policy network: MlpPolicy, [128, 64] hidden 
Value network: MlpPolicy, [128, 64] hidden 
3.3.5 Coupling to surrogate 
Training RL with FEM is too slow (100k steps * 50ms = 83 min). Use the neural surrogate from Layer 3 inside the RL environment. This makes RL training feasible in 5-10 minutes per benchmark. Document this coupling as an original contribution in Chapter 3.3.5. 
VALIDATION GATE — LAYER 4 
PPO agent, after training, must produce designs within 5% of literature optimum on at least 3 of 4 benchmarks. Training curves (episode reward vs timestep) must show monotonic or near-monotonic improvement. If not, hyperparameters or reward shaping need tuning. 
 3.4 Layer 5 — LLM Designer 
Location: src/llm/ 
Responsibility: Given a natural-language problem statement, call the Anthropic Claude API to generate an initial design vector that the classical optimizer can then refine. Demonstrate that LLM-initialized optimization converges faster than random-initialized. 
3.4.1 Why this is honest and NOT 'fake CNN with API' 
The LLM does NOT replace any model. The LLM provides a WARM START for the optimizer. The actual optimization is still done by GA/PSO/NSGA-II with FEM or surrogate. The LLM's contribution is semantic reasoning: translating 'cantilever truss for 100kN load' into plausible initial area estimates, which can accelerate convergence. 
3.4.2 Modules 
* src/llm/client.py — Thin Anthropic API wrapper (uses ANTHROPIC_API_KEY env var) 
* src/llm/prompts.py — System and user prompt templates for truss design 
* src/llm/designer.py — propose_initial_design(problem_statement, benchmark) returns design vector 
* src/llm/evaluate_llm.py — Compare LLM-initialized vs random-initialized optimization convergence 
3.4.3 Prompt architecture 
System prompt describes truss mechanics briefly, instructs the LLM to output valid JSON with design variable estimates, and emphasizes it's providing a STARTING POINT for optimization. 
User prompt contains: benchmark geometry summary (node count, element count, loads), design variable count, bounds, and the natural-language design goal. 
Expected output: JSON with 'areas' array, 'reasoning' string explaining the heuristic, and 'confidence' scalar. Parse with strict JSON validation; fall back to midpoint of bounds if parsing fails. 
3.4.4 Evaluation methodology 
* For each benchmark, run 10 seeds of GA with random initialization (baseline) 
* Run 10 seeds of GA with LLM-initialized population (first 10% of population = LLM output + small noise) 
* Compare: generations-to-convergence, final weight, Pareto spread 
* Report mean +- std. If LLM initialization shows statistically significant speedup (t-test p<0.05), that's a thesis-worthy result. 
3.4.5 Cost and reliability management 
* Cache all Claude responses to disk (results/llm_cache/). Each problem statement hashed. Prevents re-billing for repeated runs. 
* Rate-limit API calls (max 10/min) to avoid overage. 
* Handle API failures gracefully: retry with exponential backoff, fall back to midpoint-of-bounds if all retries fail. 
* Log every API call with timestamp, tokens used, cost estimate. 
VALIDATION GATE — LAYER 5 
LLM-initialized optimization must converge to same or better final weight as random-initialized, in measurably fewer generations, on at least 3 of 4 benchmarks. Target: 15-30% reduction in generations-to-90%-convergence. 
 3.5 Layer 6 — IS 800:2007 Compliance 
Location: src/constraints/ 
Responsibility: Encode Indian Standard IS 800:2007 design checks as constraint functions. This is your civil-engineering anchor for Pathak and examiner credibility. 
3.5.1 Modules 
* src/constraints/is800_checks.py — Individual clause check functions 
* src/constraints/compliance.py — Composite full_is800_check() that all benchmarks call 
3.5.2 Clauses to implement 
* Clause 3.8 — Slenderness limits: lambda <= 180 for compression members, <= 400 for tension 
* Clause 6.2 — Tension design strength: T_dg = A * fy / gamma_m0 
* Clause 6.3 — Tension rupture of critical section: T_dn = 0.9 * An * fu / gamma_m1 
* Clause 7.1 — Compression design strength with buckling curves 
* Clause 7.5 — Effective length factors (pinned-pinned = 1.0 for trusses by default) 
* Clause 5.6.1 — Deflection limits: typically span / 300 for serviceability 
3.5.3 Wire-up to existing benchmarks 
Each benchmark's evaluate() method already returns raw FEM output (forces, stresses, displacements). Add a call to full_is800_check(...) that returns a dict with per-clause pass/fail flags plus an overall_ok boolean. The Problem class in src/algorithms/problem.py adds these as additional constraints. 
3.5.4 Non-goals for Layer 6 
* Do NOT implement connection design (bolts, welds) — out of scope, members are assumed pin-jointed 
* Do NOT implement lateral-torsional buckling — trusses have axial members only 
* Do NOT implement dynamic / seismic analysis — static loading only 
* Do NOT implement fatigue — static design 
VALIDATION GATE — LAYER 6 
For each benchmark's optimized result (from GA/PSO/NSGA-II), the full_is800_check must return overall_ok=True. Spot-check one member per benchmark manually using IS 800 formulas to verify the code matches the standard. 
 3.6 Layer 7 — Plotting and Visualization 
Location: src/plotting/ 
Responsibility: Produce all ~70 publication-quality figures for the thesis. 
3.6.1 Modules 
* src/plotting/style.py — Global matplotlib style (serif font, size 10, dpi 300, consistent color palette) 
* src/plotting/truss_plot.py — 2D and 3D truss geometry rendering, with optional deformed shape overlay 
* src/plotting/convergence.py — Algorithm convergence curves, single and comparative 
* src/plotting/pareto.py — Pareto front scatter plots, single and multi-benchmark dashboards 
* src/plotting/stress_distribution.py — Color-coded truss members by stress level 
* src/plotting/area_distribution.py — Bar charts of optimized design variables 
* src/plotting/surrogate_accuracy.py — Parity plots, R², calibration 
* src/plotting/rl_training.py — PPO episode reward curves, policy analysis 
* src/plotting/llm_comparison.py — LLM-initialized vs random-initialized convergence 
3.6.2 Figure naming convention 
Save to figures/fig_ _ _. (e.g., fig_4_2_b.png for Chapter 4 Section 2 Figure b). Always save both PNG (300dpi) and SVG for publication flexibility. 
3.6.3 Consistency rules 
* Same color for same algorithm across all figures (GA=blue, PSO=orange, NSGA-II=green, RL=red, LLM=purple) 
* Same marker shape per algorithm 
* Always include axis labels with units 
* Always include title (can be removed later for thesis body, retained for slides) 
* Always save, never plt.show() in batch scripts 
3.7 Layer 8 — UI and API 
Location: src/app/ (new directory) 
Responsibility: Streamlit web app that provides a live demo of the full system. Examiners watch truss optimization happen in real time during viva — this is the wow factor. 
3.7.1 Streamlit app layout 
* Sidebar: benchmark selector, algorithm selector, surrogate toggle, LLM-init toggle, run button 
* Main panel: real-time truss geometry (with Plotly for interactivity), convergence chart, Pareto front, IS 800 compliance checklist 
* Footer: export buttons (download optimized design JSON, PDF report, figures) 
3.7.2 FastAPI backend (optional, impressive) 
Expose the optimization pipeline as a REST API: 
* POST /optimize — accepts benchmark name, algorithm, options; returns result JSON 
* POST /llm-design — accepts natural language; returns LLM's initial design 
* GET /benchmarks — lists available benchmarks 
* Streamlit calls FastAPI internally; separation allows future deployment of API separately from UI 
3.7.3 Deployment (bonus) 
* Dockerfile with multi-stage build (Python + deps) 
* docker-compose.yml for local testing 
* README section on deployment 
VALIDATION GATE — LAYER 8 
End-to-end demo: user selects 10-bar + NSGA-II + surrogate + LLM-init, clicks Run, and within 10 seconds sees: converged Pareto front, optimized truss geometry, IS 800 compliance confirmation. This is the demo you show in viva. 
 Section 4 — Data Flow and Inter-Layer Contracts 
This section specifies exactly what data crosses each layer boundary. Cursor should treat these as strict interface contracts. 
4.1 End-to-end data flow for a typical optimization run 
[User] 
 ↓ selects benchmark + algorithm + options 
[Streamlit UI] Layer 8 
 ↓ POST /optimize 
[FastAPI] Layer 8 
 ↓ calls runner.run(...) 
[Runner] src/algorithms/runner.py 
 ↓ instantiates benchmark + problem 
[Benchmark] src/benchmarks/truss_*.py 
 ↓ builds Truss object 
[Truss] src/fem/truss.py ← Day 1 ✓ 
 ↓ (inner loop: for each candidate x) 
[Problem._evaluate] src/algorithms/problem.py 
 ├→ [FEM] truss.set_areas(x).solve() [slow path] 
 │ OR 
 └→ [Surrogate] surrogate.predict(x) [fast path] 
 ↓ returns weight, displacement, stresses 
[Problem._evaluate] 
 ├→ [IS 800 check] full_is800_check(...) 
 ↓ returns objectives F + constraints G 
[pymoo algorithm] evolves population 
 ↓ termination reached 
[Runner] aggregates results 
 ↓ 
[FastAPI] returns JSON 
 ↓ 
[Streamlit UI] renders figures 
 ↓ 
[User] sees results 
4.2 Data contracts (what goes where) 
4.2.1 Benchmark → Truss 
Benchmark.build_truss() returns a fully-configured Truss object with geometry, material, loads, and boundary conditions set. Areas start at midpoint of bounds (sensible default). 
4.2.2 Optimizer → Problem 
pymoo algorithm calls Problem._evaluate(x, out) with: 
* x: 1D numpy array of design variables, shape (n_design_vars,) 
* out: dict to populate with out['F'] (objectives) and out['G'] (constraints) 
4.2.3 Problem → FEM or Surrogate 
Problem calls either truss.set_areas(x).solve() (FEM path) or surrogate.predict(x) (fast path). Both return the same dict schema (weight, displacement, stresses, slenderness, feasible). This symmetry is critical for A/B testing. 
4.2.4 Problem → IS 800 checker 
Problem calls full_is800_check(truss_result) with the dict from the previous step. Returns a dict with per-clause flags. These flags become additional constraints in out['G']. 
4.2.5 Runner → Results storage 
Every run saves to results/ _ _seed.pkl with the standardized result schema (Section 3.1.5). The results aggregator in scripts/aggregate_results.py produces summary CSVs for Chapter 4 tables. 
4.3 Critical invariants 
* The Truss object from Day 1 is never re-implemented. All layers go through its set_areas + solve interface. 
* Design variable ordering is benchmark-specific and fixed. Never reshuffle. 
* Units are benchmark-specific (10-bar uses imperial, 25-bar same, per literature). Document units in every benchmark class and never convert silently. 
* Random seeds are passed explicitly at every stochastic layer. Never use np.random.seed() globally. 
Section 5 — Validation Gates (Definition of Done) 
Each component below has a validation gate. Until the gate passes, the component is not complete. Cursor should treat these as blocking tests. 
5.1 Summary table 
Layer 
 Component 
 Pass criterion 
 1 
 FEM engine 
 DONE — 21 tests green, 3-bar verified to 1e-21 precision 
 Pre-2 
 Benchmarks (10/25/72/200) 
 Published optimum areas reproduce published weight within 0.5% 
 2 
 GA / PSO / NSGA-II 
 GA+PSO within 2% of literature on 10-bar across 10 seeds; NSGA-II produces 20+ Pareto points 
 3 
 Neural surrogate 
 R² > 0.98 on weight held-out test; surrogate GA within 3% of literature AND 50x speedup 
 4 
 RL (PPO) agent 
 Agent within 5% of literature on 3+/4 benchmarks; training curves show monotonic improvement 
 5 
 LLM designer 
 LLM-init optimization converges 15-30% faster than random-init on 3+/4 benchmarks (p<0.05) 
 6 
 IS 800 compliance 
 All optimized results pass full_is800_check; manual spot-check of one member per benchmark 
 7 
 Plotting / figures 
 ~70 figures generated, consistent style, 300 dpi PNG + SVG versions present 
 8 
 UI / demo 
 Streamlit app runs end-to-end in <10s for 10-bar benchmark demo 
 5.2 Overall acceptance (thesis-level) 
* All Layer 1-8 validation gates pass 
* Results chapter tables populated with actual numerical results across all 4 benchmarks × all algorithms 
* At least 50 figures in figures/ 
* All 165 references in docs/citation_arsenal.md have been integrated into the thesis text 
* Thesis PDF compiles without errors 
* Plagiarism check below 10% similarity index 
* Code repository is clean, documented, reproducible 
Section 6 — Deliverables 
At submission, the repository contains these artifacts. 
6.1 Code artifacts 
* src/ — ~12,000 lines of Python across 8 layers 
* tests/ — ~100+ pytest tests, all green 
* scripts/ — batch runners, data generators, figure generators 
* results/ — pickled optimization results, trained model checkpoints (.pth), CSVs 
* figures/ — ~70 publication-quality PNG + SVG 
* docs/ — vision doc (this file), playbook, citation arsenal, API reference 
6.2 Thesis artifacts 
* thesis_writeup/main.docx (or .tex) — 140-page final thesis 
* thesis_writeup/abstract.pdf — 1-page abstract 
* thesis_writeup/slides.pptx — ~30-slide viva presentation 
* thesis_writeup/compliance_report.pdf — plagiarism check output 
6.3 Demo artifacts 
* src/app/ — Streamlit + FastAPI code 
* Dockerfile, docker-compose.yml — deployable 
* README.md — setup and run instructions 
* requirements.txt — pinned versions for reproducibility 
6.4 GitHub repository structure 
https://github.com/aryan/thesis-truss-optimization 
 ├── src/ # all layers 
 ├── tests/ # pytest suite 
 ├── scripts/ # batch runners 
 ├── results/ # gitignored 
 ├── figures/ # gitignored 
 ├── docs/ # planning + references 
 ├── thesis_writeup/ # thesis source + compiled PDF 
 ├── notebooks/ # Jupyter exploration (optional) 
 ├── Dockerfile 
 ├── docker-compose.yml 
 ├── pyproject.toml 
 ├── requirements.txt 
 ├── LICENSE (MIT) 
 └── README.md 
Section 7 — External References (When to Consult What) 
This section tells Cursor and you WHICH external resource to consult WHEN. Not a bibliography dump — a decision tree. 
7.1 When building the FEM layer (already done, for reference) 
* Logan, 'A First Course in the Finite Element Method' — truss element derivation 
* Cook et al., 'Concepts and Applications of Finite Element Analysis' — assembly algorithm 
* NumPy docs: https://numpy.org/doc/ 
* SciPy sparse docs: https://docs.scipy.org/doc/scipy/reference/sparse.html 
7.2 When encoding benchmarks 
* Sunar & Belegundu 1991 (AIAA J.) — 10-bar geometry, loads, and reference optimum 
* Schmit & Farshi 1974 (AIAA J.) — 10-bar and 25-bar original formulation 
* Venkayya 1971 (C&S) — 25-bar geometry, loading conditions, symmetry grouping 
* Fleury & Schmit 1980 (NASA CR-3226) — 72-bar tower 
* If geometry is ambiguous in any paper, cross-reference with pymoo's built-in 'Truss2D' problem and with He, Gilbert, Song 2019 open-source Python truss 
7.3 When building classical optimizers (Layer 2) 
* pymoo docs: https://pymoo.org/algorithms/ 
* pymoo paper: Blank & Deb 2020 (IEEE Access) — API design and examples 
* Deb 2001 textbook 'Multi-Objective Optimization using EAs' — theory 
* Deb, Pratap, Agarwal, Meyarivan 2002 (IEEE TEC) — NSGA-II canonical paper 
* Kennedy & Eberhart 1995 (IEEE ICNN) — PSO original 
* Coello Coello 2002 (CMAME) — constraint handling 
7.4 When building neural surrogate (Layer 3) 
* PyTorch docs: https://pytorch.org/docs/ 
* Goodfellow, Bengio, Courville 'Deep Learning' — MLP and training fundamentals 
* Queipo et al. 2005 (PAS) — surrogate-based optimization foundations 
* Forrester, Sóbester, Keane 2008 (Wiley) — surrogate modeling textbook 
* Persia et al. 2025 (Neural Computing Appl.) — NN surrogate for TO postprocessing 
* Kingma & Ba 2015 — Adam optimizer 
* For architecture inspiration: Sun et al. 2023 (Biomimetics) — DNN + GA for composite structures 
7.5 When building RL agent (Layer 4) 
* Sutton & Barto 2018 'Reinforcement Learning: An Introduction' — first 6 chapters 
* Schulman et al. 2017 (arXiv) — PPO paper 
* Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/ 
* Gymnasium docs: https://gymnasium.farama.org/ 
* Brown & Mueller 2022 (Materials & Design) — RL for topology optimization (closest precedent) 
* Hayashi & Ohsaki 2020 (Frontiers Built Env) — RL for TRUSS topology (direct inspiration) 
* Rochefort-Beaudoin et al. 2024 (arXiv 2407.07288) — SOgym RL environment 
7.6 When building LLM layer (Layer 5) 
* Anthropic API docs: https://docs.claude.com/ 
* Anthropic Python SDK: pip install anthropic 
* For prompt engineering: Anthropic's prompt engineering guide 
* Wei et al. 2022 (NeurIPS) — chain-of-thought prompting 
* Geng et al. 2024 (arXiv 2504.09754) — LLM + OpenSeesPy for structural analysis (closest precedent) 
* Makatura et al. 2024 (MIT Exploration) — LLMs for design and manufacturing 
7.7 When implementing IS 800 (Layer 6) 
* IS 800:2007 PDF (Bureau of Indian Standards) — primary source, download from BIS website 
* Subramanian, N. 2008 'Design of Steel Structures' (Oxford) — IS 800 explained for engineers 
* Duggal, S. K. 2014 'Limit State Design of Steel Structures' — worked examples 
* SP 6(1):1964 — standard steel section tables 
7.8 When building plotting (Layer 7) 
* matplotlib docs: https://matplotlib.org/stable/ 
* seaborn docs: https://seaborn.pydata.org/ 
* 'Trustworthy Online Controlled Experiments' (Kohavi et al.) for result presentation guidelines 
7.9 When building UI (Layer 8) 
* Streamlit docs: https://docs.streamlit.io/ 
* FastAPI docs: https://fastapi.tiangolo.com/ 
* Plotly docs: https://plotly.com/python/ (for interactive 3D trusses) 
7.10 When writing the thesis (Days 8-10) 
* docs/citation_arsenal.md — 165 references organized by tier and chapter 
* IIT BHU Civil department M.Tech template (ask Pathak's students for a reference thesis) 
* Zotero or Mendeley for reference management 
* Claude or ChatGPT for paraphrasing paragraphs after the user writes the structure and numbers 
Section 8 — Constraints, Non-Goals, and Guardrails 
Equally important as WHAT to build: WHAT NOT to build. Cursor should enforce these boundaries. 
8.1 Hard constraints 
* All code Python 3.10+ (repo uses 3.14) 
* All dependencies open-source (no commercial licenses: no SAP2000, ETABS, ABAQUS, MATLAB) 
* All results reproducible from seeded random states 
* Thesis total length ~140 pages (not 80, not 220) 
* Plagiarism similarity < 10% on Turnitin / iThenticate 
* All optimization results must satisfy IS 800:2007 constraints (not just published generic limits) 
8.2 Non-goals (out of scope) 
* Inventing a new optimization algorithm — integrative, not algorithmic novelty 
* Beating published benchmark records — validation (match within 1-2%) is the goal 
* Training foundation models — the thesis uses off-the-shelf Claude API 
* Dynamic / seismic / fatigue analysis — static loading only 
* Nonlinear geometric analysis — linear static only 
* Connection design (bolts, welds) — assumed pin-jointed 
* Cost optimization beyond weight — weight is the economic proxy 
* Construction-stage sequencing — design only 
8.3 Academic integrity guardrails 
* No text copied from prior theses or published papers. Citations only. 
* No results fabricated. Every number in the thesis comes from an actual run, logged and reproducible. 
* LLM (Claude) is used for: (a) providing starting designs to optimizers [Layer 5], (b) helping paraphrase thesis prose from user-written outlines. Claude is NOT used to: fabricate results, write entire chapters independently, replace real ML/FEM computation. 
* All 165 references in the bibliography are actual citable works. Any reference that cannot be verified with a DOI or ISBN is removed. 
8.4 Performance guardrails 
* Any single optimization run (500 generations, pop 100) must complete in under 10 minutes WITHOUT surrogate (on a standard laptop) 
* Surrogate-accelerated run must complete in under 30 seconds 
* RL training per benchmark must complete in under 15 minutes (using surrogate) 
* Streamlit demo interaction latency must be under 10 seconds 
* If any of these are violated, re-architect — do not accept slow code 
8.5 Scope creep alarms 
If any of the following arise during implementation, STOP and consult before building: 
* 'Let's also do topology optimization' — NO, size+shape only 
* 'Let's do a GNN instead of MLP for surrogate' — NO, MLP is sufficient 
* 'Let's fine-tune a small LLM instead of API' — NO, API is simpler and cheaper 
* 'Let's add genetic programming' — NO, three algorithms is enough 
* 'Let's deploy to AWS' — NO, local Docker is enough 
Scope creep is the #1 reason M.Tech theses run over deadline. Discipline is the feature. 
8.6 What Cursor should specifically NOT do 
* Do NOT modify src/fem/* without explicit request — it's validated and locked 
* Do NOT regenerate tests that already pass — add new tests, don't rewrite passing ones 
* Do NOT commit without running pytest 
* Do NOT commit results/ or figures/ artifacts (they're gitignored) 
* Do NOT change the sign convention (+ = tension, - = compression) anywhere 
* Do NOT use np.random.seed() globally (always pass seeds explicitly) 
* Do NOT use Jupyter cell magic in production code (notebooks/ is fine for exploration) 
Section 9 — TL;DR for Cursor Plan Mode 
If Cursor only reads one page of this document, it should be this one. 
What this thesis is 
An AI-powered multi-objective truss optimization framework. Python. Open-source. Four classical benchmarks. Three evolutionary algorithms. One neural surrogate for speedup. One RL agent as alternative. One LLM for warm-starting. IS 800:2007 compliance throughout. 
What's already built (Day 1) 
FEM engine (src/fem/*) is complete and validated to 1e-21 machine precision on the 3-bar canonical problem. 21 tests green. Do not rewrite. 
What to build next (in dependency order) 
6. Benchmarks (src/benchmarks/) — encode 10/25/72/200-bar geometries. Gate: literature optimum areas reproduce published weight. 
7. Problem class + GA + PSO + NSGA-II wrappers (src/algorithms/) — Gate: match literature optimum within 2% across 10 seeds. 
8. Dataset generation + neural surrogate (src/ml/) — Gate: R² > 0.98, 50x speedup. 
9. RL environment + PPO training (src/rl/) — Gate: within 5% of literature. 
10. LLM designer (src/llm/) — Gate: 15-30% speedup in convergence. 
11. IS 800 checks (src/constraints/) — Gate: all optimal results pass, spot-checked manually. 
12. Plotting (src/plotting/) — Gate: ~70 figures produced. 
13. Streamlit UI (src/app/) — Gate: end-to-end demo runs in <10 seconds. 
14. Thesis writing — ~140 pages referencing results/figures. 
The one-sentence validation criterion 
Every benchmark, optimized by every algorithm, using both FEM and surrogate, produces a design whose weight is within 2% of the published literature value AND satisfies all IS 800:2007 constraints — and the whole thing can be run end-to-end from a Streamlit UI. 
When in doubt 
* Prefer validation over novelty 
* Prefer simplicity over optimization 
* Prefer readable code over clever code 
* Prefer explicit seeds over global state 
* Prefer citing literature over inventing 
* When scope expands, push back — ask before adding 
 
 
— End of Vision Document — 
Feed this into Cursor plan mode. 
Ship the thesis. 
Page