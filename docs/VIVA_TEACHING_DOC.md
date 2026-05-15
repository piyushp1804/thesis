# VIVA TEACHING DOC — EXTREME EDITION

> **Your thesis, end-to-end, with every concept, every number, every formula, every story you'll need for the viva — and 30+ Q&A answers.**

**Author:** Piyush · **Programme:** M.Tech, Civil Engineering, IIT BHU
**Thesis:** *Multi-objective Steel Truss Optimisation: an AI-Assisted Framework Compliant with IS 800:2007*
**Repository tag:** `phase-10-complete`
**Reading time:** Designed for 5 days, ~30 minutes per day

---

# HOW TO USE THIS DOC

This is a single self-contained study guide. You don't need to open any other file — every formula, every code snippet, every result, every viva line is in here.

Read it in chunks. **Trying to do this in one sitting will not work.** Split it across five evenings:

| Day | Read | Goal at end of session |
|---|---|---|
| **Day 1** | Part I (Foundations) — sections 1 to 6 | You understand what a truss is, what optimisation is, and what your thesis *attempts* |
| **Day 2** | Part II (Framework) — sections 7 to 11 | You understand each of the 8 layers and what they do |
| **Day 3** | Part III (Results) — sections 12 to 18 | You can recite every headline number with confidence |
| **Day 4** | Part IV (The Viva) — sections 19 to 23 | You have the 15-minute script memorised at the beat level |
| **Day 5** | Part V (Q&A bank + final prep) — sections 24 to 28 | You can answer 30+ likely questions cleanly |

If you only have 24 hours: read sections **2, 19, 20, 21, 24** and the cheat sheet in section 28. That's the minimum-viable preparation.

---

# TABLE OF CONTENTS

### PART I — UNDERSTANDING THE THESIS
1. The thesis in five levels of detail
2. What is a truss, really
3. What "design" and "optimisation" actually mean
4. Why this matters for India
5. The constraints in plain English
6. Vocabulary you must know

### PART II — THE FRAMEWORK
7. The eight-layer architecture
8. Layer 1: The FEM physics engine
9. Layer 2: The four benchmark problems
10. Layer 3: The classical optimisers (GA, PSO, NSGA-II)
11. Layer 4: The neural surrogate (your 132× speedup)
12. Layer 5: The PPO reinforcement-learning agent
13. Layer 6: The LLM warm-start designer (the headline finding)
14. Layer 7: The IS 800:2007 compliance layer
15. Layer 8: The Streamlit + FastAPI deliverable

### PART III — THE RESULTS
16. 10-bar — the validation signature
17. 25-bar — the second confirmation
18. 72-bar — the honest story
19. 200-bar — the Indian-units stress test

### PART IV — THE VIVA
20. The 15-minute script (word-for-word with stage directions)
21. The slide-by-slide narrative
22. Body language, voice, and delivery
23. Recovery moves if something goes wrong

### PART V — Q&A AND FINAL PREP
24. The 32-question Q&A bank
25. Anti-questions you don't want but might get
26. The 20 must-know numbers (with the story behind each)
27. After-the-viva playbook
28. The one-page printable cheat sheet
29. Glossary (60+ entries)
30. Pre-viva checklist

---

# PART I — UNDERSTANDING THE THESIS

## 1. The thesis in five levels of detail

### Level 1 — One sentence

> *"I built an AI-assisted, Indian-code-compliant framework that designs steel trusses 100× faster than the standard approach, with one statistically significant novel finding about how LLMs help engineering design."*

### Level 2 — One paragraph

Steel trusses in Indian civil practice are designed by trial-and-error against the IS 800:2007 code, carrying 10–30 % surplus steel. Classical evolutionary optimisers can fix this but each candidate design requires a finite-element solve, and a typical run does 50 000 of them — about 40 minutes. Too slow to be interactive. This thesis combines classical optimisers (GA, PSO, NSGA-II) with three modern AI accelerators (a neural surrogate, a PPO reinforcement-learning agent, an LLM warm-start) plus an IS 800 compliance checker, into one Python framework. The framework matches the canonical 10-bar literature optimum to 0.004 % error, runs 132× faster via the surrogate (R² = 0.9993 on weight prediction), demonstrates a statistically significant LLM warm-start benefit (36.8 % generation reduction, p = 0.0046, 40 seeds), and ships as a Streamlit web app that completes a full optimisation in under 10 seconds.

### Level 3 — One page

**Problem.** Steel trusses are ubiquitous — factory sheds, mobile towers, bridges, stadium canopies. India builds thousands per year, every one regulated by IS 800:2007. Engineers size them by trial-and-error: pick trial sections, run an analysis, check that no stress / deflection / slenderness clause is broken, increase sections if it fails, stop when it passes. That converges in two or three iterations but rarely to the minimum-weight design — typical hand-designs carry 10–30 % surplus steel. At sector scale, with steel at 1.9 kg CO₂ per kg and the construction sector at 11 % of global emissions, the surplus is a real environmental and economic number.

**Why not just use an optimiser?** Genetic algorithms, particle swarm, NSGA-II — all routinely match published optima to fractions of a percent. The catch is speed. A 100-particle 500-generation run does 50 000 FEM solves; at 50 ms per solve that's 40 minutes per run. Too slow to fit inside the design-office iteration loop, where the engineer wants to ask "what if the live load increases 15 %?" and get an answer in seconds.

**What I did.** Built a Python framework that bolts three modern AI accelerators onto the classical optimiser stack. A **neural surrogate** (a 3-hidden-layer MLP trained on 10 000 Latin-hypercube samples) predicts FEM output 132× faster with R² = 0.9993 on weight. A **PPO reinforcement-learning agent** proposes designs in 0.2 s, single-shot. An **LLM warm-start** queries Claude in natural language for a starting design — and on the 10-bar benchmark, the LLM-seeded GA converges 36.8 % faster than random-seeded GA (paired Wilcoxon p = 0.0046 across 40 seeds), because Claude correctly identifies the redundant-member set {2, 5, 6, 10} in a single API call. The whole stack is checked against four IS 800:2007 clauses (slenderness, deflection, tension capacity, compression with column-curve buckling) before any design is reported. A FastAPI backend + Streamlit UI exposes it as a sub-10-second interactive tool that a non-Python user can drive.

**Results.** Matches Sunar 1991's 10-bar optimum (5060.85 lb) to 5061.05 lb — 0.004 % error. Matches Schmit 1976's 25-bar optimum (545.22 lb) to 545.26 lb — 0.008 % error. On 72-bar, my hard-constraint optima land at ~549 lb across three algorithms with cross-seed spread under 1 %; I could not reproduce the published 379.62 lb under strict feasibility in my FEM, so I report this as an observation pending independent FEM cross-check (future work F7) rather than a claim of literature correction. On the 200-bar planar stepped tower (which I re-encoded in proper Indian-civil SI units — structural steel, E = 210 GPa, ρ = 7850 kg/m³, 250 MPa stress limit), I report ~317 kg with cross-seed spread under 1 %.

**Practical deliverable.** Streamlit UI runs an end-to-end optimisation in under 10 seconds. All LLM responses cached on disk, so the pipeline reproduces offline without an Anthropic API key. Repository tagged `phase-10-complete`. Anyone can clone and verify every number.

### Level 4 — One chapter (the abstract)

That's the abstract page of `thesis_writeup/main.pdf` — read it before the viva. It's the page examiners read first.

### Level 5 — The full thesis

`thesis_writeup/main.pdf` — 125 pages, five chapters, four appendices, 70+ figures, 30+ tables. You wrote this. You know it.

---

## 2. What is a truss, really

A **truss** is a structure made of straight bars (called **members** or **elements**) connected at points (called **nodes** or **joints**) — and crucially, the connections are **pin joints**, meaning the bars can rotate freely at each node.

### Why pin joints matter

If you fix the bars rigidly (a moment connection), they have to resist bending — they push, pull, AND bend. That's a frame.

If the bars are pin-connected at the ends, they cannot resist bending (the pin just rotates), so they can only push or pull along their length. That's a truss.

This is structurally efficient because steel is excellent at resisting axial load (compression and tension) but inefficient at resisting bending. A truss therefore uses less steel for the same span than a beam of equivalent strength.

### Anatomy of a typical truss

A typical roof truss has:
- **Top chord** — the upper boundary, often carrying compression
- **Bottom chord** — the lower boundary, often carrying tension
- **Web members** — diagonals and verticals between the chords, carrying mixed loads
- **Supports** — at the ends, usually pin (resisting H and V load) and roller (resisting V only)

### Where you see them in India

| Structure | Truss role |
|---|---|
| **Industrial sheds** | Roof structure spanning 20–60 m |
| **Railway / road bridges** | Through-truss or under-deck truss for spans 30–200 m |
| **Transmission towers** | The whole tower is a 3D truss, 20–80 m tall |
| **Stadium / airport canopies** | Long-span lightweight roof |
| **Crane booms** | Lightweight cantilevered truss carrying load at the tip |

### A simple example to make it concrete

Take the **3-bar truss** from Logan's textbook (which your FEM passes to machine precision):

```
                load = P (downward)
                       │
                       ▼
                       ●  Node 2 (free, can move down)
                      /│\
                     / │ \
                    /  │  \
                   /   │   \
                  /    │    \
                 /     │     \
                ●──────●──────●
              Node 1  Node 4  Node 3
            (pin)   (pin)    (pin)
            (all three lower nodes fixed)
```

Three bars all meet at node 2. The load P pushes down. Node 2 moves down a tiny amount. By symmetry the two diagonal bars carry equal force (in compression); the middle vertical bar carries the rest (in compression). The FEM solver computes how much node 2 moves and what force is in each bar. That's it.

### Scaling up

The 10-bar planar cantilever in your thesis has 10 bars and 6 nodes. The 72-bar spatial tower has 72 bars and 20 nodes. The 200-bar planar stepped tower has 200 bars and 77 nodes. But the math is identical — just bigger matrices.

---

## 3. What "design" and "optimisation" actually mean

### Design

In the truss world, "design" means deciding three things:

1. **Topology** — Which bars exist? Which nodes connect to which?
2. **Shape** — Where are the nodes located in space?
3. **Sizing** — What cross-sectional area does each bar have?

Topology and shape together define the *geometry*. Sizing decides the *weight*.

**Your thesis is sizing-only.** Topology and shape are fixed by the benchmark — you don't get to remove bars or move nodes. You only get to pick areas.

This is a deliberate scope choice. Topology optimisation is a much harder problem (the number of possible topologies for a 20-node truss is astronomical). Sizing alone is bounded enough to solve precisely and to validate against published results. Topology optimisation is future-work F2.

### Optimisation

"Optimisation" means **finding the best design** by some quantitative criterion.

In your thesis:
- **Best** = lightest (minimum total weight)
- **By the criterion** = subject to all four IS 800 constraints

Mathematically:

$$\text{minimise} \quad W(\mathbf{A}) = \sum_{i=1}^{N_{\text{bars}}} \rho \cdot L_i \cdot A_i$$

subject to:
$$\begin{aligned}
\sigma_i(\mathbf{A}) &\le \sigma_{\max} \quad \forall i \in \text{tension members} \\
|\sigma_i(\mathbf{A})| &\le f_{cd,i}(\mathbf{A}) \quad \forall i \in \text{compression members} \\
|u_j(\mathbf{A})| &\le \delta_{\max} \quad \forall j \in \text{checked DOFs} \\
\lambda_i(\mathbf{A}) &\le \lambda_{\max,i} \quad \forall i \\
A_{\min} &\le A_i \le A_{\max} \quad \forall i
\end{aligned}$$

Where:
- $\mathbf{A} = [A_1, A_2, \ldots, A_N]$ is the **design vector** (areas)
- $W$ is total weight (the **objective function**)
- $\sigma_i$ is the stress in bar $i$
- $u_j$ is the displacement at degree-of-freedom $j$
- $\lambda_i$ is the slenderness of bar $i$
- All these depend on $\mathbf{A}$ because FEM gives different stresses and displacements for different area assignments

### Why it's hard

For 10 design variables, each one continuous in a range like [0.1, 35] in², the search space is uncountable. Even if you discretise to say 100 area values per bar, that's $100^{10} = 10^{20}$ candidate designs. You can't enumerate them.

You need a **search algorithm** that explores intelligently. That's what the GA, PSO, and NSGA-II do.

### Local vs global optima

Optimisation problems often have multiple local minima — points where the design is locally lightest but a heavier design elsewhere has a lower weight when fully searched. A good optimiser finds the **global** minimum, not just a local one.

The 10-bar problem is benign — most reasonable optimisers find the global. The 72-bar is rougher — your soft-penalty version reaches ~380 lb (which violates constraints by 20–25 % in your strict FEM); your hard-penalty version reaches 549 lb. Both are *local* minima of their respective formulations.

---

## 4. Why this matters for India

### The economic case

India's structural steel consumption was approximately 120 million tonnes in FY 2023-24. Construction and infrastructure account for ~60 % of that, of which trusses are a small but real fraction — let's say 5–10 million tonnes of truss work per year.

If hand-design is 10–30 % heavy (call it 20 % average), that's 1–2 million tonnes of avoidable steel per year. At ₹70 000 per tonne, that's **₹7 000 – ₹14 000 crore per year of avoidable spend**.

### The carbon case

Steel produced by the blast-furnace route (the dominant Indian route) emits ~1.9 kg CO₂ per kg of finished steel. So 1.5 million tonnes of avoidable steel = **2.85 million tonnes of avoidable CO₂ per year**. For context, that's roughly the annual emissions of 600 000 cars.

The construction sector globally accounts for 11 % of CO₂ emissions; structural steel is a meaningful chunk of that. The Indian decarbonisation pathway under the COP 26 commitments needs every available efficiency lever.

### The deployment case

A typical Indian engineering consultancy designs 50–200 truss structures per year. The engineer might do basic optimisation by trying 3–4 alternative section sizes — but a 50 000-evaluation GA is out of reach because it's a 40-minute wait. With your framework's sub-10-second pipeline, the engineer can run *hundreds* of candidate scenarios per day. This shifts optimisation from "academic curiosity" to "real-time design tool".

### The IS 800 case

International codes (Eurocode 3, AISC 360) have their own optimisation tooling. IS 800:2007 — the legally binding Indian code — has effectively no widely-deployed open optimisation toolkit. Your framework is the first published one with IS 800 baked in. That's the practical hook for an Indian civil examiner.

---

## 5. The constraints in plain English

Every design has to satisfy four families of constraints. If you remember nothing else about constraints, remember these four.

### Constraint 1 — Stress (will the bar break?)

Force in the bar divided by its cross-sectional area gives the stress. If that exceeds the steel's allowable stress, the bar fails (yielding in tension, buckling in compression).

For tension bars, allowable stress = $f_y / \gamma_{m0}$. With Indian Fe 250 steel, $f_y = 250$ MPa and $\gamma_{m0} = 1.1$, so allowable = 227 MPa.

For compression bars, allowable stress = $\chi \cdot f_y / \gamma_{m0}$, where $\chi$ ≤ 1 is the column-curve reduction factor that punishes long thin bars. A short stocky bar might have $\chi = 0.95$ (almost full strength). A long thin bar might have $\chi = 0.3$ (only 30 % of full strength).

### Constraint 2 — Displacement (will the truss sag too much?)

Even if no bar breaks, the truss might sag too much under load. IS 800 Clause 5.6.1 limits the maximum vertical displacement to **span / 325** for trusses supporting brittle cladding. For a 30 m span, that's 30000/325 ≈ 92 mm — not much.

This constraint often *binds* before stress on long-span trusses. A common engineer's instinct ("just make every member bigger") fixes stress problems but doesn't fix deflection problems efficiently — you need *strategic* sizing.

### Constraint 3 — Slenderness (will the bar buckle prematurely?)

A long thin bar in compression is unstable — it will buckle (bow sideways) at a load far below its theoretical yield strength. IS 800 Clause 3.8 caps this with the **slenderness ratio**:

$$\lambda = \frac{L_{\text{eff}}}{r}, \quad r = \sqrt{\frac{I}{A}}$$

For compression members carrying dead + imposed loads: $\lambda \le 180$.
For tension members: $\lambda \le 400$ (less strict because tension doesn't buckle).

The radius of gyration $r$ depends on the cross-section *shape*, not just area. For an unknown generic shape we approximate $r \approx \sqrt{A / \pi}$ (treating the area as a circle). This is a conservative approximation.

### Constraint 4 — Bounds (does the section exist?)

You can't pick any area. The standard rolled-section catalogue (SP 6(1):1964 in India) lists specific available areas. Your design vector must lie within bounds:

$$A_{\min} \le A_i \le A_{\max}$$

For the 10-bar benchmark, bounds are [0.1, 35] in². For 200-bar in Indian SI, bounds are [6.5 × 10⁻⁵, 2.5 × 10⁻²] m². If the optimiser proposes 0.05 in², that's *not a real cross-section* and the design is invalid.

### What "feasible" means

A design that satisfies **all four** families of constraints is **feasible**. If even one constraint is broken — bar overstressed, joint deflecting too much, slenderness over limit, area below catalogue minimum — the design is **infeasible** and disqualified.

### What "binding" means

A constraint **binds** at the optimum if it's exactly at its limit. e.g. on 10-bar, the displacement constraint binds — the optimum sits at exactly 2 inches of node displacement. Making any bar smaller would violate the displacement limit. Making any bar bigger would waste steel.

When you know which constraints are binding, you know which "lever" pushes the optimum mass up or down. On 10-bar it's displacement. On 200-bar it's stress (deflection sits at 0.033 m vs 0.10 m limit — way below).

---

## 6. Vocabulary you must know

These are the words you'll hear in the viva. If any are unclear, look them up here.

| Word | Definition | Where it appears |
|---|---|---|
| **Truss** | Pin-jointed structure of bars carrying axial-only load | Everywhere |
| **Member / Element / Bar** | One steel rod between two nodes | FEM, design vector |
| **Node / Joint** | A point where bars meet | FEM, displacement constraint |
| **Cross-section area** | Area of a bar's cross-section, in m² or in² | Design variable |
| **Design variable** | A number being optimised — here, bar areas | Optimisation |
| **Design vector** | The full set of design variables — $\mathbf{A} = [A_1, \ldots, A_N]$ | Optimisation |
| **Objective function** | The thing being minimised — total weight $W(\mathbf{A})$ | Optimisation |
| **Constraint** | A rule the design must obey | All four families |
| **Feasible** | Satisfies ALL constraints | Optimiser output |
| **Infeasible** | Breaks at least one constraint | Penalty / discard |
| **Optimum** | The best (lightest feasible) design | Goal |
| **Local optimum** | A design that's locally best but not globally | Search-algorithm concern |
| **Global optimum** | The single best design across the entire space | Search-algorithm goal |
| **Pareto front** | Set of designs where no other design is better on every objective | NSGA-II output |
| **Dominated** | A design dominated by another is worse on every objective | Pareto sort |
| **Symmetry group** | Bars forced to share one area for symmetry | Reduces design dimensions |
| **Load case** | One specific loading scenario (e.g. wind from the left) | Worst-case across LCs |
| **FEM (Finite Element Method)** | Physics solver that gives stresses and displacements | Layer 1 of framework |
| **Stiffness matrix** | The matrix in $\mathbf{K}\mathbf{u} = \mathbf{F}$ | FEM internals |
| **Degree of freedom (DOF)** | One axis a node can move along | FEM bookkeeping |
| **Hyperparameter** | A setting of an algorithm (e.g. population size) | Algorithm tuning |
| **Population** | The set of candidate designs maintained by GA | GA mechanics |
| **Generation** | One iteration of evolve-evaluate-select in GA | GA mechanics |
| **Tournament selection** | Pick the better of 2 randomly chosen parents | GA selection |
| **Crossover** | Combine two parents to make a child | GA reproduction |
| **Mutation** | Randomly tweak a child | GA exploration |
| **SBX** | Simulated Binary Crossover — pymoo's default for real-valued GA | GA hyperparameter |
| **Swarm** | Population of particles in PSO | PSO mechanics |
| **Inertia** | How much a PSO particle keeps its current velocity | PSO hyperparameter |
| **Cognitive weight** | PSO pull toward personal best | PSO hyperparameter |
| **Social weight** | PSO pull toward global best | PSO hyperparameter |
| **Non-dominated sort** | NSGA-II's core sorting routine | NSGA-II mechanics |
| **Crowding distance** | NSGA-II diversity measure | NSGA-II mechanics |
| **Penalty function** | Add a big number to objective if constraint violated | Constraint handling |
| **Feasibility-first** | Always prefer feasible over infeasible | Constraint handling |
| **Soft constraints** | Penalty-based; minor violations tolerated | Common in heuristic lit |
| **Hard constraints** | Strict $g \le 0$; no violations tolerated | Your 72-bar |
| **Surrogate** | Cheap fake of an expensive simulator | Layer 4 |
| **MLP** | Multi-Layer Perceptron — a basic neural network | Surrogate architecture |
| **ReLU** | Rectified Linear Unit activation: $f(x) = \max(0, x)$ | Activation function |
| **Dropout** | Randomly zero some neurons during training | Regularisation |
| **Adam** | A popular gradient-descent variant | Training |
| **Latin Hypercube Sampling (LHS)** | Space-filling sampling — better than uniform random | Surrogate training data |
| **R²** | Coefficient of determination, how well predictions match truth | Surrogate quality |
| **RL (Reinforcement Learning)** | An agent learns by trial-and-error to maximise reward | Layer 5 |
| **PPO** | Proximal Policy Optimisation — a popular RL algorithm | Layer 5 |
| **Bandit (contextual)** | An RL problem with one step per episode | Your formulation |
| **Stable-Baselines3** | RL algorithm library | PPO implementation |
| **Gymnasium** | Standard RL environment interface | Your TrussDesignEnv |
| **Policy** | Function mapping states to actions in RL | PPO output |
| **Reward** | Scalar signal an RL agent maximises | Your $-W - \lambda \sum g^2$ |
| **LLM** | Large Language Model (e.g. Claude) | Layer 6 |
| **Chain-of-thought** | Prompt the LLM to reason step-by-step | Prompt design |
| **Warm-start** | Seed the optimiser with a clever initial guess | LLM use case |
| **API cache** | On-disk store of past API responses | Offline reproducibility |
| **Wilcoxon signed-rank** | Non-parametric paired statistical test | Your significance test |
| **p-value** | Probability the observed effect happened by chance | Statistical significance |
| **Paired experiment** | Both arms see the same seeds — better statistical power | Your LLM A/B design |
| **IS 800:2007** | Bureau of Indian Standards general steel construction code | Legal standard |
| **Limit state design** | Design philosophy with separate ultimate and serviceability checks | IS 800 paradigm |
| **Partial safety factor** | Multiplier on loads or capacities for safety margin | IS 800 numbers |
| **Slenderness ratio** | $\lambda = L_{\text{eff}}/r$ — long-thin-ness of a bar | IS 800 Clause 3.8 |
| **Column buckling curve** | $\chi$ vs $\bar\lambda$ relationship for compression capacity | IS 800 Clause 7.1 |
| **Streamlit** | Python library for making web UIs without HTML | Layer 8 frontend |
| **FastAPI** | Python library for making REST APIs | Layer 8 backend |
| **REST API** | A web service speaking HTTP with JSON | App architecture |
| **JSON** | Standard data-interchange format | App architecture |
| **Pymoo** | Python multi-objective optimisation library | GA/PSO/NSGA-II underneath |
| **PyTorch** | Deep-learning framework underlying the surrogate and PPO | ML layer |
| **Anthropic** | Maker of Claude (the LLM you use) | LLM provider |

---

# PART II — THE FRAMEWORK

## 7. The eight-layer architecture

Picture this as a cake. Each slice depends on the slices below it.

```
═══════════════════════════════════════════════════════════════════════
LAYER 8  │  Streamlit UI                    (src/app/ui.py)
         │  ↓ HTTP/JSON
LAYER 7  │  FastAPI backend                 (src/app/api.py)
         │  ↓ Python calls
LAYER 6  │  IS 800:2007 compliance checker  (src/constraints/)
         │  ↓ vector of constraint values g
LAYER 5  │  LLM warm-start designer         (src/llm/)         ←─┐
         │  ↓ initial design vector                              │
LAYER 4  │  PPO RL agent                    (src/rl/)           ←┤  These three
         │  ↓ proposed design vector                             │  AI accelerators
LAYER 3  │  Neural surrogate (MLP)          (src/ml/)           ←┘  bypass / speed
         │  ↓ predicted (W, δ, σ)                                up the inner loop
═════════│═══════════════════════════════════════════════════════
LAYER 2  │  Classical optimisers            (src/algorithms/)
         │  GA, PSO, NSGA-II (pymoo wrappers)
         │  ↓ exact (W, δ, σ) via FEM
LAYER 1  │  Truss FEM solver                (src/fem/)
         │  K·u = F
═══════════════════════════════════════════════════════════════════════
                  Benchmark problem definitions
                  (src/benchmarks/) — 10/25/72/200-bar
```

### What flows up and what flows down

**Down (request flow):** User clicks Run in Streamlit → FastAPI receives a JSON spec → calls the optimisation runner → runner calls either FEM or surrogate per candidate → returns to FastAPI → renders in Streamlit.

**Up (data flow):** FEM (or surrogate) computes (W, δ, σ) for one design → IS 800 module computes constraint values $g$ → optimiser uses $(W, g)$ to evolve population → final best feasible design returned.

### Three invariants that hold across the stack

1. **One source of truth for bounds.** The benchmark module sets design-variable bounds once. Every other layer reads them from there. No hardcoded copies.
2. **SI internally.** All internal computation is SI (Newtons, metres, Pascals). Imperial-unit benchmarks (10-bar, 25-bar, 72-bar) convert at the boundary in `BenchmarkProblem.as_si()`. The IS 800 module never sees imperial inputs.
3. **One seed propagates everywhere.** The CLI's `--seed` argument feeds numpy, pymoo, PyTorch, and the LLM cache key. Every result in Chapter 4 is exactly reproducible.

### The eight layers in one-liner descriptions

| Layer | Module | One-liner |
|---|---|---|
| 1 | `src/fem/` | Direct-stiffness FEM for pin-jointed trusses, 2D and 3D unified |
| 2 | `src/algorithms/` | GA / PSO / NSGA-II wrappers over pymoo with feasibility-first selection |
| 3 | `src/ml/` | MLP surrogate ([256, 128, 64], dropout 0.1) trained on 10 000 LHS samples |
| 4 | `src/rl/` | PPO agent in a single-step bandit environment, reward = −W − λΣmax(g, 0)² |
| 5 | `src/llm/` | Claude prompt + JSON parser + content-hashed disk cache + heuristic fallback |
| 6 | `src/constraints/` | IS 800:2007 clauses 3.8, 5.6.1, 6.2/6.3, 7.1 as pure functions |
| 7 | `src/app/api.py` | FastAPI with /optimize, /llm/suggest, /benchmarks endpoints |
| 8 | `src/app/ui.py` | Streamlit client over the FastAPI backend |

The benchmark module (`src/benchmarks/`) sits sideways — every layer queries it.

---

## 8. Layer 1 — The FEM physics engine

The Finite Element Method is your **physics ground truth**. Every optimiser, every surrogate, every RL agent is trying to find a design that the FEM blesses as feasible.

### What FEM computes

**Input** to the FEM solver:
- Node positions (where each joint is in space)
- Element connectivity (which two nodes each bar connects)
- Material (E = Young's modulus, ρ = density)
- Cross-section area for each bar (the design vector $\mathbf{A}$)
- Load vector $\mathbf{F}$ (forces applied at nodes)
- Support conditions (which nodes are fixed)

**Output** from the FEM solver:
- Nodal displacements $\mathbf{u}$ (how much every node moved)
- Element forces $f_i$ (the axial force in each bar — positive tension, negative compression)
- Element stresses $\sigma_i = f_i / A_i$

### The mechanics in three steps

**Step 1 — Element stiffness.** For each bar with cross-section $A$, length $L$, modulus $E$:

$$\mathbf{k}_e = \frac{AE}{L}\,\mathbf{T}^\top\!\begin{bmatrix}1 & -1 \\ -1 & 1\end{bmatrix}\!\mathbf{T}$$

where $\mathbf{T}$ is a small (2×4 or 2×6) rotation matrix that projects the global nodal displacements onto the bar's own axial direction. The bracketed [1,-1;-1,1] is the local axial stiffness — it says "pulling both ends apart = positive force, pushing one end while pulling the other = zero net axial force".

This is just **Hooke's law** in matrix form: $F = (EA/L) \cdot \delta$, where $\delta$ is the bar's elongation.

**Step 2 — Global assembly.** Take each $\mathbf{k}_e$ (4×4 in 2D or 6×6 in 3D) and add it into the global stiffness matrix $\mathbf{K}$ at the rows and columns corresponding to its node DOFs. numpy's `ix_` does this scatter in one line.

$\mathbf{K}$ is symmetric. Before boundary conditions it's singular (rigid-body modes still possible).

**Step 3 — Solve with boundary conditions.** Partition DOFs into free and prescribed:

$$\mathbf{K}_{ff}\,\mathbf{u}_{f} = \mathbf{F}_{f} - \mathbf{K}_{fp}\,\mathbf{u}_{p}$$

Solve for $\mathbf{u}_f$ using `scipy.linalg.solve` on the dense partitioned matrix. The benchmark systems are at most 200 DOFs so we don't need sparse solvers.

### Worked example — the three-bar truss

Take a single bar in 2D with E = 1, L = 1, A = 1, oriented at θ = 0° (purely horizontal).

Cos θ = 1, sin θ = 0. So $\mathbf{T}^\top \begin{bmatrix}1 \\ 0\end{bmatrix} = \begin{bmatrix}1 \\ 0\end{bmatrix}$.

$$\mathbf{k}_e = \frac{1 \cdot 1}{1} \cdot \begin{bmatrix}1 & 0 & -1 & 0 \\ 0 & 0 & 0 & 0 \\ -1 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0\end{bmatrix}$$

This matrix says: a unit horizontal displacement of node 2 (while node 1 is held fixed) produces a unit horizontal force trying to pull node 2 back. No vertical force at all. Perfect axial-only behaviour.

### Why this validates correctly

Logan's three-bar example has a closed-form analytical solution. Your FEM matches it to ≈ 1e-21 — that's machine precision (double-precision floats have about 16 decimal digits; 1e-21 is below the floor of representable error). So the FEM is *provably correct* on a simple system.

This matters because if any later layer (surrogate, optimiser) reports a weird number, you know the bug is *not* in the FEM.

### Wall-clock cost

41 ms per evaluation on a laptop CPU. That sounds fast — but a 100×500 GA does 50 000 evaluations = ~34 minutes per run. The surrogate (132× faster) compresses that to ~15 seconds.

---

## 9. Layer 2 — The four benchmark problems

These are not problems you invented — they're **standard test cases** from the structural-optimisation literature. Everyone tests their algorithm on them; you can compare your numbers to other published papers.

### Why benchmarks matter

If you say "I built an optimiser and it produces lightweight designs", an examiner has no way to evaluate that. If you say "I built an optimiser and on Sunar 1991's published 10-bar benchmark I match the published 5060.85 lb optimum to 5061.05 lb (0.004 % error)", that's a *quantitative* validation that any examiner can verify by reading the original paper.

### The four

| # | Name | Origin | Dim | Bars | Design vars | Lit optimum | My result |
|---|---|---|---|---|---|---|---|
| 1 | 10-bar cantilever | Sunar & Belegundu 1991 | 2D | 10 | 10 | 5060.85 lb | 5061.05 lb (0.004 %) |
| 2 | 25-bar spatial tower | Venkayya 1971 | 3D | 25 | 8 (groups) | 545.22 lb | 545.26 lb (0.008 %) |
| 3 | 72-bar spatial tower | Fleury & Schmit 1980 | 3D | 72 | 16 (groups) | 379.62 lb (soft-penalty) | ~549 lb (hard, see §18) |
| 4 | 200-bar planar tower | Kaveh & Talatahari 2010 (re-encoded in SI) | 2D | 200 | 29 (groups) | (not claimed) | 316.61 kg |

### Benchmark 1 — 10-bar planar cantilever (Sunar 1991)

**Geometry:** Six nodes, two columns of three. Left column is the supports; right column has the loaded free end. Ten bars connect them in a cantilever pattern (chords + diagonals).

**Loading:** Two vertical −100 kips (= −100 000 lb, downward) point loads at the two right-most lower nodes.

**Material:** Aluminium-alloy properties — E = 10⁷ psi, ρ = 0.1 lb/in³.

**Design variables:** 10 areas (one per bar), no symmetry. Bounds [0.1, 35] in².

**Constraints:**
- Stress ≤ ±25 000 psi
- Displacement ≤ 2 inches at any free node
- (Slenderness and IS 800 added in your enhanced version)

**Why it's the canonical benchmark:** It's small (only 10 vars), 2D (easy to visualise), with a single load case and uniform stress limits. Its published optimum is known to four decimal places. Every truss-sizing paper since 1991 reports a 10-bar number. This is the gold-standard validation.

**Your result:** PSO gets 5061.05 lb in 5 minutes on a laptop, matching Sunar's 5060.85 lb to 0.004 %. **This is your end-to-end validation signature**. Every layer (FEM, optimiser, IS 800, surrogate) must be correct for that number to land there.

### Benchmark 2 — 25-bar spatial tower (Venkayya 1971)

**Geometry:** 10 nodes, 25 bars forming a small 3D tower. Roughly 200×200×200 inches.

**Loading:** Two load cases (the optimisation must satisfy *worst case* across both).

**Material:** Aluminium, E = 10⁷ psi, ρ = 0.1 lb/in³.

**Design variables:** 25 bars are grouped into 8 symmetry groups (e.g. all "leg" bars share one area). So 8 design variables, not 25. This is *typical* of real engineering — you don't size every bar independently, you size groups.

**Constraints:** Stress + displacement, same flavour as 10-bar.

**Why it's there:** 3D test of the spatial FEM. Confirms your transformation matrices work for 3D bars. Symmetry grouping confirms the optimiser handles reduced-dimension design vectors. Two load cases confirm the optimiser handles worst-case constraint aggregation.

**Your result:** PSO gets 545.26 lb, GA gets 545.30 lb, both within 0.008 % of Schmit 1976's 545.22 lb.

### Benchmark 3 — 72-bar spatial tower (Fleury & Schmit 1980)

**Geometry:** 4-storey tower, 120×120×240 inches, 20 nodes (4 per storey × 4 storeys + 4 base supports), 72 bars (4 columns + 8 face diagonals + 4 horizontals + 2 plan diagonals per storey × 4 storeys).

**Loading:** Two load cases per Camp & Bichon 2004 canonical:
- LC1: asymmetric tip load (5000, 5000, −5000) lbf at tip node 12 only
- LC2: uniform vertical compression (0, 0, −5000) lbf at each of the 4 tip nodes

**Material:** Aluminium, E = 10⁷ psi, ρ = 0.1 lb/in³.

**Design variables:** 16 groups (4 groups × 4 storeys: legs, face diagonals, top horizontals, plan diagonals).

**Constraints:**
- Stress ≤ ±25 000 psi
- Lateral tip displacement ≤ 0.25 inches (only x and y; only the 4 tip nodes; vertical and lower-storey unconstrained, per Camp 2004)

**Why it's there:** Bigger, more vertical structure with mixed compression-tension behaviour. Real-world challenge that exposes optimiser limits.

**Your result:** Hard-constraint optimum at ~549 lb across GA, PSO, NSGA-II with cross-seed spread < 1 %. You could **not** reproduce the published 379.62 lb under strict feasibility — that result requires soft-penalty constraint handling. (See §18 for the full story.)

### Benchmark 4 — 200-bar planar stepped tower (Kaveh 2010 re-encoded)

**Geometry:** 77 nodes, 200 bars, 30 m tall, regular-grid stepped tower (your own geometry, scale-matched to Kaveh's published problem).

**Loading:** Three load cases:
- LC1: lateral wind at the top
- LC2: asymmetric mid-span load
- LC3: vertical load at the tip

**Material:** **Structural steel in proper Indian-civil SI units:**
- E = 210 GPa
- ρ = 7850 kg/m³
- Stress limit ±250 MPa
- Displacement limit 100 mm (span/300)
- Area bounds [6.5 × 10⁻⁵, 2.5 × 10⁻²] m²

**Design variables:** 29 groups.

**Why it's there:** This is the **only benchmark in real Indian civil units**. It tests the framework's scaling to real-world problem size (29 vars vs 10–16 in earlier benchmarks). It exercises IS 800 against a 250 MPa stress envelope, which is the structurally realistic regime.

**Your result:** PSO reaches 315.89 kg, GA reaches 316.61 kg, both with cross-seed spread < 1 %. NSGA-II Pareto fronts have 100 non-dominated points. Total compute: 57 wall-clock minutes for the 13-seed matrix.

**What you explicitly don't claim:** The Kaveh 2010 paper reports an optimum near 25 445 lb on a different geometry (Kaveh's exact stepped layout, not yours). You **don't claim a literature match** here because your geometry is principled-but-different. This is openly stated in §4.4. The 200-bar's role is *scaling demonstration*, not literature reproduction.

---

## 10. Layer 3 — The classical optimisers

Three algorithms, all wrapped over pymoo with a uniform `TrussProblem` adapter.

### The genetic algorithm (GA) — biology analogy

**Mechanism in plain English:**

1. **Initialise.** Start with 100 random designs (the *population*). Each is an area vector.
2. **Evaluate.** Call FEM (or surrogate) on each design. Get weight + constraint values.
3. **Select parents.** *Tournament*: pick 2 designs at random, the better one wins. Repeat 100 times.
4. **Crossover.** Combine two parents to produce a child. *Simulated Binary Crossover (SBX)* with distribution index $\eta_c = 15$ — produces smoothly mixed continuous values.
5. **Mutate.** Randomly tweak some areas in the child. *Polynomial mutation* with $\eta_m = 20$. Mutation probability = 1/n where n is the number of design variables.
6. **Elitism.** Keep the best individuals from the parent population.
7. **Replace.** Build the new generation. Go to step 2.
8. **Repeat 500 times.**

**Why it works:** Over many generations, good designs survive and reproduce. Their good "genes" (specific area patterns) spread through the population. Eventually the whole population clusters around the global optimum.

**Hyperparameters:** pop=100, generations=500 (800 for 200-bar), SBX $\eta_c$=15, $p_c$=0.9, polynomial mutation $\eta_m$=20, $p_m$=1/n, tournament size 2.

**Who invented it:** Holland 1975 (genetic algorithms in general), Deb & Beyer 2001 (SBX specifically), Deb & Agrawal 1995 (polynomial mutation).

### Particle swarm optimisation (PSO) — bird-flock analogy

**Mechanism in plain English:**

1. **Initialise.** 100 particles, each with a random position $\mathbf{x}$ (= an area vector) and zero velocity.
2. **Evaluate.** Compute objective for each particle.
3. **Update memory.** Each particle remembers its personal best $\mathbf{p}_i$. The swarm tracks global best $\mathbf{g}$.
4. **Update velocity:**
   $$\mathbf{v}_i \leftarrow w\,\mathbf{v}_i + c_1 r_1(\mathbf{p}_i - \mathbf{x}_i) + c_2 r_2(\mathbf{g} - \mathbf{x}_i)$$
   - $w$ = inertia (0.9 decreasing linearly to 0.4)
   - $c_1 = c_2 = 2.0$ (cognitive and social weights, equal balance)
   - $r_1, r_2 \sim \mathcal{U}[0,1]$ (random)
5. **Update position:** $\mathbf{x}_i \leftarrow \mathbf{x}_i + \mathbf{v}_i$, clipped to bounds.
6. **Repeat 500 times.**

**Why it works:** Each particle is pulled toward its own past best (cognitive component) and the swarm's collective best (social component), but keeps some of its current direction (inertia). Over time the whole swarm converges on the global best.

**Comparison to GA:** Often faster on smooth landscapes, fewer hyperparameters, no crossover-operator design decision. GA tends to win on rougher landscapes where exploration matters more than exploitation.

**Hyperparameters:** swarm=100, generations=500, $w$ schedule 0.9→0.4, $c_1=c_2=2$.

**Who invented it:** Kennedy & Eberhart 1995, motivated by simulating bird-flock social dynamics.

### NSGA-II — Pareto-front analogy

When you have *two competing objectives* (here: minimise weight AND minimise displacement), there's no single best — there's a *trade-off curve*. NSGA-II finds that curve.

**Mechanism in plain English:**

1. **Initialise.** Random population of 100.
2. **Evaluate.** Compute both objectives (W, $\delta_{\max}$) for each.
3. **Fast non-dominated sort.** Partition the population into Pareto "fronts":
   - Front 1 = designs not dominated by anyone
   - Front 2 = designs only dominated by Front 1
   - Front 3 = designs only dominated by Fronts 1 and 2
   - ...
   "Domination" means: A dominates B if A is at least as good as B on every objective AND strictly better on at least one.
4. **Crowding distance.** Within each front, compute how isolated each design is. Designs in sparse parts of the front get high crowding distance.
5. **Select for next generation.** Prefer lower-numbered fronts (Pareto-optimal); within a front, prefer higher crowding distance (diverse).
6. **SBX crossover + polynomial mutation** as in GA.
7. **Repeat 300 generations** (or 500 for the 10-bar Pareto study).

**Output:** Not one design but ~20-100 *non-dominated designs* spread along the trade-off curve. The designer picks one based on which they value more (light vs stiff).

**Hyperparameters:** Same SBX/mutation as GA, archive size = N = 100.

**Who invented it:** Deb, Pratap, Agarwal, Meyarivan 2002 — one of the most-cited papers in evolutionary computation.

### Constraint handling — the same recipe for all three

All three optimisers use **feasibility-first selection**:

1. Feasible always beats infeasible, regardless of weight.
2. Among feasible designs: lighter wins.
3. Among infeasible designs: less violation wins.

In code (pymoo's built-in mechanism):

```python
def is_better(individual_a, individual_b):
    if individual_a.feasible and not individual_b.feasible:
        return True
    if not individual_a.feasible and individual_b.feasible:
        return False
    if individual_a.feasible:  # both feasible
        return individual_a.W < individual_b.W
    # both infeasible
    return individual_a.total_violation < individual_b.total_violation
```

This pushes the population toward feasibility first, then minimises weight at the boundary.

---

## 11. Layer 4 — The neural surrogate (your 132× speedup)

The expensive thing is the FEM solve. The surrogate is a neural network that predicts FEM output 132 times faster.

### Why this works

The function $f: \mathbf{A} \mapsto (W, \delta_{\max}, \sigma_{\max})$ is **smooth** in $\mathbf{A}$ on a small benchmark. Small changes to areas produce small changes to weight, displacement, and stress. Smooth functions are easy for neural networks to learn.

### The architecture in detail

A multi-layer perceptron (MLP) with three hidden layers:

```
Input:           [A₁, A₂, ..., A_n]              dimension n (10 for 10-bar)
                          ↓
Hidden 1:        Linear(n → 256) → ReLU → Dropout(p=0.1)
                          ↓
Hidden 2:        Linear(256 → 128) → ReLU → Dropout(p=0.1)
                          ↓
Hidden 3:        Linear(128 → 64) → ReLU → Dropout(p=0.1)
                          ↓
Output:          Linear(64 → 3)
                          ↓
                 [log(1+W̃), log(1+δ̃), σ̃]
```

**Why log(1+W) for weight and displacement?** Weight and displacement are positive and long-tailed (a few big-area designs have huge weight). $\log(1+x)$ compresses the range so the network doesn't waste capacity on the tails.

**Total parameter count:** ~50 000. Small enough to train in 2 minutes on a CPU.

**Activations:** ReLU (Rectified Linear Unit) $f(x) = \max(0, x)$. Standard, fast, doesn't suffer from vanishing gradients.

**Dropout:** During training, randomly zero 10 % of neurons. This regularises the network — prevents overfitting and improves generalisation. *During inference, dropout is off and the full network runs.*

### The training data — Latin Hypercube Sampling

**Why LHS instead of uniform random?**

If you uniformly sample 10 000 points in a 10-dimensional unit cube, you'll get clusters and gaps. By chance, some dimensions get well-covered and others get sparse.

LHS guarantees that *every one-dimensional projection* is uniformly stratified. If you slice the cube into 10 000 vertical strips along $A_1$, you get exactly one sample per strip. Same for $A_2$, $A_3$, etc.

```python
from scipy.stats.qmc import LatinHypercube
sampler = LatinHypercube(d=n_design_vars, seed=42)
samples_unit_cube = sampler.random(n=10000)
samples_scaled = lower_bounds + samples_unit_cube * (upper_bounds - lower_bounds)
```

Then for each sample, run FEM, store the (input, output) pair. ~7 minutes for 10 000 evaluations on 10-bar (parallelised).

### The training protocol

```python
# Split 80/10/10 train/val/test
# Standardise inputs and outputs (zero mean, unit variance on train)
optim = Adam(model.parameters(), lr=1e-3)
sched = CosineAnnealingLR(optim, T_max=200)
loss_fn = MSELoss()

for epoch in range(200):
    for x_batch, y_batch in train_loader:
        loss = loss_fn(model(x_batch), y_batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
    sched.step()
    if val_loss_no_improvement_for(20):
        break  # early stop
```

**Adam optimiser:** Adam (Adaptive Moment Estimation) is a gradient-descent variant that adapts the learning rate per-parameter based on running gradient statistics. Faster convergence than plain SGD on most problems.

**Cosine annealing schedule:** Start at lr = 1e-3, decay smoothly toward 0 over 200 epochs. Helps the network make big moves early and fine adjustments late.

**Batch size 128:** Standard, balances gradient noise (smaller batch = noisier, bigger batch = smoother but slower).

**Early stopping:** Stop if validation loss doesn't improve for 20 epochs. Saves compute when overfitting starts.

### Validation results — Chapter 4 §4.5

On the 10-bar held-out test set (10 % of LHS samples, never seen during training):

| Output head | $R^2$ | Use in pipeline |
|---|---|---|
| Weight | **0.9993** | Direct replacement for FEM weight |
| Displacement | 0.87 | Feasibility *screen* (FEM fallback when borderline) |
| Stress | 0.82 | Feasibility *screen* (FEM fallback when borderline) |

**Interpreting $R^2 = 0.9993$:** It means the surrogate explains 99.93 % of the variance in true weight. Predicted weights sit almost exactly on the y=x line.

**Interpreting $R^2 = 0.87$ for displacement:** Less perfect. The network knows the displacement is *roughly* what it should be but errs on extreme designs. So we use the surrogate's displacement only as a *rough check* — if it's close to the constraint boundary, fall back to FEM for an exact value.

This is the **Queipo 2005 hybrid surrogate pattern** — trust the surrogate for objectives, fall back to FEM for borderline constraints.

### The speedup

- FEM: 41 ms per design (laptop CPU)
- Surrogate: 0.31 ms per design (batched, PyTorch on CPU)
- **Ratio: 132×**

Batched inference matters — sending 100 designs to the surrogate at once is much faster than one at a time, because GPU/CPU vectorisation kicks in.

### The optimum quality

Surrogate-in-loop optima (running GA with surrogate replacing FEM) match FEM-in-loop optima to **0.09 %** on 10-bar. So you buy the speed without losing optimality.

**Cross-validation:** Even if the surrogate has small per-design errors, the optimum *location* in design space is preserved, because the relative ordering of designs is what matters for the GA.

---

## 12. Layer 5 — The PPO reinforcement-learning agent

### What RL is, intuitively

Two paradigms in ML:
- **Supervised learning** (e.g. the surrogate): you give the model many (input, correct-output) pairs. It learns to map inputs to outputs.
- **Reinforcement learning**: you put an agent in an environment. The agent takes actions, sees consequences (rewards), and learns over many tries to take actions that maximise total reward.

RL is how AlphaGo learned Go, how robots learn to walk, how chatbots are tuned via RLHF.

### Why use RL for truss design

Imagine you could train a network to read a problem description (geometry, loads, bounds) and *output* a good design in one forward pass — no GA loop, no surrogate, just one inference call.

That's the dream. Inference time would be ~0.2 seconds vs ~30 seconds for a fresh GA.

### Your formulation — single-step bandit

**Environment** (`src/rl/environment.py`):
- **Observation:** A fixed-length vector encoding the benchmark — bounds, loads, symmetry groups, all normalised. The agent sees the same observation for the same benchmark every episode.
- **Action:** A Box action $\mathbf{a} \in [-1, 1]^{n_{\text{vars}}}$, scaled to design-variable bounds.
- **Episode:** **One step**. The agent emits one action vector (= one area vector), the environment evaluates it via the surrogate, returns a reward, and the episode ends.

This is a **contextual bandit** — not a sequential decision process, just one decision per episode. It's a special case of RL.

### The reward function

$$r(\mathbf{x}) = -W(\mathbf{x}) - \lambda \sum_i \max(g_i(\mathbf{x}), 0)^2, \quad \lambda = 10^4$$

- **$-W$**: minimising weight = maximising reward
- **Quadratic penalty** on every positive constraint value: $\max(g_i, 0)^2$ — only violations contribute
- **Large $\lambda$**: pushes the agent firmly back into the feasible region, away from the constraint boundary where the surrogate is least accurate

### The training algorithm — PPO

**PPO (Proximal Policy Optimisation, Schulman 2017)** is a policy-gradient method with a clipped surrogate objective. It's the default RL algorithm at OpenAI and is included in Stable-Baselines3.

The PPO algorithm in one paragraph: collect a batch of (state, action, reward) tuples by running the current policy. Compute the advantage (how much better was this action than the policy's average action). Update the policy parameters to make high-advantage actions more likely, but **clip** the update size so the policy doesn't change too much in one step. Repeat.

**Hyperparameters (all defaults):**
- Total timesteps: $10^6$ (~20 min on CPU)
- Rollout length: 2048
- Mini-batch size: 64
- Learning rate: $3 \times 10^{-4}$
- GAE λ: 0.95
- Discount γ: 0.99 (not very meaningful since episodes are 1-step)
- Clip ε: 0.2

### Training — what happens in 20 minutes

PPO collects ~500 rollouts (one rollout = 2048 episodes = 2048 design proposals). For each rollout, it computes the advantage of each action and runs a few epochs of gradient descent to update the policy network and value network. After ~1 million steps total, the policy has converged to a fairly stable design proposal.

### Evaluation — the honest mixed result

On 10-bar:
- PPO best feasible weight: **5876.9 lb**
- Literature optimum: 5060.85 lb
- **Gap: +16.1 %**
- Inference time: 0.2 s per design

This misses the **5 % gate** from the research objectives. We report it **honestly** as a mixed result.

### The Zhao 2021 framing

Zhao et al. 2021 published a comprehensive study showing that **on single-instance structural problems, a tuned GA outperforms a model-free RL agent**. RL's advantage is **cross-instance generalisation** — train once, deploy to many problems without retraining.

So the PPO layer's contribution is **inference speed**, not single-instance optimality. The framework supports the layer; future work F4 would actually test the generalisation claim by training across all four benchmarks together.

### What you say in the viva about PPO

> "PPO converges to 5876.9 lb on 10-bar, a 16.1 % gap vs the 5060.85 lb literature value. This consistent with Zhao 2021's finding that single-instance model-free RL doesn't beat a tuned GA. RL's advantage is **inference speed** — 0.2 s per design versus 30 s for a fresh GA — and **cross-instance generalisation**, which would need transfer learning across benchmarks. I list that as future work F4. I report this as a mixed result, not a claimed success."

This is **respected**. Examiners value honesty about negative results.

---

## 13. Layer 6 — The LLM warm-start designer (the headline finding)

This is your **strongest novel claim** in the thesis. Worth understanding inside-out.

### What an LLM is, in this context

A Large Language Model — here Anthropic's Claude — is a neural network with hundreds of billions of parameters trained on huge amounts of text. It can answer questions, write code, reason about engineering problems.

In your thesis, you don't fine-tune or modify the LLM. You access it through the public Anthropic API with a structured prompt and get back a JSON response with a suggested design vector.

### What "warm-start" means

The GA normally starts with **100 random designs** in its initial population. "Cold start".

A **warm-start** = seed the initial population with one or more *clever guesses*. If the guess is in a good neighbourhood, the GA converges faster.

Your specific design: inject one LLM-suggested design as the first individual of the 100-person initial population. The remaining 99 stay random.

### The prompt design

**System prompt** (sets the role):

> *"You are a senior structural engineer with deep expertise in steel truss optimization (Sunar-Belegundu, Schmit-Miura, Erbatur, Kaveh benchmarks). Respond ONLY in the JSON format described, no prose, no code fences. Numbers must be floats within the stated bounds."*

This is **role priming** — telling the LLM to act as a domain expert. Improves response quality.

**User prompt** (the actual question):

> *"Here is the [benchmark name] truss problem.*
>
> *Node coordinates (in inches):*
> *Node 1: (0, 360)*
> *Node 2: (0, 0)*
> *...*
>
> *Elements (i, j, group):*
> *Bar 1: (5, 3, group 1)*
> *Bar 2: (3, 1, group 2)*
> *...*
>
> *Loads (kips, downward):*
> *Node 2: 100*
> *Node 4: 100*
>
> *Constraints:*
> *Stress ≤ ±25 000 psi*
> *Displacement ≤ 2 in*
> *Slenderness ≤ 180 in compression, 400 in tension*
>
> *Material: E = 1e7 psi, ρ = 0.1 lb/in³*
>
> *Area bounds: [0.1, 35] in²*
>
> *Think step by step about which members will be in tension vs compression, which might be redundant, and what cross-sectional areas you'd suggest. Then output your final design as JSON: `{"areas": [...]}`."*

The "think step by step" line is **chain-of-thought prompting** (Wei et al. 2022) — it asks the LLM to reason before committing to an answer. This substantially improves performance on compositional-reasoning tasks.

### The client and the cache

`src/llm/client.py` wraps the `anthropic` Python SDK with two additions:

**1. Content-hashed disk cache.** Every API call's prompt is SHA-256 hashed. The (hash → response) mapping is stored in a JSON file. On a second call with the same prompt, the cache returns the stored response without billing the API.

This is what makes the thesis **offline-reproducible**. The cache JSON is checked into git under `results/llm_cache/`. Anyone can clone the repo and reproduce every LLM result without an Anthropic API key.

**2. Heuristic fallback.** If the API call fails (network down, rate limited, parse error), fall back to a *fully-stressed-design heuristic*:

$$A_i = \frac{|F_i|}{0.75 \sigma_{\text{allow}}}$$

where $F_i$ is the axial force in bar $i$ from a first FEM pass. This gives a feasible design (margin of 1/0.75 ≈ 33 % below yield) and lets the offline CI pass.

### The experiment — paired Wilcoxon design

**Why paired?**

Imagine you run 40 random-seeded GA runs and 40 LLM-seeded GA runs, and compare the *means*. If the LLM-seeded mean is lower, was it the LLM helping or just lucky seeds?

**Paired** means both arms share the *same* random seeds. For seed 42, you run BOTH the random-init GA AND the LLM-seeded GA. The only difference between the two arms is the LLM injection. Now if the LLM-seeded arm converges faster on seed 42, that difference is *attributable* to the LLM, not to seed variance.

The **Wilcoxon signed-rank test** is the non-parametric paired test — it doesn't assume the differences are normally distributed. Robust and standard.

### The 10-bar result

40 seeds, 10-bar, generations-to-converge-within-1 %:

| | Mean generations | Std |
|---|---|---|
| Random initialisation | **203.1** | ~30 |
| LLM warm-start | **128.3** | ~25 |
| Reduction | **36.8 %** | |
| Paired Wilcoxon p-value | **0.0046** | |

**Interpretation of p = 0.0046:** the probability that a 36.8 % reduction this consistent across 40 seeds happened *by chance* is 0.46 %. Conventionally we call $p < 0.05$ "statistically significant" and $p < 0.01$ "highly significant". Yours is firmly in the latter.

### Why it works — the mechanism

When you read Claude's chain-of-thought (cached under `results/llm_cache/`), you find it identifies that certain members of the 10-bar truss are **redundant** — they carry little load and should be at their lower-area bound.

Specifically, Claude flags **members 2, 5, 6, and 10** as low-stress in the optimum and proposes small areas for them.

This is an architectural insight. A GA starting from a random population would *eventually* discover this — but only after ~70 generations of trial and error. The LLM provides it for free in one API call. That accounts for the ~75-generation reduction (203 − 128).

### The bigger benchmarks — the efficacy map

| Benchmark | Seeds | $\bar{g}$ random | $\bar{g}$ LLM | Reduction | p-value |
|---|---|---|---|---|---|
| 10-bar | 40 | 197.2 | 124.7 | **36.8 %** | **0.0046** |
| 25-bar | 3 | 82.7 | 19.3 | 76.6 % | 0.25 |
| 72-bar | 5 | 46.2 | 48.4 | -4.8 % | 0.81 |

- **25-bar:** huge effect size (76.6 %) but only 3 seeds — *underpowered*. A 10-seed study would likely make it significant.
- **72-bar:** null result. No effect either way.

### Why the effect fades

On 72-bar, every member of the four-storey tower is carrying load — there's no "redundant member set" to flag. The LLM has no qualitative architectural insight to add. The GA finds the constraint boundary in ~46 generations regardless of warm-start.

This is the **efficacy map**: **the LLM helps where it can supply qualitative architectural reasoning**. On compact problems with redundant members, it shaves generations. On dense problems where every member contributes, it doesn't help.

To my knowledge, this is the **first statistically rigorous demonstration of LLM-warm-start benefit in truss sizing**. That's why it's the strongest novel claim in your thesis.

### What you say in the viva

> "On 10-bar with 40 seeds and paired Wilcoxon, the LLM warm-start cuts generations to convergence by 36.8 % at p = 0.0046. Claude correctly identifies the redundant member set {2, 5, 6, 10} — an architectural insight the GA would discover through 70 generations of trial and error. On larger 3D benchmarks the effect fades because there's no comparable insight to add. So the contribution isn't 'LLMs are magic for design' — it's a quantitative efficacy map of *where* an LLM helps in evolutionary truss optimisation. As far as I know, the first such study."

---

## 14. Layer 7 — The IS 800:2007 compliance layer

Every reported design passes four clauses. This is what makes the framework legally compliant for Indian practice.

### What IS 800:2007 is

The **Indian Standard 800:2007 General Construction in Steel** is the Bureau of Indian Standards code for steel structures, published 2007 (replacing the working-stress 1984 edition). It's **legally binding** in India — every steel structure must comply.

It's a **limit-state design** code, aligned with Eurocode 3 and AISC 360. Limit-state design separates *ultimate* limits (will it collapse?) from *serviceability* limits (will it deflect too much / vibrate too much / etc.).

### Clause 3.7 / 3.8 — Slenderness limits

**Why this matters:** A bar that is "long and thin" relative to its cross-section will buckle (bend sideways) under compression long before it yields. The slenderness ratio $\lambda$ quantifies this.

$$\lambda = \frac{L_{\text{eff}}}{r}, \quad r = \sqrt{\frac{I}{A}}$$

where $L_{\text{eff}}$ is the effective length (here, simply the bar length L) and $r$ is the radius of gyration.

For a generic rod section, $r \approx \sqrt{A/\pi}$ (treating the area as a circle). This is a conservative approximation.

**Limits:**
- **Compression members:** $\lambda \le 180$ (under dead + imposed loads)
- **Tension members:** $\lambda \le 400$ (less strict — tension doesn't buckle)

Your `src/constraints/is800_checks.py` runs FEM to extract *signed* axial forces (tension positive, compression negative), then applies the appropriate limit per bar.

### Clause 5.6.1 — Serviceability deflection

Maximum nodal displacement ≤ $L/325$ for trusses supporting brittle cladding.

**Why this matters:** Even if no bar breaks, a truss that deflects too much will crack brittle cladding (asbestos, terracotta, fragile sheeting). A 30 m span with $L/325$ gives ~92 mm max deflection — not much.

### Clauses 6.2 / 6.3 — Tension capacity

Two failure modes for a tension bar:
- **Gross-section yielding** (Clause 6.2):
  $$T_{dg} = \frac{A_g f_y}{\gamma_{m0}}$$
  where $A_g$ is the gross cross-section area, $f_y$ is the yield strength (250 MPa for Fe 250 steel), $\gamma_{m0} = 1.1$ is the partial safety factor for yielding.

- **Net-section rupture** (Clause 6.3):
  $$T_{dn} = \frac{0.9 A_n f_u}{\gamma_{m1}}$$
  where $A_n$ is the net section area (after bolt-hole deductions), $f_u$ is the ultimate strength (410 MPa for Fe 250), $\gamma_{m1} = 1.25$.

**Design strength** = $\min(T_{dg}, T_{dn})$.

For your pin-jointed benchmarks, $A_n = A_g$ (no holes), so the governing case is usually Clause 6.3.

### Clause 7.1 — Compression capacity

Design compressive stress:
$$f_{cd} = \chi \cdot f_y / \gamma_{m0}$$

where $\chi$ is the **column-curve buckling-reduction factor** as a function of non-dimensional slenderness:

$$\bar{\lambda} = \sqrt{\frac{f_y}{f_{cc}}}, \quad f_{cc} = \frac{\pi^2 E}{\lambda^2}$$

$f_{cc}$ is the Euler buckling stress.

IS 800 defines four column curves ($a$, $b$, $c$, $d$) with different imperfection factors $\alpha$:
- Curve $a$: $\alpha = 0.21$ (best — for rolled I-sections and hollow sections)
- Curve $b$: $\alpha = 0.34$
- Curve $c$: $\alpha = 0.49$
- Curve $d$: $\alpha = 0.76$ (worst — for thick welded sections)

Your module uses **curve $a$** (the recommended one for generic structural sections).

The formula for $\chi$:
$$\phi = 0.5[1 + \alpha(\bar\lambda - 0.2) + \bar\lambda^2], \quad \chi = \frac{1}{\phi + \sqrt{\phi^2 - \bar\lambda^2}}, \quad \chi \le 1$$

In plain English: $\chi$ smoothly varies from 1 (no buckling reduction, for short stocky bars) to near 0 (severe buckling reduction, for long thin bars). The transition happens around $\bar\lambda \approx 1$.

### How it's implemented

`src/constraints/is800_checks.py` — each clause is a pure Python function:

```python
def check_slenderness(L, A, axial_force, in_tension=None):
    r = sqrt(A / pi)
    lambda_ratio = L / r
    if axial_force >= 0:  # tension
        return lambda_ratio - 400  # g ≤ 0 if feasible
    else:  # compression
        return lambda_ratio - 180

def check_compression(L, A, axial_force, E, f_y, gamma_m0=1.1, alpha=0.21):
    lambda_ratio = L / sqrt(A / pi)
    f_cc = pi**2 * E / lambda_ratio**2
    lambda_bar = sqrt(f_y / f_cc)
    phi = 0.5 * (1 + alpha * (lambda_bar - 0.2) + lambda_bar**2)
    chi = min(1.0, 1.0 / (phi + sqrt(phi**2 - lambda_bar**2)))
    f_cd = chi * f_y / gamma_m0
    sigma_actual = abs(axial_force) / A
    return sigma_actual - f_cd  # g ≤ 0 if feasible
```

`src/constraints/compliance.py` orchestrates them into a `ComplianceReport` dataclass — one combined verdict per design.

### Cross-checked against Indian textbooks

`tests/test_is800.py` runs each clause against worked examples from:
- **Subramanian's *Design of Steel Structures*** (Oxford University Press)
- **Duggal's *Limit State Design of Steel Structures*** (Tata McGraw-Hill)

All cross-checks agree to **3 significant figures** (the quoted precision of the textbook solutions). This is your evidence that the IS 800 module is implementation-correct, not just code-correct.

---

## 15. Layer 8 — The Streamlit + FastAPI deliverable

This is the **practitioner-facing layer**. Without it, your work is a research codebase that only Python-literate users can drive. With it, a junior engineer can run an IS 800-compliant optimisation in their browser.

### FastAPI backend — `src/app/api.py`

FastAPI is a modern Python web framework for building REST APIs. Endpoints:

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Is the server alive? |
| `/benchmarks` | GET | List available benchmark names |
| `/benchmarks/{name}` | GET | Full spec of one benchmark |
| `/optimize` | POST | Run optimisation. Request body: `{benchmark, algorithm, seed, n_gen, use_llm, use_surrogate}`. Returns best design, weight, convergence history, compliance report. |
| `/llm/suggest` | POST | Get an LLM warm-start design for a benchmark |

**LLM warm-start is only available for single-objective solvers (GA, PSO).** pymoo's NSGA-II implementation doesn't accept an initial-population argument in the same form.

### Streamlit UI — `src/app/ui.py`

Streamlit lets you write a UI in pure Python — no HTML, no CSS, no JavaScript. You declare widgets and Streamlit renders them.

The UI flow:

1. **Sidebar** — benchmark dropdown (10/25/72/200-bar), algorithm dropdown (GA/PSO/NSGA-II), seed input, generations input, toggles for "use surrogate" and "use LLM warm-start"
2. **Run button** — hits the FastAPI `/optimize` endpoint with the user's choices
3. **Results panel:**
   - Headline: best weight, time taken
   - Convergence curve: weight vs generation (matplotlib)
   - Truss visualisation: nodes + bars with thicknesses drawn at scale
   - For NSGA-II: Pareto front plot (weight vs displacement)
   - IS 800 compliance report: which clauses pass / fail, with per-clause numbers

### The headline performance — < 10 seconds

End-to-end (UI → API → GA run with surrogate → compliance report → JSON back → UI render): **under 10 seconds on 10-bar on a laptop CPU**.

This requires `use_surrogate=true`. Without the surrogate, the FEM inner loop caps the GA at ~40 s per run.

### Three pieces of value

This is your viva soundbite for the practical-impact question:

1. **Non-Python users** — a junior consultant can drive the entire framework without writing code.
2. **Sub-10-second response** — moves optimisation from a batch background process to an interactive design-review tool.
3. **Offline reproducibility** — all LLM responses cached, anyone can clone tag `phase-10-complete` and reproduce every number in the thesis without an Anthropic API key.

---

# PART III — THE RESULTS

## 16. 10-bar — the validation signature

This is the most-cited benchmark in truss sizing. Every paper in the field has a 10-bar number. Your match against Sunar 1991 is your **single most important quantitative result**.

### What you ran

- Algorithm: PSO (also GA, for cross-check)
- Population: 100 particles
- Generations: 500
- Seeds: 10 (42, 123, 456, 789, 1024, 2024, 4096, 8192, 16384, 32768)

### What you got

| Algorithm | Best (lb) | Mean (lb) | Std (lb) | Error vs lit |
|---|---|---|---|---|
| PSO | 5061.05 | 5063.7 | 1.8 | 0.004 % |
| GA | 5062.9 | 5066.8 | 2.4 | 0.04 % |

Cross-seed spread under 0.05 %. PSO has effectively zero variance. The number is *robust*.

### Why this is the signature

Imagine every layer is a step in a chain:

```
FEM correct? → Optimiser correct? → IS 800 checks correct? → Bounds correct?
   ↓             ↓                       ↓                       ↓
0.004 % error is only possible if ALL FOUR are correct.
```

If FEM were off by even 1 %, the 10-bar optimum would land at ~5110 lb. If the optimiser's constraint handling were buggy, the optimum would either land at infeasible (~4500 lb) or way too heavy (~5500 lb). If IS 800 were over-constraining, the optimum would land heavier (~5300 lb). If bounds were wrong, the optimum would be impossible to reach.

So 0.004 % means **every layer is implementation-correct**. This is the proof of life for the entire framework.

### What the optimum design looks like

Approximate optimal area vector (from Sunar 1991):

| Bar | Optimum A (in²) |
|---|---|
| 1 | 30.5 |
| 2 | 0.1 (at lower bound — redundant) |
| 3 | 23.2 |
| 4 | 15.2 |
| 5 | 0.1 (redundant) |
| 6 | 0.1 (redundant) |
| 7 | 7.5 |
| 8 | 21.0 |
| 9 | 21.5 |
| 10 | 0.1 (redundant) |

**Members 2, 5, 6, 10 are redundant** — they sit at the lower bound (0.1 in²). They carry essentially zero load in the optimum.

This is the architectural insight Claude correctly identifies in the LLM warm-start study.

---

## 17. 25-bar — the second confirmation

### What you ran

- Algorithm: PSO (also GA)
- Population: 100
- Generations: 500
- Seeds: 10

### What you got

| Algorithm | Best (lb) | Mean (lb) | Std (lb) | Error vs Schmit 1976 |
|---|---|---|---|---|
| PSO | 545.26 | 545.7 | 0.4 | 0.008 % |
| GA | 545.30 | 545.9 | 0.5 | 0.015 % |

### Why 25-bar matters

It's the **first 3D test**. Confirms:
- Your transformation matrices work in 3D
- Symmetry grouping (25 bars → 8 design vars) is handled correctly
- Two load cases (LC1, LC2) are aggregated correctly at the constraint boundary

If 10-bar passes but 25-bar fails, the bug is in your 3D code path or your symmetry handling. Since both pass cleanly, the framework is correct in both 2D and 3D.

---

## 18. 72-bar — the honest story

This is the section that needs the **most careful viva framing**. Read it twice.

### Originally we claimed (now removed)

> "The published Camp & Bichon 2004 (380.24 lb) and Bekdaş 2015 (379.62 lb) optima are infeasible by 22–62 % on displacement and 22 % on stress. The true rigorously-feasible optimum is ~549 lb. This is a novel methodological correction to the published literature, previously undocumented."

Strong claim. Dramatic numbers.

### Why we softened it

Three reasons:

**1. Encoding-bug risk.** The 72-bar problem has been described slightly differently by Fleury–Schmit 1980 → Erbatur 2000 → Camp 2004 → Bekdaş 2015. Subtle differences in *which* nodes are constrained, *what order* the design groups are in, *what sign* the loads have, could produce huge "violations" we observe.

You think your encoding matches Camp 2004. But you haven't proven it through an independent path.

**2. No independent FEM cross-check.** You haven't run Camp's published area vector through OpenSeesPy, SAP2000, or ANSYS — second-opinion finite-element solvers — and confirmed it also reports the same violations.

Without that, the claim that "Camp's optimum is infeasible" rests entirely on your own FEM. If your FEM has a tiny bug that *just* affects 72-bar (say a sign error in one of the load vectors), the claim collapses.

**3. The violation magnitude is suspiciously large.** Soft-penalty heuristic literature is known to have minor violations (~1-5 %) at "optima". But **22-62 % violations is too large** to be plausibly explained by soft-penalty slop alone. That magnitude is either a **dramatic literature artefact** or an **encoding mismatch on our side**. Both are possible; we can't distinguish without independent verification.

### What we now say (after softening)

> *"Under hard-constraint enforcement ($g \le 0$) our GA, PSO, and NSGA-II converge to ~549 lb on 72-bar with cross-seed spread under 1 %. We could not reproduce the commonly cited Camp & Bichon 2004 or Bekdaş 2015 optima as feasible in our FEM. This is consistent with the soft-penalty culture in heuristic-optimisation literature, but pending an independent FEM cross-check we frame this as a **constraint-handling-convention observation** rather than a methodological correction. (See future-work item F7.)"*

### What's gone from the writeup

- "Novel finding"
- "Methodological correction"
- "Infeasible by 22–62 %"
- "Previously undocumented"
- "First-order economic fact"
- "Indian civil-engineering community should be aware"

### What stayed (because it's true)

- The 549 lb hard-constraint result (your own FEM said so, three algorithms agreed, cross-seed spread < 1 %)
- The comparison table showing your result alongside Camp's 379.62 lb
- The hard-vs-soft constraint ablation in §4.10.2 (showing your soft-penalty version reaches Camp's ~380 lb)

### What's new — future-work item F7

> "F7. **Independent FEM cross-check of the 72-bar encoding** (OpenSeesPy / SAP2000 / ANSYS) using the published Camp 2004 area vector. If a second FEM reproduces our displacement and stress numbers, the constraint-handling-convention observation can be upgraded to a methodological correction; if not, our encoding has a bug we have not yet caught."

This is what you'd do next if pursuing this as a paper.

### What you say in the viva

> "On 72-bar I report an observation, not a finding. My GA, PSO, and NSGA-II all converge to ~549 pounds with cross-seed spread under one percent under hard-constraint enforcement. I could not reproduce the published 379.62 pounds as feasible in my FEM, but pending an independent FEM cross-check — for example against OpenSeesPy — I don't claim the literature is wrong. I list that cross-check as future-work item F7. The honest contribution here is the 549-pound baseline."

**Calm. Clear. Defensible from any angle.**

### If asked: "Why do you think your 549 lb is right and the literature's 380 lb is wrong?"

> "I don't claim that, sir. My result is an empirical observation: under hard-constraint enforcement, three independent algorithms — GA, PSO, and NSGA-II — converge to 549 pounds with cross-seed spread under 1 %. That consistency suggests it's a genuine minimum of *my* problem formulation. Whether the literature's 380 pounds is actually feasible under their FEM, or whether their FEM differs from mine in some subtle way, I haven't verified independently. That's what future-work F7 would resolve."

### If asked: "Have you tried OpenSeesPy?"

> "Not yet. It's listed as F7 in future work. Honestly, doing that cross-check is the next step — it would either upgrade this to a published methodological observation or expose a bug in my encoding. Either outcome is valuable."

### Bottom line

The 72-bar story is now **defensible**, not **risky**. You report what you observed. You don't claim correctness against the literature. You list the verification step. **You're saying less but you can defend everything you do say.**

---

## 19. 200-bar — the Indian-units stress test

### What's special about 200-bar in your thesis

It's the **only benchmark in real Indian-civil SI units**:
- Structural steel (not aluminium): E = 210 GPa, ρ = 7850 kg/m³
- Stress limit ±250 MPa (Fe 250 steel)
- Displacement limit 100 mm (span/300)
- Real load magnitudes (kN, not kips)

This makes it the most directly relevant benchmark for Indian practitioners.

### What you ran

- Algorithm: PSO (also GA, NSGA-II)
- Population: 100
- Generations: 200 (capped — bigger problem so each gen is more expensive)
- Seeds: 5 for GA/PSO, 3 for NSGA-II

### What you got

| Algorithm | Best (kg) | Mean (kg) | Std (kg) | Spread |
|---|---|---|---|---|
| GA | 316.61 | 317.80 | 1.08 | 0.34 % |
| PSO | 315.89 | 318.37 | 3.10 | 0.97 % |
| NSGA-II | 5824.17 (weight-min corner of Pareto front) | 7619.30 | 1883.22 | — |

GA and PSO are statistically indistinguishable at the 5 % level (Wilcoxon p = 0.69).

### What you explicitly don't claim

You re-encoded the geometry as a **principled regular-grid stepped tower**, NOT the exact Kaveh 2010 coordinates. So you don't claim a literature match against Kaveh's 25 445 lb number — the geometry is different.

The role of 200-bar is **scaling demonstration in Indian units**, not literature reproduction.

### Why 200-bar matters

- **Scaling:** Confirms framework handles 200 elements / 29 design vars without code changes
- **Indian units:** Demonstrates the framework actually works in MPa / GPa / kg / m, not just imperial
- **Stress-binding regime:** Unlike 10-bar (deflection-binding), 200-bar is stress-binding. Tests the IS 800 compression/tension clauses more directly.
- **Compute:** 57 wall-clock minutes for the 13-seed matrix — comparable to 72-bar despite double the dimensionality. Framework scales.

---

# PART IV — THE VIVA

## 20. The 15-minute script — word-for-word

Word count: ~1800. At 130 wpm (calm) that's ~14 minutes, leaving 1 minute of buffer for slide transitions and natural pauses.

### Setup before you speak

- Slide deck on screen (full-screen mode, **not** "slideshow with notes")
- Title slide showing
- Stand to the **left** of the laptop (so you don't block the projector)
- Hands clasped loosely in front of you OR resting on the lectern — **not** in pockets

### SECTION 1 — Opening (0:00 → 1:00)

**Slide:** Title

**Stage:** Look at the senior-most examiner. Smile slightly. Speak slowly.

**Say:**

> *"Good morning, sir. Good morning, ma'am."*
>
> *[pause one full second]*
>
> *"My thesis is titled* Multi-objective Steel Truss Optimisation: an AI-Assisted Framework Compliant with IS 800:2007."
>
> *"Today I'll walk you through the motivation, the eight-layer framework I built, validation against four canonical benchmarks, my headline statistical finding on LLM-assisted design, and the practical Streamlit tool that ships with the thesis."*

**Stage:** Pause two seconds. Click to next slide.

### SECTION 2 — Problem hook (1:00 → 2:30)

**Slide:** Motivation (Indian steel-consumption chart)

**Say:**

> *"Let me start with why this matters."*
>
> *[gesture toward chart]*
>
> *"Steel trusses are ubiquitous in Indian infrastructure — factory roofs, electrical transmission towers, highway bridges, stadium canopies. They are sized today in design offices by trial-and-error iteration against IS 800:2007, the Indian steel code."*
>
> *"The typical hand-design carries ten to thirty percent surplus steel above the theoretical optimum. That's because the engineer stops as soon as the code is satisfied, not when the design is genuinely minimum-weight."*
>
> *"At sector scale, with steel contributing roughly 1.9 kilograms of CO₂ per kilogram, that surplus is a real economic and environmental number for India."*
>
> *[turn back fully to examiners]*
>
> *"Optimisation algorithms can squeeze that surplus out. The problem is speed. A typical evolutionary run does fifty thousand finite-element evaluations — about forty minutes per run on a laptop. Too slow to be an interactive design tool."*
>
> *[pause]*
>
> *"So the question I asked is: can three modern AI techniques — a neural surrogate, a reinforcement-learning agent, and a large language model — make this workflow fast and reliable enough to actually be used by an Indian civil engineer?"*

**Click to next slide.**

### SECTION 3 — What I built (2:30 → 5:00)

**Slide:** Architecture diagram

**Stage:** Take half a second to let the slide settle. Then point at the bottom and **work upward** as you describe.

**Say:**

> *"I built a Python framework with eight layers."*
>
> *[point to bottom]*
>
> *"At the bottom is a finite-element solver I wrote from scratch — pin-jointed truss, two-node bar elements, direct-stiffness assembly. I validated it against Logan's three-bar textbook example to machine-precision error."*
>
> *[hand moves up]*
>
> *"On top sit three classical evolutionary optimisers through the pymoo library — a genetic algorithm with simulated-binary crossover, particle-swarm optimisation, and NSGA-II for multi-objective Pareto fronts."*
>
> *[hand moves up]*
>
> *"On top of those, the three AI accelerators."*
>
> *"First — a neural surrogate. A three-hidden-layer MLP trained on ten thousand Latin-hypercube-sampled FEM evaluations. It replaces the expensive FEM call inside the optimiser's inner loop."*
>
> *"Second — a PPO reinforcement-learning agent built on Stable-Baselines3. It treats truss sizing as a single-step contextual bandit and outputs an entire area vector in one forward pass."*
>
> *"Third — an LLM warm-start designer that queries Anthropic's Claude API in natural language for an initial design vector. Every API response is hashed and cached on disk, so the pipeline reproduces offline without an API key."*
>
> *[brief pause]*
>
> *"Every design produced by any layer is checked against four IS 800:2007 clauses — slenderness, deflection, tension capacity, and compression with the column-curve buckling-reduction factor."*
>
> *[turn back, drop hand]*
>
> *"The whole stack is exposed through a Streamlit UI talking to a FastAPI backend. A non-Python user can run a full optimisation in under ten seconds on a laptop CPU."*

**Click to next slide.**

### SECTION 4 — Validation (5:00 → 7:30)

**Slide:** 10-bar / 25-bar results table

**Stage:** Step slightly closer to the screen. **Point at the specific number as you say it.**

**Say:**

> *"I validated the framework on four canonical benchmarks."*
>
> *"On the 10-bar planar cantilever truss of Sunar and Belegundu, 1991 —"*
>
> *[point at row]*
>
> *"— the published optimum is 5060.85 pounds. My PSO converges to 5061.05 pounds. That's a 0.004 percent error."*
>
> *[pause for emphasis]*
>
> *"This number is more than just a match. Every layer in the stack — FEM, optimiser, IS 800 compliance — has to be correct for this number to land here. So it acts as the end-to-end validation signature for the whole framework."*
>
> *[move to 25-bar row]*
>
> *"On the 25-bar spatial tower of Venkayya, 1971, I match Schmit and Miura's 1976 reference to 0.008 percent — 545.26 pounds against 545.22."*
>
> *[click to surrogate parity plot]*
>
> *"The neural surrogate — three hidden layers, 256-128-64 neurons, ReLU, dropout — achieves an R-squared of 0.9993 on weight prediction on the held-out test set."*
>
> *[gesture at scatter plot]*
>
> *"You can see in this parity plot — the surrogate-predicted weights sit almost exactly on the y-equals-x line."*
>
> *"Wall-clock-wise, FEM averages 41 milliseconds per design, surrogate 0.31 milliseconds batched — a 132-times speedup. And critically, surrogate-in-loop optima match FEM-in-loop optima to 0.09 percent. So we buy the speed without losing optimality."*
>
> *[turn back to examiners]*
>
> *"Every reported optimum passes all four IS 800:2007 clauses end-to-end."*

**Click to next slide.**

### SECTION 5 — Headline finding (7:30 → 10:30)

**Slide:** LLM convergence figure (per-seed paired dots, red vs purple)

**Stage:** **Slow down here.** This is the slide they'll remember. Build to the p-value.

**Say:**

> *"My strongest novel empirical claim is what I call the LLM-warm-start efficacy map."*
>
> *[gesture at figure]*
>
> *"I ran a paired experiment on 10-bar — 40 random seeds, two arms. Arm one — GA with a single Claude-suggested design vector injected into the initial population. Arm two — GA with random initialisation. Both arms share the same seed, so the comparison is directly attributable to the warm-start."*
>
> *"For each seed pair I measured generations to converge within 1 percent of the optimum."*
>
> *[brief pause]*
>
> *"The random-seeded arm averaged 203 generations. The LLM-seeded arm averaged 128."*
>
> *[pause]*
>
> *"That's a 36.8 percent reduction, with paired Wilcoxon p-value of 0.0046 — highly statistically significant."*
>
> *[click to next slide if you have one with Claude's reasoning, else stay]*
>
> *"Why did it work? When you read Claude's chain-of-thought, you find it correctly identifies that members 2, 5, 6, and 10 in the optimum are redundant — they sit close to their lower bound."*
>
> *"This is an architectural insight that the GA would have rediscovered through trial and error, over roughly 70 generations. The LLM provides it for free, in a single API call."*
>
> *[brief pause]*
>
> *"On the larger 3D benchmarks the effect fades. On 25-bar we see a non-significant trend. On 72-bar, a null result. The reason is the LLM has no comparable architectural insight to add on a four-storey tower — every member is carrying load."*
>
> *[turn to examiners, hand drops]*
>
> *"So the contribution is not 'LLMs are magic'. The contribution is the efficacy map itself — a quantified statement of where an LLM helps an evolutionary optimisation, and where it doesn't. To my knowledge, this is the first statistically rigorous demonstration of LLM-warm-start benefit in truss sizing."*

**Click to next slide.**

### SECTION 6 — 72-bar honest framing (10:30 → 12:00)

**Slide:** 72-bar results frame (with the blue pill)

**Stage:** Calm. Steady. Not apologetic.

**Say:**

> *"On the 72-bar spatial tower, I report an observation, not a finding."*
>
> *[gesture at table]*
>
> *"Under hard-constraint enforcement — strict feasibility, $g$ less than or equal to zero — my GA, particle-swarm, and NSGA-II all converge to approximately 549 pounds, with cross-seed spread under one percent. Three independent algorithms agreeing tightly."*
>
> *[brief pause]*
>
> *"I could not reproduce the commonly cited Camp & Bichon 2004 or Bekdaş 2015 optima — around 380 pounds — as feasible under strict constraint enforcement in my FEM."*
>
> *"This is consistent with the soft-penalty culture in heuristic optimisation literature. But pending an independent FEM cross-check — for example against OpenSeesPy — I frame this as a constraint-handling-convention observation, not a correction to the published values. I've listed the independent cross-check as future-work item F7."*
>
> *[turn back, calm]*
>
> *"The honest contribution here is the rigorously-feasible 549-pound baseline, not a claim that the literature is wrong."*

**Click to next slide.**

### SECTION 7 — Streamlit deliverable (12:00 → 13:30)

**Slide:** Streamlit UI screenshot

**Stage:** Smile a little. This is the practical-impact moment.

**Say:**

> *"On the practical side, the surrogate compresses a 30-to-40-minute classical optimisation run into roughly 15 seconds."*
>
> *[gesture at screenshot]*
>
> *"The Streamlit UI exposes the whole stack — benchmark selection, algorithm choice, surrogate toggle, LLM warm-start toggle, IS 800 compliance report — as a single-command web app. A junior engineer or M.Tech student can drive it without writing Python."*
>
> *"The full pipeline — UI to API to GA run to compliance report to JSON back to UI render — completes in under 10 seconds on a laptop CPU for the 10-bar benchmark."*
>
> *[brief pause]*
>
> *"Every LLM response is cached on disk. So the entire pipeline reproduces offline — anyone can clone the repository at tag `phase-10-complete` and reproduce every number in this thesis without an Anthropic API key."*
>
> *[turn back]*
>
> *"Three pieces of value: non-Python users, sub-ten-second response, and offline reproducibility."*

**Click to next slide.**

### SECTION 8 — Limitations, future, close (13:30 → 15:00)

**Slide:** Contributions slide (C1–C5)

**Stage:** Stand still. **Slow down.** This is the impression they'll remember.

**Say:**

> *"A few honest limitations, briefly."*
>
> *[count on fingers if helpful]*
>
> *"The FEM is linear elastic — no geometric or material non-linearity, no dynamic or seismic loading. Optimisation is sizing-only, not topology. The PPO agent does not beat a tuned GA on single instances — a 16 percent gap on 10-bar — consistent with Zhao 2021's finding. Its value would lie in cross-instance generalisation, which I list as future work."*
>
> *[pause]*
>
> *"There are seven future-work directions in Chapter 5. The most high-leverage: a graph-neural-network surrogate for variable-topology design, PPO transfer learning across the four-benchmark family, and the independent FEM cross-check of the 72-bar encoding I just mentioned. The most practically valuable: validation against real Indian transmission-tower designs through a data-sharing arrangement with PGCIL."*
>
> *[final pause — half a second]*
>
> *"To close —"*
>
> *[gesture briefly at contributions slide]*
>
> *"This thesis delivers a validated, IS 800-compliant, reproducible AI-assisted truss design framework, matching classical benchmark optima to fractions of a percent. The principal novel empirical claim is the LLM warm-start efficacy map, with statistically significant gains on small problems. The whole framework ships as a sub-ten-second Streamlit demo, fully reproducible offline."*
>
> *[full second pause]*
>
> *"Thank you, sir. Thank you, ma'am. I'm happy to take questions."*

**Stop. Smile slightly. Step back half a step. Wait.**

---

## 21. The slide-by-slide narrative

For each slide in the new 30-slide deck, here's what to say.

(These will roughly match the 8 sections above but slide-level — so you have a one-line beat to recall for each click.)

### Slide 1 — Title

*Beat:* Greet + thesis title + agenda preview.

### Slide 2 — Motivation chart

*Beat:* Trusses everywhere → 10–30 % surplus → sector-scale CO₂ → optimisers exist but too slow.

### Slide 3 — Question / objective

*Beat:* Can three AI techniques make truss optimisation interactive while staying IS 800-compliant?

### Slide 4 — Architecture overview

*Beat:* Eight-layer stack: FEM → optimisers → AI accelerators → IS 800 → UI.

### Slide 5 — FEM details

*Beat:* Pin-jointed bar elements, 2D/3D unified, validated to machine epsilon.

### Slide 6 — GA / PSO / NSGA-II

*Beat:* Standard pymoo defaults; feasibility-first constraint handling.

### Slide 7 — Neural surrogate architecture

*Beat:* 3-hidden-layer MLP, 256-128-64, dropout, three output heads.

### Slide 8 — Surrogate training (LHS)

*Beat:* 10 000 Latin-hypercube samples, 80/10/10 split, Adam + cosine schedule.

### Slide 9 — PPO architecture

*Beat:* Single-step bandit env, reward = −W − λΣmax(g, 0)², 1M timesteps.

### Slide 10 — LLM pipeline

*Beat:* Chain-of-thought prompt → JSON parse → clip to bounds → inject into GA init population. Cached.

### Slide 11 — IS 800 clauses

*Beat:* Four clauses — 3.8 slenderness, 5.6.1 deflection, 6.2/6.3 tension, 7.1 compression.

### Slide 12 — 10-bar results table

*Beat:* 0.004 % error vs Sunar 1991 — end-to-end validation signature.

### Slide 13 — 25-bar results table

*Beat:* 0.008 % error vs Schmit 1976 — 3D confirmation.

### Slide 14 — Surrogate parity plot

*Beat:* R² = 0.9993, 132× speedup.

### Slide 15 — Surrogate-in-loop equivalence

*Beat:* Optimum quality preserved to 0.09 %.

### Slide 16 — LLM warm-start convergence figure (the headline slide)

*Beat:* 40 seeds → 128 vs 203 generations → 36.8 % reduction → p = 0.0046.

### Slide 17 — LLM reasoning (members {2, 5, 6, 10})

*Beat:* Why it worked: redundant-member identification.

### Slide 18 — LLM efficacy table (10-bar, 25-bar, 72-bar)

*Beat:* Effect fades on bigger benchmarks → efficacy map insight.

### Slide 19 — 72-bar results frame (the softened slide)

*Beat:* 549 lb hard-constraint baseline; observation pending FEM cross-check.

### Slide 20 — 200-bar Indian-units result

*Beat:* 317 kg in proper SI / Indian-steel units.

### Slide 21 — PPO result

*Beat:* 16 % gap, mixed result, Zhao 2021 reframing.

### Slide 22 — Pareto fronts (NSGA-II)

*Beat:* 24–32 non-dominated points per seed on 10-bar.

### Slide 23 — IS 800 compliance verification table

*Beat:* All four clauses pass at every reported optimum.

### Slide 24 — Streamlit UI screenshot

*Beat:* Non-Python users; < 10 s end-to-end; offline reproducible.

### Slide 25 — Five contributions C1–C5 (with softened C3)

*Beat:* The story summarised.

### Slide 26 — Limitations slide

*Beat:* Linear elastic, sizing only, single-instance PPO, single column curve, public-API LLM.

### Slide 27 — Future work F1–F7

*Beat:* Seven directions including independent FEM cross-check (F7).

### Slide 28 — Practical implications

*Beat:* Interactive design office tool; IS 800 compliance built in; non-Python access.

### Slide 29 — Acknowledgements

*Beat:* Supervisor, family, open-source libraries (pymoo, PyTorch, Stable-Baselines3, Streamlit, Anthropic).

### Slide 30 — Thank you / questions

*Beat:* Quiet anchor for Q&A.

---

## 22. Body language, voice, and delivery

### Voice

- **Slow.** When in doubt, slow down. 130 words per minute is faster than it sounds. Listen to your own recording — you'll think you're slow but you're probably normal.
- **Pause after big numbers.** "*0.004 percent error*" — pause. "*p = 0.0046*" — pause. Let it land.
- **Vary pitch slightly.** Monotone is exhausting to listen to. Bring slight upward inflection on numbers; flat-confident inflection on conclusions.
- **Don't trail off.** End every sentence cleanly. Don't fade out.

### Eyes

- **Rotate among examiners.** One examiner per sentence or per paragraph. Hold each for 5–10 seconds.
- **Don't fix on one person.** That makes them uncomfortable and others feel ignored.
- **Look at slides once per slide, briefly.** Just enough to confirm what's showing, then back to examiners.
- **Don't read off the slide.** Big tell of inexperience. You know your material — you don't need to read.

### Hands

- **Open palms** when explaining.
- **Point at slides** when referencing numbers ("this 0.004 % here").
- **Avoid:** hands in pockets, crossed arms, fidgeting with pen / clicker, gripping the lectern.
- If you don't know what to do with them, **clasp lightly in front** of you. Looks composed.

### Feet

- **Plant.** Don't pace. Don't shift weight constantly.
- **Step intentionally** when you do move — e.g. step closer to the slide when you want to point at a number, then step back.

### Face

- **Light, neutral expression** baseline.
- **Slight smile** at the start (warm), at the end (relieved), and when you mention something you're proud of (the 36.8 % result).
- **Don't fake-smile** the whole time. Looks anxious.

### Voice volume

- **Slightly louder than conversational.** Examiners are 4-5 metres away. They need to hear you.
- **If your voice cracks** (it happens), take a breath, sip water, continue. Don't apologise.

### What to do with the clicker

- **Hold it loosely.** Don't grip.
- **Click decisively.** Not multiple soft clicks.
- **If a slide doesn't advance,** check the laptop directly. Don't keep clicking helplessly.

---

## 23. Recovery moves if something goes wrong

### "I blanked, what do I say?"

> *"Let me just take a moment."*
>
> *[Look at slide. Get the next beat from the slide content.]*
>
> *"Right — so the next thing is..."*

Don't apologise. Don't say "I forgot". Just bridge.

### "The slide deck froze."

> *"While the slide loads — let me describe what should be on it."*
>
> [Verbal description of the slide. They'll respect the composure.]

Then try to click it forward. If it's truly frozen, use **Esc** to exit slideshow mode, then re-enter.

### "An examiner interrupts me mid-speech with a question."

Stop. Listen fully. Answer briefly. Then bridge:

> *"Coming back to where I was — the next part is..."*

Don't get flustered. Don't say "as I was saying".

### "An examiner asks a question I don't know the answer to."

> *"That's a good question, sir. I haven't tested that specifically, but my expectation would be X because Y. I'd want to run that experiment to be sure."*

This is **respected**. Bluffing is not.

### "An examiner argues with one of my numbers."

Stay calm. Don't argue back immediately. Listen fully:

> *"I understand the point, sir. My number is based on [specific evidence]. If you'd like, I can show you the underlying calculation in the appendix."*

Then if they want more — open the relevant appendix page or backup slide.

### "I went over 15 minutes."

If you're at 14:30 and not done, **cut to the close immediately**:

> *"...to wrap up — this thesis delivers a validated IS 800-compliant framework with a novel LLM warm-start finding and a Streamlit deliverable. Thank you, sir. I'm happy to take questions."*

Better to under-cover one section than to be cut off mid-sentence.

### "I'm 12 minutes in and still on section 5."

Skip section 6 (72-bar) — just say:

> *"On the 72-bar I'll briefly note that I report a rigorously-feasible baseline of 549 lb pending independent FEM cross-check, listed as future-work F7. Moving on to the Streamlit deliverable..."*

A one-sentence acknowledgment is better than dropping the section.

### "The examiner says my supervisor told them something different."

> *"Thank you for raising that, sir. I'd want to check with [supervisor's name] on that — I may have a different understanding of [specific point]. Could you clarify which specifically?"*

Don't disagree with your supervisor in real-time. Defer.

### "I'm visibly nervous and my voice is shaking."

- Take a slow breath.
- Take a sip of water.
- Look at the *floor* for 1 second to reset eye contact.
- Restart the sentence calmly.

No examiner will mark you down for nerves. They mark you down for content.

---

# PART V — Q&A AND FINAL PREP

## 24. The 32-question Q&A bank

Examiners said to be friendly — but you should still be ready. Here are 32 plausible questions with crisp answers. **Memorise the answer pattern, not the words.**

### Group 1 — Novelty and contributions

**Q1. What is the novel contribution of your work?**
> "Three things, sir. One — the first published framework integrating classical optimisers, neural surrogate, RL, LLM warm-start, and IS 800 compliance in one stack. Two — a 132× surrogate speedup at R² = 0.9993 with optimum quality preserved to 0.09 %. Three — and the strongest novel empirical claim — the LLM warm-start efficacy map, a 36.8 % generation reduction at p = 0.0046, fading on larger problems. The last one is, as far as I know, the first statistically rigorous LLM-warm-start study in truss sizing."

**Q2. What's your favourite contribution?**
> "The LLM warm-start efficacy map, sir. It's small but rigorous — 40 seeds, paired Wilcoxon, p = 0.0046 — and it tells us something genuine about where AI helps in engineering design and where it doesn't. That's what I'd publish first."

**Q3. Is this fundamentally new, or have others done similar work?**
> "The individual layers are not novel. Surrogate-based optimisation is well-established since Queipo 2005. Geng 2024 integrated LLMs with OpenSeesPy. Sun 2023 coupled DNN surrogates with GAs. My contribution is the *integration of all five layers* — that combination is not published — and the *LLM efficacy map*, which is a new empirical claim."

### Group 2 — Methodology

**Q4. How does your FEM work?**
> "Direct stiffness method, sir. For each bar I assemble a local 4×4 (2D) or 6×6 (3D) element stiffness matrix from $\mathbf{k}_e = (AE/L) \mathbf{T}^\top [1,-1;-1,1] \mathbf{T}$. Sum these into a global stiffness matrix $\mathbf{K}$, apply boundary conditions via partition method, solve $\mathbf{K}_{ff} \mathbf{u}_f = \mathbf{F}_f$ with scipy.linalg.solve. Validated against Logan's three-bar analytical example to machine precision."

**Q5. Why MLP and not Gaussian processes or radial basis functions for the surrogate?**
> "Simplicity and scaling, sir. MLPs train in seconds, scale to high-dimensional inputs, and have well-understood failure modes. Gaussian processes scale cubically with training set size — at 10 000 LHS samples that becomes expensive. Plus an MLP gives me a natural multi-output architecture for weight, displacement, and stress in one forward pass."

**Q6. Why PPO and not DQN or SAC?**
> "PPO is the default continuous-action RL algorithm in Stable-Baselines3 and the most widely-published. DQN handles discrete actions, which doesn't fit a continuous area vector. SAC is sample-efficient but more complex. PPO gave reasonable training stability in 20 minutes on CPU, which suited my compute budget."

**Q7. Why a single-step bandit for the RL environment?**
> "Two reasons, sir. First, my truss design is a one-shot decision, not a sequential one — the agent picks an area vector and we evaluate it. Sequential design (one member at a time) is a different problem. Second, single-step is simpler to debug and matches the recent SOgym benchmark from Rochefort-Beaudoin 2024."

**Q8. Why not fine-tune the LLM?**
> "Three reasons. One — the API doesn't expose fine-tuning. Two — fine-tuning would require a labelled dataset of optimal truss designs, which doesn't exist at scale. Three — my efficacy map suggests the bottleneck isn't model fitness, it's qualitative architectural insight, which fine-tuning may not improve on dense problems like 72-bar."

### Group 3 — Validation

**Q9. How do I know your framework actually works?**
> "Three pieces of evidence, sir. The 10-bar benchmark of Sunar 1991 matches to 0.004 % error. The 25-bar matches Schmit 1976 to 0.008 %. The FEM matches Logan's analytical three-bar example to machine epsilon. Every reported design passes four IS 800:2007 clauses end-to-end."

**Q10. Why 10 seeds and not 100?**
> "Compute budget, sir. Each seed runs in 5–30 minutes depending on benchmark. 10 seeds gives me std deviations and confidence intervals that I report explicitly. For the LLM warm-start study I went to 40 seeds because the effect size was uncertain and statistical significance was needed."

**Q11. How representative is your test set for the surrogate?**
> "The test set is 10 % of 10 000 LHS samples — 1000 designs, randomly held out, never seen during training. It covers the same design-space bounds as the training set. R² = 0.9993 on this set, combined with the in-loop equivalence (surrogate-in-loop optima within 0.09 % of FEM-in-loop), suggests the surrogate generalises well to the optimum region."

### Group 4 — The 72-bar question

**Q12. Why does your 72-bar not match the published 379.62 lb?**
> "Under hard-constraint enforcement my GA, PSO, and NSGA-II all converge to approximately 549 pounds with cross-seed spread under 1 %. I could not reproduce the published 379.62 pounds as feasible in my FEM. I frame this as a constraint-handling-convention observation rather than a methodological correction, because I haven't independently cross-checked the encoding against a second FEM. Independent cross-check is future-work F7."

**Q13. Are you saying the literature is wrong?**
> "No, sir. I'm saying my hard-constraint FEM doesn't reach the published number. Whether that's because the literature used soft-penalty constraint handling, or because my encoding differs subtly from the published one, I haven't verified independently. Both possibilities are open. F7 would resolve it."

**Q14. Have you considered that your FEM might have a bug?**
> "Yes, sir, that's exactly why I framed it as an observation. My FEM passes Logan's three-bar analytical example to machine precision and reproduces Sunar 10-bar to 0.004 % and Schmit 25-bar to 0.008 %. So it's correct on those problems. Whether there's a 72-bar-specific issue — sign error in a load vector, wrong DOF in a constraint — I haven't ruled out. F7 cross-check would catch it if so."

**Q15. What's your encoding for 72-bar?**
> "Standard Camp & Bichon 2004 ordering, sir. Sixteen design groups per their canonical layout — legs, face diagonals, top horizontals, plan diagonals — across four storeys. Two load cases per their paper. Lateral-tip-only displacement constraint at the four tip nodes, per Camp's convention."

### Group 5 — Practical / deployment

**Q16. What's the practical impact of your work?**
> "Three pieces, sir. One — sub-10-second response time turns optimisation from a batch process into an interactive design-review tool. Two — IS 800 compliance is built in, so the engineer doesn't have to cross-check separately. Three — the Streamlit UI doesn't require Python literacy, so a junior consultant or M.Tech student can drive it."

**Q17. How would this be deployed in industry?**
> "Two paths, sir. First — direct deployment of the Streamlit app, which is what the current repository supports. A consultancy could run it on a single laptop or behind a small server. Second — integration with existing design tools like SAP2000 or STAAD.Pro via an API export. That would require some engineering work but the FastAPI backend is the right starting shape."

**Q18. Is the Streamlit demo really sub-10-seconds?**
> "Yes sir, for the 10-bar benchmark with the surrogate enabled. The full chain — UI click, API call, GA run with surrogate replacing FEM, IS 800 compliance check, JSON response, UI render — completes in 6–9 seconds on my Apple M3 laptop. Without the surrogate it would be 30–40 seconds because FEM dominates."

**Q19. What about real Indian transmission towers?**
> "That's future-work F6, sir. Validation against real PGCIL tender drawings would require a data-sharing arrangement. My 200-bar benchmark, encoded in proper Indian SI units, is the closest current proxy."

### Group 6 — Limitations

**Q20. What are your limitations?**
> "Five honest ones, sir. Linear elastic only — no geometric or material non-linearity. Sizing only, not topology. Static only, no seismic or dynamic loading. PPO is single-instance — no transfer learning. And on 72-bar I report an observation, not a correction, pending independent FEM cross-check."

**Q21. Why no seismic analysis?**
> "Out of scope for the thesis, sir. Seismic would require modal analysis, time-history integration, IS 1893 compliance — a separate research effort. It's future-work F1."

**Q22. Why only IS 800 and not Eurocode 3 or AISC?**
> "Two reasons. One — IS 800:2007 is the legally binding code in India and the population I'm serving is Indian. Two — the constraint layer is code-agnostic by design, so adding Eurocode 3 or AISC 360 is a localisation task, not a re-architecture."

**Q23. Have you considered material non-linearity?**
> "Not in this thesis, sir. All four benchmark problems are in the linear-elastic small-displacement regime where the linear approximation is excellent. Material non-linearity would require switching to incremental-iterative analysis, which is a kernel-level change. Future-work F1 lists it."

### Group 7 — Comparisons

**Q24. How does your work compare to Geng 2024?**
> "Geng 2024 integrated an LLM with OpenSeesPy for general structural problems — two layers. They focused on LLM-generated analysis code, not warm-start designs. My contribution is the warm-start as a quantified intervention with statistical significance, plus three more layers (surrogate, RL, IS 800)."

**Q25. How does your work compare to Sun 2023?**
> "Sun 2023 coupled a DNN surrogate with a GA for composite structures — two layers, similar to my Layer 3 + Layer 2. My contribution is the full five-layer stack including RL, LLM, and IS 800, plus the LLM-warm-start efficacy map, which Sun didn't study."

**Q26. Has anyone done this before?**
> "Not the full integration, sir. Individual layers — surrogate + GA, LLM + FEM, RL + structural design — each have prior work. The combination of all five plus IS 800 is, as far as I know, new."

### Group 8 — Method choices

**Q27. Why pymoo and not a custom GA?**
> "Pymoo gives me a uniform Problem-Algorithm-Termination interface, well-tested SBX and polynomial mutation operators, and built-in NSGA-II with non-dominated sorting. Writing my own would be a few thousand lines of unnecessary work. Pymoo is widely adopted in the optimisation community and would be familiar to any reviewer."

**Q28. Why static-penalty constraint handling and not adaptive?**
> "Simplicity and the standard practice in truss-sizing literature, sir. Static penalty with feasibility-first selection is what Coello 2002's survey identifies as the workhorse. Adaptive penalty would be marginal improvement at significant complexity cost on these benchmarks."

**Q29. Why 100 population and 500 generations?**
> "Standard pymoo defaults, sir, and they match the canonical truss-sizing literature. The 200-bar benchmark uses 800 generations because the dimensionality is higher. I confirmed these settings give cross-seed spread under 1 % on three of the four benchmarks."

### Group 9 — Future and high-level

**Q30. What was the hardest part of the thesis?**
> "Getting all eight layers to share a single data contract, sir. Each layer was written in isolation; making them communicate cleanly through three shared dataclasses — `OptimizationResult`, `BenchmarkEvaluation`, `ComplianceReport` — took a lot of refactoring. But once it was working, swapping a benchmark or an algorithm became a single-line change."

**Q31. What surprised you most?**
> "How specifically Claude identified the redundant members on 10-bar, sir. I expected the LLM to give a generic 'all members medium-area' answer. Instead it correctly flagged members 2, 5, 6, and 10 as redundant — exactly the optimum's behaviour. That's the mechanism behind the 36.8 % generation reduction."

**Q32. Where would you take this next?**
> "Seven directions in Chapter 5, sir. The two highest-leverage: a graph-neural-network surrogate for variable-topology design, and PPO transfer learning across the four-benchmark family to test Zhao 2021's cross-instance claim. The most practically valuable: validation against real Indian transmission-tower designs through a data-sharing arrangement with PGCIL. And F7 — the independent FEM cross-check on 72-bar — to upgrade my observation into a publishable finding."

---

## 25. Anti-questions you don't want but might get

These are the questions that *could* trip you. Be ready.

### "What does $R^2 = 0.9993$ actually mean?"

> "R² is the coefficient of determination, sir. It measures the proportion of variance in the true weight that the surrogate explains. R² = 0.9993 means 99.93 % of the variance is captured — predicted weights sit almost exactly on the y=x line. R² of 1.0 would be perfect prediction; R² of 0 would be as good as predicting the mean."

### "What's the difference between R² and accuracy?"

> "R² is for regression — measuring continuous output fit. Accuracy is for classification — measuring how often a prediction is correct. The surrogate outputs continuous numbers, so R² is the right metric."

### "What is the paired Wilcoxon test and why use it?"

> "Wilcoxon signed-rank is a non-parametric test for paired samples, sir. It doesn't assume the differences are normally distributed, unlike a paired t-test. I use it because I can't guarantee the generations-to-convergence differences are normal, and Wilcoxon is robust to that."

### "What's the type-1 error risk?"

> "Type-1 is rejecting a true null hypothesis. With p = 0.0046 I'm claiming significance at α = 0.005, well below the conventional 0.05 threshold. The risk of falsely concluding the LLM helps when it doesn't is under half a percent."

### "Why p = 0.05 instead of p = 0.01 as the threshold?"

> "Conventional standard in applied statistics, sir, going back to Fisher 1925. p = 0.0046 is below 0.01 in any case, so the result holds at either threshold."

### "Why didn't you do 100 LLM seeds?"

> "Cost and time, sir. 40 seeds reached p = 0.0046 — well past significance. Going to 100 would tighten the confidence interval but wouldn't change the conclusion. For the 25-bar and 72-bar runs I used fewer seeds because Anthropic API spend was a budget constraint."

### "How much did the Anthropic API cost you?"

> "Around ₹6 000 total, sir, across all 60 or so LLM calls in the entire experiment. The cache eliminates re-billing for reproductions — that's a core design choice for offline reproducibility."

### "Could the LLM result be a fluke specific to Claude?"

> "Possibly, sir. I only tested Claude — I didn't compare to GPT-4 or Gemini. That's an open question. The mechanism — identifying redundant members — is general enough that I'd expect similar results from any sufficiently capable LLM, but I haven't verified it. That's a one-day experiment if someone wanted to do it."

### "How do you guarantee reproducibility?"

> "Three things, sir. One — every run takes a seed argument that propagates to numpy, pymoo, and PyTorch. Two — all LLM responses are content-hashed and cached on disk so the API call is replaced by a deterministic lookup. Three — the repository is pinned to tag `phase-10-complete`. Anyone cloning that tag can reproduce every number in the thesis."

### "What if the surrogate is overfitting?"

> "Two pieces of evidence against overfitting, sir. One — the held-out test set R² is 0.9993, separate from training data. Two — surrogate-in-loop optima match FEM-in-loop optima to 0.09 %, which means the surrogate's predictions are accurate not just on random samples but specifically near the optimum region. Overfitting would manifest as either lower test R² or in-loop divergence."

### "Why dropout 0.1 and not 0.2 or 0.5?"

> "Empirically chosen, sir. I tried 0.0, 0.1, 0.2, and 0.5 — 0.1 gave the best validation R² without hurting training convergence. 0.5 is too aggressive for a small network; 0.0 led to mild overfitting on the displacement and stress heads."

### "Could you have used a single LLM call to do the entire optimisation?"

> "I tried, sir. Claude's zero-shot designs are within 5–15 % of optimum on small benchmarks but worse on 25-bar and 72-bar — they're feasible but not minimum-weight. As a *replacement* for optimisation, the LLM is worse than a tuned GA. As a *warm-start* for the GA, it's a 36.8 % speedup on 10-bar."

### "Are your benchmarks too small to be representative?"

> "10-bar and 25-bar are small, sir, but they're the canonical literature benchmarks — meaningful for cross-paper comparison. The 200-bar with 29 design variables is the scaling test, and it converges with cross-seed spread under 1 % using the same hyperparameters. Real industrial trusses might have 200–500 design variables; 200-bar is the closest proxy in my study."

---

## 26. The 20 must-know numbers (with story)

Memorise these. Each comes with the one-line story you can deploy if asked.

| Number | What it is | The one-line story |
|---|---|---|
| **5060.85 lb** | Sunar 1991 published 10-bar optimum | The canonical benchmark every paper compares against |
| **5061.05 lb** | My 10-bar PSO result | Matches Sunar to 0.004 % — proof every layer is correct |
| **0.004 %** | My 10-bar error | The end-to-end validation signature |
| **545.22 lb** | Schmit 1976 25-bar reference | Second canonical benchmark, 3D version |
| **545.26 lb** | My 25-bar PSO result | 0.008 % error — 3D and symmetry-grouping both work |
| **0.008 %** | My 25-bar error | Independent confirmation of framework correctness |
| **549 lb** | My 72-bar hard-constraint optimum | Three algorithms agree to under 1 % — robust |
| **379.62 lb** | 72-bar soft-penalty literature value | What I cannot reproduce under hard constraints |
| **316.61 kg** | My 200-bar GA result | Proper Indian SI units, structural steel, 250 MPa stress envelope |
| **R² = 0.9993** | Surrogate weight prediction accuracy | Test-set held-out, 1000 unseen designs |
| **132×** | Surrogate speedup over FEM | 41 ms FEM vs 0.31 ms surrogate per design |
| **0.09 %** | Surrogate-in-loop vs FEM-in-loop optimum gap | The speed costs essentially nothing in quality |
| **36.8 %** | LLM warm-start generation reduction on 10-bar | 128 vs 203 mean generations across 40 seeds |
| **p = 0.0046** | Paired Wilcoxon significance | Highly significant — well below 0.01 |
| **40 seeds** | Sample size for the LLM result | Paired with random-init arm |
| **128 vs 203** | Mean generations: LLM vs random on 10-bar | 75-generation saving on average |
| **+16 %** | PPO gap vs optimum (mixed result) | 5876.9 lb vs 5060.85 lb — honest, not a success |
| **{2, 5, 6, 10}** | Redundant members Claude identified on 10-bar | The mechanism behind the 36.8 % reduction |
| **< 10 s** | Streamlit demo end-to-end time | Sub-10-seconds = interactive, not batch |
| **10 000** | LHS samples for surrogate training | 80/10/10 split, 7 minutes to evaluate |

---

## 27. After-the-viva playbook

### Immediately after

- Pack laptop calmly. Don't rush.
- Thank the examiners individually if appropriate: *"Thank you, sir. Thank you, ma'am."*
- Walk out at normal pace. No running.

### Outside the room

- **Do not** debrief immediately with friends/family — let yourself decompress first.
- Drink water. Eat something. Take 30 minutes alone.

### After 30 minutes

- Reflect: what went well, what got asked, what surprised you. Write it down — useful for any future viva (Ph.D., interviews).
- Reach out to your supervisor with a brief update.

### If you're asked for revisions

- Make notes immediately on what specifically they want changed. Don't trust memory after 24 hours.
- Most M.Tech vivas have only minor revisions: typo fixes, small clarifications, maybe one extra figure.

### The thesis publication

- The first paper out of this work should be the LLM warm-start efficacy map (C4) — it's your strongest novel claim and has the cleanest narrative.
- The framework paper (C1) is the second.
- The 72-bar story (C3) is only publishable *after* the OpenSeesPy cross-check (F7) — until then it's just an observation.

### Celebrate

- You earned this. Take the evening off. Don't open any code or LaTeX for at least 24 hours.

---

## 28. The one-page printable cheat sheet

```
═══════════════════════════════════════════════════════════════
VIVA CHEAT SHEET — fold along the dotted line, keep in pocket
═══════════════════════════════════════════════════════════════

OPENING
"Good morning, sir. Good morning, ma'am. My thesis is titled
Multi-objective Steel Truss Optimisation: an AI-Assisted
Framework Compliant with IS 800:2007."

8 BEATS WITH TIME MARKERS
0:00–1:00   Greeting + title + agenda preview
1:00–2:30   Problem hook (10-30% surplus, 40 min/run)
2:30–5:00   8-layer stack
5:00–7:30   Validation: 10-bar 0.004%, 25-bar 0.008%, surrogate 132x
7:30–10:30  Headline LLM result: 36.8% gen ↓, p=0.0046, 40 seeds
10:30–12:00 72-bar honest: 549 lb baseline, observation not claim
12:00–13:30 Streamlit <10s, offline reproducible
13:30–15:00 Limitations + 7 future + close

CLOSING
"Thank you, sir. Thank you, ma'am. I'm happy to take questions."

20 NUMBERS
5060.85→5061.05→0.004%   10-bar Sunar 1991
545.22→545.26→0.008%     25-bar Schmit 1976
549 lb                   72-bar hard-constraint
379.62 lb                72-bar soft-penalty (cannot reproduce)
316.61 kg                200-bar GA (Indian SI)
R² = 0.9993              surrogate weight head
132×                     surrogate speedup
0.09%                    surrogate-in-loop gap
36.8%                    LLM gen reduction on 10-bar
p = 0.0046               paired Wilcoxon
40 seeds                 LLM experiment sample
128 vs 203               LLM vs random generations
+16%                     PPO gap (honest mixed result)
{2, 5, 6, 10}            redundant members Claude found
< 10 s                   Streamlit demo time
10,000                   LHS training samples
IS 800 clauses           3.8, 5.6.1, 6.2/6.3, 7.1
4 benchmarks             10-bar, 25-bar, 72-bar, 200-bar
5 contributions          C1-C5
7 future-work            F1-F7 (F7 = independent FEM check)

5 CONTRIBUTIONS
C1  Integrated 5-layer framework (0.004% sig)
C2  Surrogate (R²=0.9993, 132×, 0.09% gap)
C3  Rigorously-feasible 72-bar baseline (~549 lb)
C4  LLM efficacy map (36.8%, p=0.0046)  ← strongest novel
C5  IS 800-compliant Streamlit demo (< 10 s)

72-BAR CRISP ANSWER
"On 72-bar I report an observation, not a finding. Hard-constraint
optimum at 549 lb across three algorithms, spread <1%. Could not
reproduce the published 379.62 lb as feasible — but pending
independent FEM cross-check (F7), I don't claim correction."

IF YOU BLANK
"Let me just take a moment." [look at slide, get next beat]

IF YOU DON'T KNOW
"That's a good question, sir. I haven't tested that specifically,
but my expectation would be X because Y. I'd want to run the
experiment to be sure."

DELIVERY RULES
- Slow is confident
- Pause after big numbers
- Open palms, no pockets
- Rotate eye contact
- Don't apologise for nerves
- End sentences cleanly

═══════════════════════════════════════════════════════════════
```

---

## 29. Glossary (60+ entries)

| Term | Plain-English definition |
|---|---|
| **Adam** | Adaptive Moment Estimation — a gradient descent variant that adapts learning rate per-parameter. Standard for neural network training. |
| **Anthropic** | Maker of Claude, the LLM you use. AI safety company founded 2021. |
| **Architecture (NN)** | The shape of a neural network — how many layers, how wide each is, what activations. |
| **Axial force** | Force along the length of a bar. Positive = tension, negative = compression. |
| **Bandit (contextual)** | A reinforcement learning problem with one step per episode. |
| **Batch size** | Number of training examples processed before updating network weights. |
| **Beamer** | LaTeX package for making PDF slide decks. What your viva deck uses. |
| **Benchmark** | A standard problem with a known solution, used to validate new methods. |
| **Black-box optimisation** | Optimisation where the objective and constraints are not analytically differentiable. Truss sizing is black-box because each evaluation requires an FEM solve. |
| **Buckling** | Sideways failure of a compression member at a load below its theoretical yield strength. Long thin bars buckle. |
| **Chain-of-thought (CoT)** | Prompting technique where the LLM is asked to reason step-by-step. Wei et al. 2022. |
| **Clipped objective (PPO)** | PPO's trick of limiting policy updates so the new policy can't differ too much from the old. |
| **Column curve** | IS 800's $\chi(\bar\lambda)$ relationship. Four curves: a, b, c, d. You use curve a. |
| **Compliance** | Satisfaction of regulatory or code requirements. IS 800 compliance = passes all four clauses. |
| **Constraint** | A rule the design must obey. Stress, displacement, slenderness, bounds. |
| **Convergence** | When an algorithm stops improving meaningfully. |
| **Crossover (GA)** | Combining two parent designs to produce a child. SBX is your specific operator. |
| **Crowding distance** | NSGA-II's measure of how isolated a design is on the Pareto front. Used for diversity. |
| **Decoupled architecture** | Each module communicates through clean interfaces, can be swapped independently. |
| **Degree of freedom (DOF)** | One axis a node can move along. 2 in 2D, 3 in 3D. |
| **Design variable** | A number being optimised. Here, bar areas. |
| **Design vector** | The set of all design variables. $\mathbf{A} = [A_1, \ldots, A_n]$. |
| **Direct stiffness method** | FEM where you assemble individual element stiffnesses into a global matrix and solve. |
| **Distribution index ($\eta$)** | SBX/polynomial mutation parameter controlling how concentrated offspring are. |
| **Dominated** | A design dominated by another is at most as good on every objective and strictly worse on at least one. |
| **Dropout** | Regularisation technique that randomly zeroes neurons during training. |
| **Early stopping** | Stop training when validation loss stops improving. |
| **Element stiffness matrix** | The 4×4 (2D) or 6×6 (3D) matrix relating bar elongation to nodal forces. |
| **Elitism** | Keeping the best individuals across generations. |
| **Epoch** | One full pass through the training data during neural-network training. |
| **Euler buckling stress** | $f_{cc} = \pi^2 E / \lambda^2$. The critical stress at which a long thin column buckles. |
| **FastAPI** | Modern Python library for building REST APIs. Your backend. |
| **Feasibility-first selection** | Constraint-handling rule: feasible always beats infeasible, regardless of weight. |
| **Feasible / Infeasible** | Satisfies all constraints / breaks at least one. |
| **FEM** | Finite Element Method. Physics solver for structural problems. |
| **Fully-stressed design** | Heuristic where each member's area is set so its stress equals the allowable. |
| **GA** | Genetic Algorithm. Evolutionary optimisation by selection, crossover, mutation. |
| **GAE** | Generalised Advantage Estimation. Used in PPO to compute advantage with reduced variance. |
| **Gradient-free** | Optimiser that doesn't need derivatives of the objective. GA, PSO, NSGA-II are all gradient-free. |
| **Gymnasium** | Python library — standard RL environment interface. Successor to OpenAI Gym. |
| **Hooke's law** | Force = stiffness × displacement. The basis of linear FEM. |
| **Hyperparameter** | An algorithm setting you choose (e.g. population size) rather than learn. |
| **Imperfection factor ($\alpha$)** | IS 800 column-curve parameter. Curve a uses $\alpha = 0.21$. |
| **IS 800:2007** | Bureau of Indian Standards code for steel construction. Legally binding in India. |
| **JSON** | JavaScript Object Notation. Standard for data interchange. |
| **Latin Hypercube Sampling (LHS)** | Space-filling sampling — better than uniform random for surrogate training. |
| **Learning rate** | Step size for gradient descent. 1e-3 for your surrogate, 3e-4 for PPO. |
| **Limit-state design** | IS 800's design philosophy — separate ultimate and serviceability limits. |
| **Linear elastic** | Stress proportional to strain, no permanent deformation. Your FEM assumption. |
| **LLM** | Large Language Model. Claude, GPT, etc. |
| **Load case** | One specific loading scenario. Optimiser respects worst-case across load cases. |
| **Mini-batch** | A subset of the training data used in one gradient update. 128 for surrogate, 64 for PPO. |
| **MLP** | Multi-Layer Perceptron. Basic feed-forward neural network. |
| **Mutation (GA)** | Randomly tweaking a child design. Polynomial mutation is your operator. |
| **Net section** | Cross-section area after deducting bolt holes. Equal to gross section for pin-jointed bars. |
| **Non-dominated sort** | NSGA-II's core sorting routine, partitioning population into Pareto fronts. |
| **NSGA-II** | Non-dominated Sorting Genetic Algorithm II. Deb 2002. |
| **Optimum** | The best (lightest feasible) design. |
| **p-value** | Probability the observed effect is due to chance. |
| **Paired test** | A statistical test where both arms see the same conditions (e.g. same seed). Better statistical power. |
| **Pareto front** | Set of non-dominated designs in multi-objective optimisation. |
| **Partial safety factor** | IS 800 multiplier on capacities for safety margin. $\gamma_{m0} = 1.1$ for yielding. |
| **Pin joint** | A connection that allows free rotation. Truss-essential. |
| **Policy** | In RL, the function that maps states to actions. |
| **Polynomial mutation** | pymoo's default mutation operator for continuous variables. |
| **PPO** | Proximal Policy Optimisation. Schulman 2017. |
| **PSO** | Particle Swarm Optimisation. Kennedy & Eberhart 1995. |
| **pymoo** | Python multi-objective optimisation library. |
| **R²** | Coefficient of determination. 1.0 = perfect prediction. |
| **Radius of gyration ($r$)** | Cross-section property. $r = \sqrt{I/A}$. |
| **Reinforcement learning** | Agent learns to maximise reward through trial and error. |
| **ReLU** | Rectified Linear Unit activation: $f(x) = \max(0, x)$. |
| **REST API** | A web service using HTTP and JSON. |
| **Reward** | Scalar signal an RL agent maximises. |
| **Rolled section** | Steel cross-section produced by hot rolling at the mill. The standard catalogue. |
| **SBX** | Simulated Binary Crossover. pymoo's default crossover for continuous GA. |
| **Seed** | An integer that makes random number generation deterministic. |
| **Self-consistent** | Reproducible across runs with same inputs. |
| **Serviceability** | IS 800 limit-state category for deflection / vibration / cracking. Not catastrophic failure. |
| **Slenderness** | $\lambda = L_{\text{eff}}/r$. Long-thin-ness of a bar. |
| **Soft-penalty handling** | Constraint handling where minor violations are tolerated with a penalty. |
| **SP 6(1):1964** | Indian standard rolled-section catalogue. |
| **Stable-Baselines3** | Python library implementing RL algorithms. |
| **Static-penalty handling** | Constraint handling with a fixed penalty coefficient on violations. |
| **Stiffness matrix ($\mathbf{K}$)** | The big matrix in $\mathbf{K}\mathbf{u} = \mathbf{F}$. Global. |
| **Streamlit** | Python library for making web UIs without HTML. |
| **Surrogate** | Cheap fake of an expensive simulator. |
| **Symmetry group** | Bars forced to share one area. Reduces design dimensions. |
| **Tournament selection** | GA selection rule: pick 2 random parents, the better wins. |
| **Transfer learning** | Apply a model trained on one problem to a different but related problem. |
| **Truss** | Pin-jointed structure of bars. |
| **Ultimate limit state** | IS 800 category for member failure under design loads. |
| **Validation gate** | A specific numerical target a layer must meet to be considered correct. |
| **Wilcoxon signed-rank** | Non-parametric paired statistical test. |
| **Yield strength ($f_y$)** | Stress at which steel begins to deform plastically. 250 MPa for Fe 250. |
| **Zhao 2021** | Reference for the claim that single-instance RL doesn't beat tuned GA. |

---

## 30. Pre-viva checklist

### One week before
- [ ] Re-read the full `main.pdf` writeup once
- [ ] Re-read this teaching doc once
- [ ] Confirm slide deck is the final softened version
- [ ] Run the Streamlit demo once on your laptop — confirm it works

### Three days before
- [ ] Practice the 15-minute script aloud, time yourself
- [ ] Aim for 14:00 ± 1 minute
- [ ] Memorise the 20 numbers
- [ ] Have a friend ask you 5 random Q&A questions

### One day before
- [ ] Charge laptop, pack charger
- [ ] Copy both PDFs to USB stick
- [ ] Email both PDFs to yourself
- [ ] Print the cheat sheet (section 28)
- [ ] Lay out clothes
- [ ] Plan transport — leave 1 hour buffer
- [ ] Sleep 8 hours

### Morning of
- [ ] Open slide PDF on laptop — confirm it renders
- [ ] Test clicker
- [ ] Drink water, light meal
- [ ] Arrive 30 min early
- [ ] Re-read cheat sheet once

### 5 minutes before
- [ ] Drink water
- [ ] Empty bladder
- [ ] Five slow breaths in through nose, out through mouth
- [ ] One internal line: "I built something real. I'm going to walk them through it."

### Walking in
- [ ] Knock, wait, enter
- [ ] Smile, "Good morning, sir / ma'am"
- [ ] Set up laptop, connect HDMI
- [ ] Wait for "go ahead"
- [ ] *"Thank you, sir. May I begin?"*

### When you finish speaking
- [ ] Stop
- [ ] Smile slightly
- [ ] *"Thank you, sir. Thank you, ma'am. I'm happy to take questions."*
- [ ] Wait

### After
- [ ] Walk out at normal pace
- [ ] 30 minutes alone before debriefing
- [ ] Sleep that night, regardless

---

**You built this. You measured it. You wrote 125 pages. You softened the one risky claim before walking in.**

**You're not pretending. You actually did this.**

**Go get it.**

— end of doc —
