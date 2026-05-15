# Viva Teaching Doc — Everything You Need to Know About Your Own Thesis

**Author:** Piyush · **Programme:** M.Tech, Civil Engineering, IIT BHU
**Thesis:** *Multi-objective Steel Truss Optimisation: an AI-Assisted Framework Compliant with IS 800:2007*
**Repository tag:** `phase-10-complete`

---

## How to use this doc

Read it in chunks. Don't try to absorb it all at once.

- **Day 1:** Sections 0–3 (foundations + framework overview)
- **Day 2:** Sections 4–6 (each AI layer + benchmarks)
- **Day 3:** Sections 7–9 (results + novel finding + 72-bar honest story)
- **Day 4:** Sections 10–11 (viva script + numbers)
- **Day 5:** Section 12 (Q&A bank, relaxed review)

If anything is unclear, the chapter number in `thesis_writeup/chapters/` is given in square brackets.

---

# TABLE OF CONTENTS

0. The thesis in one paragraph
1. Foundations — what's a truss, what's optimisation
2. The eight-layer framework
3. The four canonical benchmarks
4. The neural surrogate
5. The PPO reinforcement-learning agent
6. The LLM warm-start designer
7. The IS 800:2007 compliance layer
8. The Streamlit + FastAPI deliverable
9. The five contributions
10. The 72-bar honest story (the softened claim)
11. The 15-minute viva script
12. The Q&A bank
13. The 20 must-know numbers
14. Glossary

---

# 0. THE THESIS IN ONE PARAGRAPH

Steel trusses in Indian civil practice are designed today by trial-and-error against the IS 800:2007 code, leaving 10–30 % surplus steel on the table. Classical optimisers can fix this, but they are too slow to be interactive. This thesis combines classical optimisers (GA, PSO, NSGA-II) with three modern AI accelerators (a neural surrogate, a PPO agent, and an LLM warm-start) plus an IS 800 compliance checker, into one Python framework. The framework matches the canonical 10-bar literature optimum to 0.004 % error, runs 132× faster via the surrogate, demonstrates a statistically significant LLM warm-start benefit (36.8 % generation reduction, p = 0.0046, 40 seeds), and ships as a Streamlit web app that completes a full optimisation in under 10 seconds on a laptop CPU.

---

# 1. FOUNDATIONS — WHAT'S A TRUSS, WHAT'S OPTIMISATION

## 1.1 What is a truss

A **truss** is a structure made of straight steel bars (called **members**) joined together at points (called **nodes**). Each member carries load primarily along its length — either pulling (tension) or pushing (compression). No bending.

You see trusses everywhere in Indian infrastructure:
- Factory roof frames (industrial sheds)
- Electrical transmission towers
- Bridges (highway / railway)
- Stadium canopies

The reason engineers love trusses is that **axial force is the most efficient way a steel bar can carry load**. Bending wastes material.

## 1.2 What does designing a truss mean

Two things:
1. **Topology + Shape** — *Where* are the nodes, *which* bars connect them. **Fixed** in your thesis — set by the benchmark.
2. **Sizing** — *How thick* is each bar's cross-section. **What you optimise.**

Each bar has a cross-sectional area, measured in m² (SI) or in² (imperial).

## 1.3 Why does it matter

Three reasons:
- **Cost:** Less steel = less rupees.
- **Carbon:** Steel = 1.9 kg CO₂ per kg, structural steel is high-embodied-carbon.
- **Sector scale:** India builds thousands of these every year. Tiny per-design savings × thousands of structures = real national-scale numbers.

A 15 % steel-weight saving across the population of small-to-medium Indian trusses is measurable.

## 1.4 The four constraints (what makes a design "feasible")

1. **Stress constraint** — Force inside each bar ÷ area must not exceed the steel's stress limit. Otherwise the bar breaks.
2. **Displacement constraint** — No node can move more than a small amount under load. Otherwise the structure sags too much, cracks brittle cladding, etc.
3. **Slenderness constraint** — Long thin bars in compression buckle (bend sideways). IS 800:2007 limits λ ≤ 180 in compression, λ ≤ 400 in tension.
4. **Bound constraint** — Each area between a minimum (smallest rolled section in the catalogue) and a maximum.

A design satisfying ALL four is **feasible**. Otherwise **infeasible**.

## 1.5 What is the optimum

The **optimum** = the lightest feasible design.

Mathematically:
```
minimise   W(A) = Σ ρ · L_i · A_i      (total weight)

subject to:
  stress_i(A) ≤ σ_max        for every member i
  displacement_j(A) ≤ δ_max  for every node j
  λ_i(A) ≤ λ_max             slenderness
  A_min ≤ A_i ≤ A_max        bounds
```

The optimum is a specific area vector $[A_1^*, A_2^*, \ldots, A_n^*]$ that minimises weight while satisfying all constraints.

## 1.6 Why is this hard

The space is high-dimensional and non-convex. For 10 design variables with say 100 discrete cross-sections each, that's 10²⁰ candidate designs. You cannot brute-force search it. You need an optimisation algorithm.

## 1.7 Key vocabulary

| Word | Meaning |
|---|---|
| **Design variable** | A number you can change. Areas of each bar (or each *group* of bars if symmetric). |
| **Objective** | The thing you minimise. Total weight, here. |
| **Constraint** | A rule the design must obey. |
| **Feasible / Infeasible** | Obeys / breaks at least one constraint. |
| **FEM** | Finite Element Method. The physics engine. |
| **Benchmark** | A standard test problem with a known answer. |
| **Symmetry group** | A set of bars forced to share one area (reduces design dimensions). |

---

# 2. THE EIGHT-LAYER FRAMEWORK

This is the architecture you built. Each layer has a clear input/output contract.

```
┌──────────────────────────────────────────────┐
│  USER  picks benchmark + algo + toggles       │
└────────────────┬─────────────────────────────┘
                 │
        ┌────────▼────────┐
        │  Streamlit UI   │   src/app/ui.py
        └────────┬────────┘
                 │ HTTP
        ┌────────▼────────┐
        │   FastAPI       │   src/app/api.py
        └────────┬────────┘
                 │
┌────────────────┼─────────────────────────────┐
│                │                              │
▼                ▼                              ▼
┌─────────┐   ┌────────────┐               ┌──────────┐
│ LLM     │   │ GA/PSO/    │               │ PPO RL   │
│warmstart│──▶│ NSGA-II    │               │ agent    │
│(Claude) │   │ (pymoo)    │               │ (SB3)    │
└─────────┘   └─────┬──────┘               └────┬─────┘
                    │                            │
                    │ many calls                 │
                    ▼                            ▼
              ┌─────────────────────────────────────┐
              │ Surrogate (132×)  │     FEM          │
              └────────┬──────────────────┬──────────┘
                       │                  │
                       ▼                  ▼
                ┌──────────────────────────┐
                │ IS 800 compliance        │
                └──────────────────────────┘
```

## 2.1 Layer 1: FEM solver — `src/fem/`  [Ch 3 §3.2]

**Job:** Given areas + geometry + loads, return forces, stresses, displacements.

**How it works:**

For each bar element with cross-section A, length L, modulus E:

$$\mathbf{k}_e = \frac{AE}{L}\,\mathbf{T}^\top\!\begin{bmatrix}1 & -1 \\ -1 & 1\end{bmatrix}\!\mathbf{T}$$

This is Hooke's law in matrix form. $\mathbf{T}$ is a rotation matrix aligning the bar's local axis to global x-y-z.

Assemble all $\mathbf{k}_e$ into one global stiffness matrix $\mathbf{K}$. Solve:

$$\mathbf{K}\,\mathbf{u} = \mathbf{F}$$

where $\mathbf{F}$ is the load vector and $\mathbf{u}$ is the unknown displacement vector. Use `scipy.linalg.solve`. Back-substitute to get bar forces.

**Validation:** Matches Logan's three-bar textbook example (which has an exact analytical solution) to ≈ 1e-21 — that's machine epsilon. So your FEM is correct beyond any reasonable doubt.

**Wall-clock cost:** 41 ms per design. This is the bottleneck the surrogate solves.

## 2.2 Layer 2: Benchmarks — `src/benchmarks/`  [Ch 3 §3.3]

Each benchmark is a class subclassing `BenchmarkProblem`. It bundles:
- Node coordinates
- Element connectivity
- Material properties
- Load cases
- Support conditions
- Design-variable bounds
- Symmetry groups
- Published optimum (for validation)

All four benchmarks expose the same `evaluate(x)` method. So the optimisers don't need to know which benchmark they're running.

## 2.3 Layer 3: Classical optimisers — `src/algorithms/`  [Ch 3 §3.4]

Thin pymoo wrappers around a uniform `TrussProblem` adapter.

### 2.3.1 Genetic Algorithm (GA)

**Analogy:** Breeding cows. Best milk-producers reproduce; worst get culled. Over many generations the herd converges on great milk-producers.

**Mechanism:**
1. Start with 100 random designs (population).
2. Evaluate each: weight + constraint violations.
3. Rank: feasible > infeasible; lighter > heavier.
4. **Tournament-select** parents (pick 2, keep the better).
5. **SBX crossover** — make a child by mixing two parents' areas smoothly.
6. **Polynomial mutation** — randomly tweak the child slightly.
7. Repeat for 500 generations.

**Hyperparameters:** pop=100, gens=500, $\eta_c$=15 ($p_c$=0.9), $\eta_m$=20 ($p_m$=1/n).

### 2.3.2 Particle Swarm Optimisation (PSO)

**Analogy:** A flock of birds searching for food. Each bird remembers its best spot and sees the flock's best spot — and is pulled toward both.

**Mechanism:** Each particle has a position (design) and a velocity (how it's changing). Velocity update:

$$\mathbf{v}_i \leftarrow w\,\mathbf{v}_i + c_1 r_1(\mathbf{p}_i - \mathbf{x}_i) + c_2 r_2(\mathbf{g} - \mathbf{x}_i)$$

- $w$ = inertia (0.9 → 0.4 over the run)
- $c_1$ = pull toward personal best
- $c_2$ = pull toward global best
- $r_1, r_2$ = random numbers

**Hyperparameters:** swarm=100, gens=500, $c_1=c_2=2.0$.

### 2.3.3 NSGA-II — Non-dominated Sorting GA

**Analogy:** Buying a phone — you want it cheap AND high-spec. No single best, but a **Pareto front** of trade-offs.

**Job:** Given two competing objectives (weight + max displacement), find the trade-off curve.

**Mechanism:**
1. Random population.
2. Evaluate both objectives.
3. **Non-dominated sorting:** A dominates B if A is better on every objective. Pareto front = non-dominated designs.
4. **Crowding distance:** Within a front, prefer isolated designs (so front is well-spread).
5. SBX + polynomial mutation (same as GA).
6. Repeat.

**Output:** Not one design, but 20–100 non-dominated designs along the curve.

### 2.3.4 Constraint handling — feasibility-first

All three use the same rule:
- Feasible always beats infeasible.
- Among feasible: lighter wins.
- Among infeasible: less violation wins.

This is pymoo's built-in feasibility-first selection.

## 2.4 Layer 4: Neural surrogate — `src/ml/`  [§4, this doc + Ch 3 §3.5]

A 3-hidden-layer MLP that predicts (weight, displacement, stress) from an area vector — 132× faster than FEM.

(Full deep dive in Section 4 below.)

## 2.5 Layer 5: RL agent — `src/rl/`  [§5, this doc + Ch 3 §3.6]

A PPO agent that proposes a complete area vector in one shot.

(Full deep dive in Section 5 below.)

## 2.6 Layer 6: LLM warm-start — `src/llm/`  [§6, this doc + Ch 3 §3.7]

Claude proposes an initial design in natural language; we inject it into the GA's starting population.

(Full deep dive in Section 6 below.)

## 2.7 Layer 7: IS 800 compliance — `src/constraints/`  [§7, this doc + Ch 3 §3.8]

Checks every design against four IS 800:2007 clauses.

(Full deep dive in Section 7 below.)

## 2.8 Layer 8: UI — `src/app/`  [§8, this doc + Ch 3 §3.9]

FastAPI backend + Streamlit frontend. Sub-10-second user experience.

(Full deep dive in Section 8 below.)

---

# 3. THE FOUR CANONICAL BENCHMARKS

These are standard test problems from the literature. Everyone tests their algorithm against them, so you can quantitatively validate yours.

## 3.1 The 10-bar planar cantilever (Sunar & Belegundu 1991)

| Parameter | Value |
|---|---|
| Dimension | 2D |
| Nodes | 6 |
| Bars | 10 |
| Design variables | 10 (no symmetry) |
| Material | Aluminium, E = 10⁷ psi, ρ = 0.1 lb/in³ |
| Loads | Two vertical −100 kips at the two lower free nodes |
| Stress limit | ±25 000 psi |
| Displacement limit | 2 in |
| Area bounds | [0.1, 35] in² |
| **Published optimum** | **5060.85 lb** |
| **My PSO result** | **5061.05 lb** |
| **Error** | **0.004 %** |

This is your **end-to-end validation signature**. Every layer in the stack — FEM, optimiser, compliance — has to be correct for this number to land here.

## 3.2 The 25-bar spatial tower (Venkayya 1971)

| Parameter | Value |
|---|---|
| Dimension | 3D |
| Nodes | 10 |
| Bars | 25 |
| Design variables | 8 (after symmetry grouping) |
| Material | Aluminium |
| Load cases | 2 |
| **Published optimum** | **545.22 lb** |
| **My PSO result** | **545.26 lb** |
| **Error** | **0.008 %** |

## 3.3 The 72-bar spatial tower (Fleury & Schmit 1980)

| Parameter | Value |
|---|---|
| Dimension | 3D |
| Nodes | 20 |
| Bars | 72 (four-storey tower) |
| Design variables | 16 (after symmetry grouping per Camp & Bichon 2004) |
| Material | Aluminium |
| Load cases | 2 (LC1 asymmetric tip load, LC2 uniform vertical compression) |
| Stress limit | ±25 000 psi |
| Displacement limit | 0.25 in lateral, tip nodes only |
| Area bounds | [0.1, 4.0] in² |
| **Published optimum** | **379.62 lb** (soft-penalty literature value) |
| **My hard-constraint result** | **~549 lb** (cross-seed spread < 1 %) |

(See Section 10 for the full honest story.)

## 3.4 The 200-bar planar stepped tower (Kaveh 2010, re-encoded in SI)

| Parameter | Value |
|---|---|
| Dimension | 2D |
| Nodes | 77 |
| Bars | 200 |
| Design variables | 29 (after symmetry grouping) |
| Material | **Structural steel** (the only benchmark in real Indian-civil units) |
| Young's modulus | 210 GPa |
| Density | 7850 kg/m³ |
| Stress limit | ±250 MPa |
| Load cases | 3 (lateral wind top, asymmetric mid-span, vertical at tip) |
| Total height | 30 m |
| **My GA result** | **316.61 kg** |
| **My PSO result** | **315.89 kg** |
| **My NSGA-II** | 3 seeds, 100-point Pareto fronts |

I explicitly do **not** claim a literature match against Kaveh's 25 445 lb because my geometry is a principled regular-grid stepped tower, not Kaveh's exact coordinates. This is openly stated; the 200-bar's role in the thesis is to prove the framework scales past 72-bar and works in proper SI Indian-civil units.

---

# 4. THE NEURAL SURROGATE — DEEP DIVE  [Ch 3 §3.5, Ch 4 §4.5]

## 4.1 What it is and why

A surrogate is a cheap fake of an expensive simulator. The expensive simulator here is FEM (41 ms / design). The surrogate is a neural network that predicts almost the same answer in 0.31 ms — **132× faster**.

When you put the surrogate inside the GA's inner loop instead of FEM, a 40-minute optimisation drops to about 20 seconds. That's the speed unlock that makes the framework interactive.

## 4.2 Architecture

A multi-layer perceptron (MLP):

```
Input layer:       n_vars numbers (area vector)
   ↓
Hidden layer 1:    256 neurons, ReLU activation
   ↓ Dropout 0.1
Hidden layer 2:    128 neurons, ReLU
   ↓ Dropout 0.1
Hidden layer 3:    64 neurons, ReLU
   ↓ Dropout 0.1
Output layer:      3 numbers:
                     - log(1 + Weight)
                     - log(1 + max displacement)
                     - max stress
```

About 50 000 parameters total. The `log(1+·)` transform stabilises training on the long-tailed weight and displacement distributions.

## 4.3 Training data — Latin Hypercube Sampling (LHS)

**Why LHS, not uniform random?** Uniform random points cluster; LHS stratifies — it guarantees one point per "slice" of every dimension. Better coverage of the corners of the design space, which is where many optima live.

Using `scipy.stats.qmc.LatinHypercube`, draw 10 000 area vectors uniformly within the bounds. Evaluate each via the FEM engine. Store (input, output) pairs.

## 4.4 Training protocol

- Train / validation / test split: 80 / 10 / 10
- Inputs and outputs standardised (zero mean, unit variance) on training split
- Optimiser: Adam, lr = 1e-3
- Schedule: cosine annealing
- Batch size: 128
- Budget: 200 epochs with early stop on validation MSE

Total training time: < 2 minutes on a laptop CPU.

## 4.5 Validation results — Chapter 4 §4.5

On the 10-bar held-out test set:

| Output head | R² | Use |
|---|---|---|
| Weight | **0.9993** | Direct use in optimiser |
| Displacement | 0.87 | Feasibility *screen* (FEM fallback when borderline) |
| Stress | 0.82 | Feasibility *screen* (FEM fallback when borderline) |

**Hybrid surrogate pattern (Queipo 2005):** Trust the surrogate for the objective (weight), but use it only as a *rough check* for constraints. Borderline candidates re-run on FEM.

## 4.6 Speedup

- FEM: 41 ms / design
- Surrogate (batched): 0.31 ms / design
- **Speedup: 132×**

## 4.7 Optimum quality

Surrogate-in-loop optima match FEM-in-loop optima to **0.09 %** on 10-bar. So you buy the speed without losing optimality.

## 4.8 Why this is contribution C2

You have a quantitative, reproducible, validated speedup of 132× with R² = 0.9993, on a problem of direct civil-engineering interest. That's an unambiguous engineering contribution.

---

# 5. THE PPO REINFORCEMENT-LEARNING AGENT — DEEP DIVE  [Ch 3 §3.6, Ch 4 §4.6]

## 5.1 What is reinforcement learning

Two paradigms in machine learning:
- **Supervised learning:** show the model (input, correct-output) pairs. Surrogate does this.
- **Reinforcement learning:** put an agent in an environment. Agent takes an action, gets a reward, learns over many tries to maximise reward.

RL is how AlphaGo learned chess and Go.

## 5.2 The environment — `src/rl/environment.py`

A Gymnasium environment called `TrussDesignEnv`. **Single-step bandit** formulation:
- Observation: a normalised tensor packing benchmark bounds, loads, and symmetry groups.
- Action: a Box action in $[-1, 1]^{n_{\text{vars}}}$, scaled to design-variable bounds.
- One step per episode: agent outputs a complete area vector, environment evaluates it via surrogate.

## 5.3 The reward function

$$r(\mathbf{x}) = -W(\mathbf{x}) - \lambda \sum_i \max(g_i(\mathbf{x}), 0)^2,\quad \lambda = 10^4$$

- Negative weight (so minimising weight maximises reward).
- Quadratic penalty on constraint violations.
- Large $\lambda$ pushes the agent firmly back into the feasible region.

## 5.4 Training

- Algorithm: PPO (Proximal Policy Optimisation) from Stable-Baselines3
- Default hyperparameters: GAE λ = 0.95, clip ε = 0.2, lr = 3e-4
- Rollout length: 2048
- Total timesteps: 10⁶ (~20 min on laptop CPU)

## 5.5 Result — honest mixed reporting

- Best feasible weight on 10-bar: 5876.9 lb
- **Gap vs literature optimum: +16.1 %**
- Inference time: 0.2 s vs ~30 s for a fresh GA

This misses the 5 % gate from the research objectives. **But you report it honestly as a mixed result.** Reasoning:

> Zhao et al. 2021 already showed that on a single-instance structural problem, a tuned GA outperforms a model-free RL agent. RL's strength is **cross-instance generalisation** — train once, then apply to many problems without retraining. That would be tested by transfer learning across the four benchmarks (future work F4).

So the PPO layer's contribution is **inference speed**, not single-instance optimality. We're honest about that.

---

# 6. THE LLM WARM-START DESIGNER — DEEP DIVE  [Ch 3 §3.7, Ch 4 §4.7]

This is your **strongest novel empirical claim**. Worth understanding deeply.

## 6.1 What does "warm-start" mean

A GA's initial population is normally random ("cold start"). A **warm-start** = seed the population with one or more clever guesses. If the guess is good, the GA converges faster.

## 6.2 The prompt

For each benchmark, construct a structured prompt:

> **System:** "You are a structural engineer sizing an optimum-weight truss."
>
> **User:** "Here is a 10-bar planar cantilever truss with these node coordinates [table]. Loads: [list]. Material properties: E = 1e7 psi, ρ = 0.1 lb/in³. Stress limit: ±25 000 psi. Displacement limit: 2 in. Area bounds: [0.1, 35] in².
>
> Please think step by step about which members will be in tension vs compression, which might be redundant, and what cross-sectional areas you'd suggest. Output your final design as JSON: `{"areas": [...]}`."

Chain-of-thought prompting (Wei 2022) — asking the model to reason before answering — significantly improves quality on compositional tasks.

## 6.3 The client and the cache

`src/llm/client.py` wraps the `anthropic` Python SDK with two additions:
1. **On-disk JSON cache** keyed by a content hash of the prompt. Re-running the experiment doesn't re-bill the API. The cache is committed to the repo (under `results/llm_cache/`).
2. **Heuristic fallback:** if the API call fails or response doesn't parse, return a fully-stressed-design vector $A_i = |F_i| / (0.75\,\sigma_{\text{allow}})$.

So the pipeline runs offline. Any reader can reproduce every LLM result without an API key. This is rare and valuable.

## 6.4 The experiment — paired Wilcoxon design

**Why paired?** Pairing eliminates seed variance. Both arms (LLM-seeded and random-seeded GA) share the *same* random seed for the rest of the GA. So differences in convergence are attributable to the warm-start, not to randomness.

**Procedure:**
- Pick 40 random seeds.
- For each seed:
  - Arm A: GA with Claude-suggested design vector in the initial population.
  - Arm B: GA with pure-random initialisation.
- Record generations to reach within 1 % of the optimum.
- Compare with **paired Wilcoxon signed-rank test**.

## 6.5 The result — 10-bar

| | Mean generations | |
|---|---|---|
| Random initialisation | **203.1** | |
| LLM warm-start | **128.3** | |
| Reduction | **36.8 %** | |
| Paired Wilcoxon p-value | **0.0046** | |

So highly statistically significant.

## 6.6 The mechanism — why did it work

When you read Claude's chain-of-thought, you find it correctly identifies that **members 2, 5, 6, and 10** are redundant in the optimum (they sit close to their lower bound) and suggests small areas for them.

This is an architectural insight that the GA would have rediscovered through trial and error over roughly 70 generations. The LLM provides it for free, in one API call.

## 6.7 The bigger benchmarks — and the efficacy map

| Benchmark | Seeds | $\bar{g}$ random | $\bar{g}$ LLM | Reduction | p-value |
|---|---|---|---|---|---|
| 10-bar | 40 | 197.2 | 124.7 | **36.8 %** | **0.0046** |
| 25-bar | 3 | 82.7 | 19.3 | 76.6 % | 0.25 |
| 72-bar | 5 | 46.2 | 48.4 | -4.8 % | 0.81 |

- 25-bar: huge effect size but only 3 seeds (underpowered statistically).
- 72-bar: null result. No effect.

## 6.8 Why the LLM warm-start fades on bigger problems

On 72-bar, every member of the four-storey tower is carrying load. There's no "redundant member set" to identify. The LLM can't add qualitative architectural insight. The GA finds the constraint boundary fast on its own.

**This is the headline interpretation — the efficacy map.** The LLM helps **where it can supply qualitative architectural reasoning**. When the problem is too dense for that, it has nothing to add.

To my knowledge, this is the **first statistically rigorous demonstration of LLM-warm-start benefit in truss sizing**.

---

# 7. THE IS 800:2007 COMPLIANCE LAYER  [Ch 3 §3.8, Ch 4 §4.9]

Every design that comes out of any optimiser is checked against four clauses of IS 800:2007.

## 7.1 What is IS 800:2007

The Bureau of Indian Standards's general construction code for steel. Replaced the 1984 working-stress edition with limit-state design (aligned with Eurocode and AISC). It's legally binding in Indian practice — every steel structure built in India must comply.

## 7.2 Clause 3.7 / 3.8 — Slenderness

Slenderness $\lambda = L_{\text{eff}} / r$, where $r = \sqrt{I/A}$ is the radius of gyration (for generic rod sections we approximate $r \approx \sqrt{A/\pi}$).

Limits:
- **Compression members:** $\lambda \le 180$
- **Tension members:** $\lambda \le 400$

The sign of the axial force selects which limit applies.

## 7.3 Clause 5.6.1 — Serviceability deflection

Maximum nodal displacement $\le L / 325$ where $L$ is the span. Stricter for trusses supporting brittle cladding.

## 7.4 Clause 6.2 / 6.3 — Tension design strength

Two failure modes:
- **Gross-section yielding** (Clause 6.2): $T_{dg} = A_g f_y / \gamma_{m0}$ with $\gamma_{m0} = 1.1$
- **Net-section rupture** (Clause 6.3): $T_{dn} = 0.9 A_n f_u / \gamma_{m1}$ with $\gamma_{m1} = 1.25$

Design strength = min of these. For pin-jointed rod members (no hole deductions), $A_n = A_g$.

## 7.5 Clause 7.1 — Compression design strength

Design compressive stress $f_{cd} = \chi f_y / \gamma_{m0}$ where $\chi$ is the **column-curve buckling-reduction factor** as a function of non-dimensional slenderness $\bar{\lambda} = \sqrt{f_y / f_{cc}}$ with $f_{cc} = \pi^2 E / \lambda^2$ the Euler buckling stress.

Curve $a$ (the recommended IS 800 curve for rolled I-sections and hollow sections), imperfection factor $\alpha = 0.21$.

## 7.6 Implementation

`src/constraints/is800_checks.py` implements each clause as a pure Python function. `compliance.py` orchestrates them into a `ComplianceReport` dataclass. Internally SI; benchmarks convert at the boundary.

## 7.7 Validation

`tests/test_is800.py` cross-checks each clause against worked examples in Subramanian's *Design of Steel Structures* and Duggal's *Limit State Design of Steel Structures*. All cross-checks agree to three significant figures.

---

# 8. THE STREAMLIT + FASTAPI DELIVERABLE  [Ch 3 §3.9, Ch 4 §4.7]

## 8.1 FastAPI backend — `src/app/api.py`

REST API with five endpoints:

| Endpoint | Purpose |
|---|---|
| `/health` | Is the server alive? |
| `/benchmarks` | List available benchmark problems |
| `/benchmarks/{name}` | Get one benchmark's full spec |
| `/optimize` | Run optimisation. POST {benchmark, algo, seed, n_gen, use_llm, use_surrogate}. Returns best design, weight, convergence history, compliance report. |
| `/llm/suggest` | Get an LLM warm-start design for a benchmark |

LLM warm-start is only injected for single-objective solvers (GA, PSO) — pymoo's NSGA-II doesn't accept the same initial-population argument.

## 8.2 Streamlit UI — `src/app/ui.py`

A thin client over FastAPI. The user:
1. Picks a benchmark from a dropdown.
2. Picks an algorithm (GA / PSO / NSGA-II).
3. Toggles "Use surrogate" and "Use LLM warm-start".
4. Sets seed and generation count.
5. Hits Run.

The UI then renders:
- A **convergence curve** (weight vs generation)
- A **Pareto front** plot (for NSGA-II)
- A **matplotlib visualisation of the truss** with member thicknesses drawn at scale
- An **IS 800 compliance report**

## 8.3 The headline performance claim

End-to-end (UI → API → GA run → compliance report → JSON back → UI render):
**< 10 seconds on 10-bar on a laptop CPU.**

This requires `use_surrogate=true`. Without the surrogate the FEM inner loop caps throughput around 40 seconds per run.

## 8.4 Three pieces of value

This is your viva soundbite for "what's the practical contribution":

1. **Non-Python users** — a junior engineer or M.Tech student can drive the whole stack without writing code.
2. **Sub-10-second response** — moves optimisation from batch tooling to interactive design-review usage.
3. **Offline reproducibility** — all LLM responses cached, anyone can clone tag `phase-10-complete` and reproduce every number without an Anthropic API key.

---

# 9. THE FIVE CONTRIBUTIONS

The whole thesis collapses to these five claims, each tied to a specific numerical result.

## 9.1 C1 — Validated integrated framework

**Claim:** An end-to-end Python framework composing classical optimisers + neural surrogate + RL + LLM + IS 800 + Streamlit UI, into one reproducible system. No prior published work combines all five layers.

**Proof:** PSO on 10-bar matches Sunar 1991's 5060.85 lb to **5061.05 lb — 0.004 % error**. Every layer must be correct for that number to land there.

## 9.2 C2 — 100× neural surrogate

**Claim:** A three-hidden-layer MLP trained on 10 000 LHS samples achieves R² = 0.9993 on weight prediction and delivers a 132× wall-clock speedup, with surrogate-in-loop optima matching FEM-in-loop optima to 0.09 %.

**Proof:** Quantitative test-set numbers in Ch 4 §4.5.

## 9.3 C3 — Rigorously-feasible 72-bar baseline (softened)

**Claim:** Under hard-constraint enforcement, GA, PSO, and NSGA-II converge to ~549 lb on 72-bar with cross-seed spread < 1 %. We could not reproduce the soft-penalty 379.62 lb literature value as feasible in our FEM; pending independent FEM cross-check, framed as constraint-handling-convention observation rather than methodological correction. Future-work F7 lists the OpenSeesPy / SAP2000 / ANSYS cross-check.

**Proof:** Ch 4 §4.3 results table + §4.10.2 hard-vs-soft ablation.

## 9.4 C4 — LLM warm-start efficacy map (strongest novel claim)

**Claim:** On 10-bar, LLM warm-start reduces generations-to-convergence by 36.8 % at paired Wilcoxon p = 0.0046 across 40 seeds — driven by the LLM identifying the redundant-member set {2, 5, 6, 10}. The effect diminishes on larger benchmarks where no comparable architectural insight is available.

**Proof:** Ch 4 §4.7 table + per-seed convergence figure.

## 9.5 C5 — IS 800-compliant Streamlit demo

**Claim:** A web-based design aid that lets a non-Python user run an LLM-assisted IS 800-compliant optimisation in under 10 seconds on a laptop. Repository at tag `phase-10-complete`, fully reproducible offline.

**Proof:** Streamlit UI screenshot + the < 10 s timing observation.

---

# 10. THE 72-BAR HONEST STORY

## 10.1 Originally we claimed

> "The published Camp & Bichon 2004 (380.24 lb) and Bekdaş 2015 (379.62 lb) optima are infeasible by 22–62 % on displacement and 22 % on stress. The true rigorously-feasible optimum is ~549 lb. This is a novel methodological correction to the published literature, previously undocumented."

Strong claim, big number, dramatic story.

## 10.2 Why we softened it before the viva

Three risks:

1. **Encoding bug possibility.** The 72-bar problem has been described slightly differently by Fleury–Schmit 1980 → Erbatur 2000 → Camp 2004 → Bekdaş 2015. Group ordering, load case signs, which DOFs are constrained — small differences could produce huge "violations" we observe.

2. **No independent FEM cross-check.** We haven't run Camp's area vector through OpenSeesPy or SAP2000. Without that, we can't distinguish "literature wrong" from "our encoding subtly different".

3. **The size of the violation (22–62 %) is *too* large to be plausibly soft-penalty slop.** Either the literature is dramatically wrong, or our encoding has an issue. Both are possible; we don't yet know which.

## 10.3 What we now say

> "Under hard-constraint enforcement ($g \le 0$) our GA, PSO, and NSGA-II converge to ~549 lb with cross-seed spread under 1 %. We could not reproduce the published 379.62 lb as feasible in our FEM. This is consistent with the soft-penalty culture in heuristic-optimisation literature, but pending an independent FEM cross-check we frame this as a constraint-handling-convention observation rather than a methodological correction."

**What stayed:**
- The 549 lb hard-constraint result (it's our honest empirical output).
- Cross-seed spread < 1 % (three algorithms agreeing).
- The comparison table.

**What's gone:**
- "novel finding"
- "methodological correction"
- "infeasible by 22–62 %" rhetoric
- "first-order economic fact"
- The dramatic 45 % gap framing

**What's new:**
- Future-work item F7: independent FEM cross-check via OpenSeesPy / SAP2000 / ANSYS.

## 10.4 If asked in the viva

> *"On 72-bar I report an observation, not a finding. My three algorithms converge to 549 lb with cross-seed spread under 1 % under hard-constraint enforcement. I could not reproduce the published 379.62 lb as feasible in my FEM, but pending an independent FEM cross-check I don't claim the literature is wrong — I've listed that cross-check as future-work F7. The honest contribution there is the 549-pound baseline."*

Calm. Clear. Defensible.

---

# 11. THE 15-MINUTE VIVA SCRIPT

(Detailed coaching is in `docs/VIVA_TEACHING_DOC.md` — this section is the speech only.)

## Timing skeleton

| Time | Section | Slide |
|---|---|---|
| 0:00 – 1:00 | Greeting + title | Title |
| 1:00 – 2:30 | Problem hook | Motivation (steel-consumption chart) |
| 2:30 – 5:00 | What I built | Architecture diagram |
| 5:00 – 7:30 | Validation on benchmarks | 10-bar / 25-bar table → surrogate parity plot |
| 7:30 – 10:30 | Headline finding (LLM warm-start) | LLM convergence figure |
| 10:30 – 12:00 | 72-bar honest framing | 72-bar results frame |
| 12:00 – 13:30 | Surrogate + Streamlit | UI screenshot |
| 13:30 – 15:00 | Limitations + future + close | Contributions slide |

## Beat-by-beat (memorise the beats, not the words)

### Beat 1 — Opening (1 min)
- Good-morning sir / ma'am
- Thesis title
- Today I'll cover: framework, validation, LLM finding, Streamlit tool

### Beat 2 — Problem (1.5 min)
- Trusses everywhere in India
- Hand-design 10–30 % surplus
- Sector scale: 1.9 kg CO₂/kg, large numbers
- Optimisers exist but 40 min/run too slow
- **Question:** can AI make this interactive?

### Beat 3 — What I built (2.5 min)
- 8-layer stack
- Bottom: FEM (validated machine-epsilon)
- Middle: GA / PSO / NSGA-II via pymoo
- Three AI layers: surrogate, PPO, LLM
- IS 800 compliance everywhere
- Streamlit + FastAPI on top

### Beat 4 — Validation (2.5 min)
- 10-bar: 0.004 % error vs Sunar 1991 → end-to-end signature
- 25-bar: 0.008 % error vs Schmit 1976
- Surrogate: R² = 0.9993, 132×, 0.09 % in-loop gap

### Beat 5 — Headline LLM result (3 min)
- 40 seeds, paired Wilcoxon
- 203 (random) vs 128 (LLM) generations
- 36.8 % reduction, p = 0.0046
- Mechanism: Claude found redundant members {2,5,6,10}
- Effect map: helps where insight available, null on 72-bar
- First statistically rigorous LLM-warm-start in truss sizing

### Beat 6 — 72-bar honest (1.5 min)
- Hard-constraint optimum at 549 lb, three algorithms agreeing
- Could not reproduce 379.62 lb as feasible
- Observation, not correction
- Future-work F7: independent FEM cross-check

### Beat 7 — Practical (1.5 min)
- Streamlit < 10 s on laptop
- Non-Python users, sub-10-s, offline-reproducible
- Tag `phase-10-complete`, all LLM responses cached

### Beat 8 — Close (1.5 min)
- Limitations: linear, sizing-only, four benchmarks
- PPO mixed result (Zhao 2021)
- 7 future-work directions (F1–F7)
- Strong close: framework + LLM finding + Streamlit tool
- "Thank you, sir. I'm happy to take questions."

---

# 12. THE Q&A BANK

Examiners said to be friendly; expect 2–3 light questions. Here are 12 plausible ones with crisp answers.

### Q1. "What is the novelty of your work?"
> "Three things. One: the first published framework integrating classical optimisers, a neural surrogate, RL, an LLM warm-start, and IS 800 compliance under one roof. Two: a 132× surrogate speedup at R² = 0.9993. Three — the strongest novel empirical claim — the LLM warm-start efficacy map, 36.8 % generation reduction at p = 0.005, fading on larger problems. That last one is, to my knowledge, the first statistically rigorous LLM-warm-start study in truss sizing."

### Q2. "How do I know your framework actually works?"
> "The 10-bar benchmark of Sunar 1991 has a published optimum known to four decimal places. Mine matches to 0.004 % error. The 25-bar matches to 0.008 %. The FEM matches Logan's three-bar analytical example to machine epsilon. Every reported design passes four independent IS 800:2007 clauses. The chain of validation is intact end-to-end."

### Q3. "Why didn't the PPO agent match the GA?"
> "Consistent with Zhao 2021. Model-free RL on a single-instance structural problem doesn't beat a tuned GA on that same instance. RL's advantage is inference latency — 0.2 seconds versus 30 seconds for a fresh GA — and cross-instance generalisation, which we list as future work F4. I report the PPO result honestly as mixed, not as a success."

### Q4. "Why does 72-bar not match the literature?"
> "Under our hard-constraint enforcement, three independent algorithms — GA, PSO, NSGA-II — converge to 549 lb with cross-seed spread under one percent. We could not reproduce the published 379.62 lb as feasible in our FEM. We frame this as a constraint-handling-convention observation, not a literature correction, because we have not independently cross-checked the encoding against a second FEM. Independent cross-check is future-work F7."

### Q5. "Why IS 800 and not Eurocode or AISC?"
> "IS 800:2007 is the legally binding code in India and the population of designs my framework is meant to serve are Indian. The architecture is code-agnostic at the constraint layer, so adding Eurocode 3 or AISC 360 is a localisation task, not a re-architecture."

### Q6. "How is your work different from Geng 2024 or Sun 2023?"
> "Geng 2024 integrated an LLM with OpenSeesPy — two layers. Sun 2023 coupled a DNN surrogate with a GA — two layers. My contribution is integrating five layers — classical optimisers, surrogate, RL, LLM, and an IS 800 compliance module — and validating each against published benchmarks. No prior published work combines all five."

### Q7. "What's the most surprising thing you learned?"
> "That the LLM's value isn't its design quality but its qualitative reasoning. It picked out the redundant members {2, 5, 6, 10} on 10-bar in a single call — an architectural insight the GA would have discovered through 70 generations of trial and error. That's the mechanism behind the 36.8 % generation reduction."

### Q8. "What's your favourite contribution?"
> "The LLM warm-start efficacy map, sir. It's a small but rigorous result — 40 seeds, paired Wilcoxon p-value 0.0046 — and it tells us something genuine about where AI helps in engineering design and where it doesn't. That's what I would publish first."

### Q9. "What was the hardest part?"
> "Getting all eight layers to share a single data contract, sir. Each layer was written in isolation; making them communicate cleanly through three shared dataclasses (`OptimizationResult`, `BenchmarkEvaluation`, `ComplianceReport`) took a lot of refactoring. But once it was working, swapping a benchmark or an algorithm was a single line of code."

### Q10. "What's the deployment story?"
> "Streamlit UI runs in under 10 seconds on a laptop CPU. LLM responses are cached so it works offline. A junior consulting engineer or M.Tech student can drive it without writing Python. The natural next step is validation against real PGCIL transmission-tower drawings — future work F6."

### Q11. "Limitations?"
> "Five, all called out openly. Linear elastic only — no geometric or material non-linearity. Sizing only — not topology. Static only — no seismic. PPO is single-instance — no transfer. And on 72-bar I report an observation, not a finding, pending independent FEM cross-check."

### Q12. "Where would you take this next?"
> "Seven directions listed in Chapter 5. The two highest-leverage: a GNN surrogate to handle variable topology, and PPO transfer learning across the four-benchmark family to actually test Zhao 2021's claim. The most practically valuable: validation against real Indian transmission-tower designs through a data-sharing arrangement with PGCIL."

### If you genuinely don't know
> "That's a good question, sir. I haven't tested that specifically, but my expectation would be X because Y. I'd want to run the experiment to be sure."

This is **respected**. Bluffing is not.

---

# 13. THE 20 MUST-KNOW NUMBERS

Memorise these like a phone number. They are your defence.

| Number | What it is |
|---|---|
| **5060.85 lb** | Sunar 1991 published 10-bar optimum |
| **5061.05 lb** | My PSO result |
| **0.004 %** | My 10-bar error |
| **545.22 lb** | Schmit 25-bar literature value |
| **545.26 lb** | My 25-bar result |
| **0.008 %** | My 25-bar error |
| **549 lb** | My 72-bar hard-constraint optimum |
| **379.62 lb** | The 72-bar soft-penalty literature value |
| **316.61 kg** | My 200-bar GA result (SI Indian units) |
| **R² = 0.9993** | Surrogate weight prediction accuracy |
| **132×** | Surrogate speedup over FEM |
| **0.09 %** | Surrogate-in-loop vs FEM-in-loop optimum gap |
| **36.8 %** | LLM warm-start generation reduction on 10-bar |
| **p = 0.0046** | Paired Wilcoxon significance on the 36.8 % |
| **40 seeds** | Sample size for the LLM result |
| **128 vs 203** | Mean generations: LLM vs random on 10-bar |
| **+16 %** | PPO gap vs optimum (the honest mixed result) |
| **{2, 5, 6, 10}** | Redundant members Claude identified on 10-bar |
| **< 10 s** | Streamlit demo end-to-end time |
| **10 000** | LHS samples for surrogate training |

---

# 14. GLOSSARY

| Term | Plain-English meaning |
|---|---|
| **Adam** | A popular gradient-descent variant used to train neural networks. |
| **Bandit (contextual)** | An RL problem where each episode is one step — the agent acts once and gets a reward. |
| **Beamer** | A LaTeX package for making PDF slide decks. |
| **Benchmark** | A standard problem with a known solution, used to validate new methods. |
| **Chain-of-thought** | A prompting technique where the LLM is asked to reason step-by-step before answering. Improves quality on compositional tasks (Wei 2022). |
| **Compliance check** | Verifying that a design satisfies a code (here, IS 800:2007). |
| **Crossover** | In GA, combining two parent designs to produce a child. |
| **Crowding distance** | In NSGA-II, a measure of how isolated a design is on the Pareto front — used to maintain diversity. |
| **Degree of freedom (DOF)** | One axis a node can move along. 2 in 2D, 3 in 3D. |
| **Design vector / variable** | The set of numbers being optimised (here: bar areas). |
| **Dropout** | A neural-network regularisation technique that randomly zeroes neurons during training to prevent overfitting. |
| **Efficacy map** | Your novel finding: a quantitative description of where the LLM warm-start helps and where it doesn't. |
| **Elitist selection** | Keeping the best individuals across generations. |
| **Epoch** | One full pass through the training data. |
| **FastAPI** | A Python library for making REST APIs. |
| **Feasible / Infeasible** | Satisfies all constraints / breaks at least one. |
| **FEM** | Finite Element Method — the physics solver. |
| **Fully-stressed design** | Heuristic where each member's area is set so its stress equals the allowable. Used as offline fallback when the LLM API is unreachable. |
| **GA** | Genetic Algorithm. |
| **GAE** | Generalised Advantage Estimation, used in PPO. |
| **Gym / Gymnasium** | OpenAI's (now community-maintained) standard interface for RL environments. |
| **Hyperparameter** | A setting of an algorithm that you choose, rather than learn (e.g., population size, learning rate). |
| **IS 800:2007** | The Indian steel design code. |
| **JSON** | A data-interchange format used to pass results between FastAPI and Streamlit, and to cache LLM responses. |
| **Latin Hypercube Sampling (LHS)** | A space-filling sampling method better than uniform random for surrogate training. |
| **LLM** | Large Language Model (here: Anthropic Claude). |
| **MLP** | Multi-Layer Perceptron — a standard feed-forward neural network. |
| **Mutation** | In GA, randomly tweaking a child's design. |
| **NSGA-II** | A two-objective evolutionary algorithm (Deb 2002). |
| **Optimum** | The best design (lightest feasible). |
| **Pareto front** | The set of designs where no other design is better on *every* objective. |
| **PPO** | Proximal Policy Optimisation — a popular RL algorithm (Schulman 2017). |
| **PSO** | Particle Swarm Optimisation. |
| **pymoo** | A Python library for multi-objective optimisation. |
| **R²** | Coefficient of determination — how well a model's predictions match the true values. 1.0 = perfect. |
| **ReLU** | Rectified Linear Unit — a neural-network activation function: f(x) = max(0, x). |
| **RL** | Reinforcement Learning. |
| **SBX** | Simulated Binary Crossover — pymoo's default crossover for continuous variables. |
| **Slenderness** | Length over radius of gyration. Long thin bars are slender; they buckle easily under compression. |
| **Stable-Baselines3** | A Python library implementing RL algorithms including PPO. |
| **Streamlit** | A Python library for making simple web UIs without HTML. |
| **Surrogate** | A cheap fake of an expensive simulator. |
| **Truss** | A structure of straight bars connected at joints, carrying load by axial force only. |
| **Wilcoxon signed-rank** | A non-parametric paired statistical test — what we used for the LLM result. |
| **ZAhao 2021** | Reference for the framing that single-instance RL doesn't beat tuned GA. |

---

# APPENDIX A — REPO TOUR

What's in the repository (under `phase-10-complete` tag):

```
thesis/
├── README.md                          (Quick-start commands)
├── requirements.txt
├── pyproject.toml
├── .env.example                       (Anthropic API key template)
│
├── src/
│   ├── fem/                           Truss FEM solver (Logan-style)
│   ├── benchmarks/                    Four benchmark problem classes
│   ├── algorithms/                    GA / PSO / NSGA-II via pymoo
│   ├── ml/                            MLP surrogate
│   ├── rl/                            PPO agent + Gym environment
│   ├── llm/                           Claude warm-start + cache
│   ├── constraints/                   IS 800:2007 checks
│   └── app/                           FastAPI + Streamlit
│
├── tests/                             81 pytest tests
│
├── scripts/                           CLI runners
│   ├── run_single.py
│   ├── run_batch.py
│   ├── train_surrogate.py
│   ├── train_rl.py
│   └── run_llm_warmstart.py
│
├── results/
│   └── llm_cache/                     Cached Claude responses
│
├── docs/                              (this file lives here)
│   ├── thesis_vision.md
│   ├── ground_truth.md
│   ├── thesis_explained_simply.md
│   ├── HANDOFF.md
│   └── VIVA_TEACHING_DOC.md           ← you are here
│
└── thesis_writeup/                    LaTeX source
    ├── main.tex
    ├── main.pdf                       (3.9 MB compiled writeup)
    ├── preamble.tex
    ├── references.bib
    ├── frontmatter/
    │   ├── titlepage.tex
    │   ├── abstract.tex               (softened)
    │   ├── certificate.tex
    │   ├── declaration.tex
    │   ├── acknowledgements.tex
    │   └── toc.tex
    ├── chapters/
    │   ├── 01_introduction.tex
    │   ├── 02_literature.tex
    │   ├── 03_methodology.tex         (softened)
    │   ├── 04_results.tex             (softened)
    │   ├── 05_conclusion.tex          (softened, F7 added)
    │   ├── A_fem_derivation.tex
    │   ├── B_is800_provisions.tex
    │   ├── C_geometry_specs.tex
    │   └── D_reproducibility.tex
    └── slides/
        ├── main.tex                   (softened, blue pill on C3)
        └── main.pdf                   (2.4 MB compiled deck)
```

---

# APPENDIX B — ONE FINAL CHECKLIST BEFORE THE VIVA

- [ ] Both PDFs (`main.pdf` writeup and `slides/main.pdf`) on your laptop
- [ ] Same PDFs on a USB stick as backup
- [ ] Same PDFs emailed to yourself
- [ ] Charger and laptop fully charged
- [ ] HDMI cable / adapter for the projector
- [ ] Clicker or arrow keys tested
- [ ] Cheat sheet printed and folded
- [ ] Water bottle
- [ ] Formal clothes laid out
- [ ] Practiced the 15-minute story aloud at least three times
- [ ] Memorised the 20 must-know numbers
- [ ] Slept 8 hours the night before

---

**You built this. You measured it. You wrote it up. You softened the one risky claim before walking in.**

**You're ready.**

— end of doc —
