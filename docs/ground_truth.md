# Ground Truth Reference Sheet

This file is the **numerical source of truth** for Phase 1 validation.
Every benchmark geometry, load case, reference optimum, and pass-tolerance
the tests enforce must match what is written here. If the literature
source is updated, update this file first, then re-run the validation
suite.

Unit discipline (critical):
- The **10-bar, 25-bar, and 72-bar** problems are published in
  **imperial units**: in, lb, psi.
- The **200-bar** problem is published in **SI units** and uses
  **real structural steel properties**, not aluminium.
- `src/benchmarks/base.py` exposes each benchmark's unit system and
  material density so weight is computed consistently with literature.

---

## 10-bar planar truss (Sunar & Belegundu 1991)

| Parameter                  | Value                                |
|----------------------------|--------------------------------------|
| Nodes                      | 6                                    |
| Elements (bars)            | 10                                   |
| Material                   | Aluminium                            |
| Young's modulus E          | 1.0 x 10^7 psi                       |
| Density rho                | 0.1 lb/in^3                          |
| Applied loads              | 100000 lbf (vertical) at nodes 2 & 4 |
| Stress limit               | +/- 25000 psi                        |
| Displacement limit         | 2.0 in at every free DOF             |
| Area bounds                | [0.1, 35.0] in^2                     |
| Reference optimum weight   | 5060.85 lb                           |
| Pass tolerance (GA)        | within 2.0% of 5060.85 lb over 10 seeds |
| Units                      | imperial (in / lb / psi)             |

Published optimum areas (in^2), element 1..10:
`30.52, 0.100, 23.20, 15.22, 0.100, 0.551, 7.457, 21.04, 21.53, 0.100`

---

## 25-bar spatial truss (Schmit & Miura 1976)

| Parameter                  | Value                                |
|----------------------------|--------------------------------------|
| Nodes                      | 10                                   |
| Elements (bars)            | 25                                   |
| Material                   | Aluminium                            |
| Young's modulus E          | 1.0 x 10^7 psi                       |
| Density rho                | 0.1 lb/in^3                          |
| Load cases                 | 2 (combined multi-point transverse)  |
| Stress limit               | member-type-dependent (see module)   |
| Displacement limit         | 0.35 in at top nodes                 |
| Area bounds                | [0.01, 3.4] in^2                     |
| Symmetry groups            | 8 (design vars = 8)                  |
| Reference optimum weight   | 545.22 lb                            |
| Pass tolerance (GA)        | within 2.5% over 10 seeds            |
| Units                      | imperial                             |

---

## 72-bar spatial tower (Erbatur et al. 2000)

| Parameter                  | Value                                |
|----------------------------|--------------------------------------|
| Nodes                      | 20                                   |
| Elements (bars)            | 72                                   |
| Material                   | Aluminium                            |
| Young's modulus E          | 1.0 x 10^7 psi                       |
| Density rho                | 0.1 lb/in^3                          |
| Load cases                 | 2                                    |
| Stress limit               | +/- 25000 psi                        |
| Displacement limit         | 0.25 in at top node                  |
| Area bounds                | [0.1, 4.0] in^2 (discrete or continuous) |
| Symmetry groups            | 16 (design vars = 16)                |
| Reference optimum weight   | 379.62 lb                            |
| Pass tolerance (GA)        | within 2.5% over 10 seeds            |
| Units                      | imperial                             |

---

## 200-bar planar truss (Kaveh & Talatahari 2010) — "the steel one"

**TRAP WARNING**: many implementations incorrectly reuse the aluminium
constants from the smaller benchmarks. This problem is real STEEL.

| Parameter                  | Value                                |
|----------------------------|--------------------------------------|
| Nodes                      | 77                                   |
| Elements (bars)            | 200                                  |
| Material                   | Structural steel                     |
| Young's modulus E          | 210 GPa  (= 2.10 x 10^11 Pa)         |
| Density rho                | 7850 kg/m^3                          |
| Load cases                 | 3                                    |
| Stress limit               | +/- 250 MPa                          |
| Displacement limit         | per load case (see module)           |
| Area bounds                | [6.5e-5, 2.5e-2] m^2                 |
| Symmetry groups            | 29 (design vars = 29)                |
| Reference optimum weight   | 25445 kg                             |
| Pass tolerance (GA)        | within 3.0% over 10 seeds            |
| Units                      | SI (m / N / Pa / kg)                 |

---

## IS 800:2007 clauses referenced

| Clause  | Topic                          | Enforced in                              |
|---------|--------------------------------|------------------------------------------|
| 3.8     | Slenderness limits             | `is800_checks.check_slenderness`         |
| 5.6.1   | Deflection limits              | `compliance.full_is800_check`            |
| 6.2     | Tension strength (yielding)    | `is800_checks.check_tension_yield`       |
| 6.3     | Tension strength (rupture)     | `is800_checks.check_tension_rupture`     |
| 7.1     | Compression strength           | `is800_checks.check_compression`         |
| 7.5     | Angle / compound members notes | `is800_checks.check_slenderness`         |

Default steel: Fe 410, fy = 250 MPa, fu = 410 MPa, gamma_m0 = 1.10,
gamma_m1 = 1.25. All spelled out in `src/constraints/is800_checks.py`.

---

## Phase 1 scoreboard targets

| Benchmark      | Literature optimum | Tolerance | Feasibility (GA) |
|----------------|--------------------|-----------|------------------|
| 10-bar planar  | 5060.85 lb         | +/- 2.0 % | 100 %            |
| 25-bar spatial | 545.22 lb          | +/- 2.5 % | 100 %            |
| 72-bar tower   | 379.62 lb          | +/- 2.5 % | 100 %            |
| 200-bar planar | 25445 kg           | +/- 3.0 % | 100 %            |

If any row misses its tolerance over 10 seeds, Phase 1 gate **fails** and
no downstream phase starts until GA/NSGA-II/problem encoding or benchmark
data is corrected.

---

## Phase 2 surrogate targets (for reference, enforced later)

- `R^2` on weight prediction > 0.98 on every benchmark (held-out test set).
- Inner-loop speedup vs pure-FEM GA: > 50x on 10-bar, > 100x on 200-bar.

## Phase 4 LLM targets (for reference)

- Generations-to-90%-convergence reduction: >= 15 %, p < 0.05 paired test
  across seeds, on at least 2 benchmarks.
