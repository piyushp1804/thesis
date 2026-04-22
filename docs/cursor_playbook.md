# The Exhaustive Cursor Playbook

Every prompt. Every file. Every day.
Zero to thesis in 10 days.

**For:** Aryan — 5th year IDD Civil — IIT BHU
**Thesis:** Multi-Objective Truss Optimization
**~80 prompts · 10 days · 8,000+ lines · 142 pages**

---

## How to use this doc

This is a recipe book. Not a theory book.

Each day has a sequence of prompts. Open Cursor, paste a prompt, review the output, save the file. Move to the next prompt.

Every prompt is a complete paragraph you can copy and paste into Cursor's chat. Do not modify unless you understand what you're changing.

---

## Ground rules for using Cursor

- Work in a single folder — treat it as a git repo from Day 0.
- Commit to git after every working prompt. If something breaks later, `git checkout` saves you.
- When Cursor writes code, actually READ it once before running. Not to change it, just to know what's there.
- When an error appears, copy-paste the entire error back to Cursor. Don't try to fix yourself first.
- If a prompt output feels wrong, ask Cursor: 'explain what this code does, step by step.' Then decide.
- Save conversations in Cursor — long context helps later prompts reference earlier code.

---

## What you need before Day 1

- [ ] A laptop with at least 8GB RAM (any modern machine is fine)
- [ ] Python 3.10 or newer installed
- [ ] Cursor IDE installed (cursor.sh)
- [ ] Git installed
- [ ] GitHub account for backing up the repo
- [ ] 2-3 hours of uninterrupted time per day
- [ ] Coffee/chai. Important.

---

## Day 0 — Setup (2 hours before Day 1)

Get your environment ready. This is the only manual work in the whole project.

### 0.1 Install Python (if not already)

Open terminal and check:

```bash
python3 --version
```

If you see 3.10 or higher, you're good. If not, download from python.org.

### 0.2 Install Cursor

Go to cursor.sh, download, install. Sign in with GitHub.

### 0.3 Create your project folder

```bash
mkdir truss_optimization
cd truss_optimization
git init
cursor .
```

This opens Cursor in your new folder. From here, everything happens inside Cursor.

### 0.4 Create Python virtual environment

In Cursor's terminal (Ctrl+\` or Cmd+\`):

```bash
python3 -m venv venv
source venv/bin/activate    # Mac/Linux
# venv\Scripts\activate     # Windows
```

You'll see `(venv)` appear in your prompt. That means it worked.

### 0.5 First Cursor prompt — create the scaffold

(Use Day 0 scaffold prompt from Cursor chat)

### 0.6 Install packages

```bash
pip install -r requirements.txt
```

Takes 2-3 minutes. Get chai while it runs.

### 0.7 First git commit

```bash
git add .
git commit -m 'Day 0: initial scaffold'
```

---

## Day 1 — Build the FEM Solver

**Goal:** a working truss solver that computes rod forces and joint displacements.
By end of day, you'll validate against a hand-calculated 3-bar truss and match to 3 decimals.

### Morning prompts (2 hours)

**Prompt 1.1 — Truss element class**

After Cursor writes this:
- [ ] Read the code. You should see: `__init__`, `length`, `direction_cosines`, `local_stiffness`, `transformation_matrix`, `global_stiffness`, `dof_indices`.
- [ ] Run: `python src/fem/truss_element.py` — should print a 4x4 matrix with values like 2e9, -2e9.
- [ ] If it errors, copy the error to Cursor and ask for a fix.

**Prompt 1.2 — Global assembly**

**Prompt 1.3 — Linear solver**

### Afternoon prompts (2 hours) — Validation

**Prompt 1.4 — The canonical 3-bar validation test**

Running the test:

```bash
pytest tests/test_fem_3bar.py -v
```

- [ ] See GREEN 'PASSED'? Your FEM works. Celebrate.
- [ ] See RED 'FAILED'? Copy error to Cursor. Iterate until green.

### End-of-day checklist

- [ ] truss_element.py written and tested
- [ ] assembly.py written
- [ ] solver.py written
- [ ] test_fem_3bar.py PASSES
- [ ] `git commit -m 'Day 1: FEM solver working, 3-bar validated'`

---

## Day 2 — Encode the 4 Benchmarks

**Goal:** All 4 famous trusses modeled and solved for their original loading. Deflections match literature within 2%.

### Morning prompts

**Prompt 2.1 — 10-bar planar truss**

**Prompt 2.2 — 25-bar spatial truss**

**Prompt 2.3 — 72-bar and 200-bar**

### Afternoon prompts — Verify all 4

**Prompt 2.4 — Unified benchmark tester**

### End-of-day checklist

- [ ] 10-bar truss builds and solves
- [ ] 25-bar truss builds and solves
- [ ] 72-bar truss builds and solves
- [ ] 200-bar truss builds and solves
- [ ] All 4 tests PASS with sensible (non-NaN) values
- [ ] `git commit -m 'Day 2: all 4 benchmarks encoded and tested'`

---

## Day 3 — Genetic Algorithm (GA)

**Goal:** GA finds optimum weight for 10-bar truss within 5% of published 5,060 lb.

### Morning prompts

**Prompt 3.1 — The pymoo Problem class**

**Prompt 3.2 — GA wrapper**

### Afternoon prompts

**Prompt 3.3 — First real optimization run**

Run it:

```bash
python scripts/run_ga_10bar.py
```

- [ ] Takes 2-5 minutes
- [ ] Final weight should be 5000-5500 lb range
- [ ] If it's way off (< 4000 or > 7000), something is wrong — paste output to Cursor

**Prompt 3.4 — Multiple seeds for statistical confidence**

### End-of-day checklist

- [ ] TrussOptimizationProblem class working
- [ ] GA wrapper running on 10-bar
- [ ] Best weight within 5,000–5,500 lb range
- [ ] 10-seed statistical run completed
- [ ] Mean result within 3% of Sunar & Belegundu's 5,060 lb
- [ ] `git commit -m 'Day 3: GA working, 10-bar converges to literature'`

---

## Day 4 — Particle Swarm Optimization (PSO)

**Goal:** PSO working on all 4 benchmarks. Results comparable to GA.

### Morning prompts

**Prompt 4.1 — PSO wrapper**

**Prompt 4.2 — Run PSO on all 4 benchmarks**

### Afternoon prompts

**Prompt 4.3 — Parallel runs (optional but fast)**

**Prompt 4.4 — GA on remaining 3 benchmarks**

### End-of-day checklist

- [ ] PSO wrapper running
- [ ] GA completed on all 4 benchmarks (30 seed runs total)
- [ ] PSO completed on all 4 benchmarks (30 seed runs total)
- [ ] Results saved in results/ folder (60 pickle files total)
- [ ] Summary table showing comparable GA vs PSO performance
- [ ] `git commit -m 'Day 4: GA and PSO on all 4 benchmarks'`

---

## Day 5 — NSGA-II (Multi-Objective)

**Goal:** Pareto fronts for all 4 benchmarks showing weight vs displacement trade-off.

### Morning prompts

**Prompt 5.1 — NSGA-II wrapper**

**Prompt 5.2 — Modify Problem for multi-objective mode**

### Afternoon prompts

**Prompt 5.3 — Run NSGA-II on all benchmarks**

**Prompt 5.4 — Quick Pareto front plot**

### End-of-day checklist

- [ ] NSGA-II wrapper running
- [ ] Pareto fronts generated for all 4 benchmarks
- [ ] Hypervolume metrics computed
- [ ] 4 Pareto front plots saved as PNG and SVG
- [ ] `git commit -m 'Day 5: NSGA-II working, Pareto fronts generated'`

---

## Day 6 — IS 800 Constraints + Full Batch

**Goal:** Add Indian code compliance and run every remaining configuration.

### Morning prompts

**Prompt 6.1 — IS 800 constraint module**

**Prompt 6.2 — Integrate IS 800 into Problem**

### Afternoon prompts

**Prompt 6.3 — Parametric sensitivity runs**

**Prompt 6.4 — Full batch orchestrator**

### End-of-day checklist

- [ ] IS 800 checks implemented and tested
- [ ] Problem class uses IS 800 by default
- [ ] Parametric sensitivity study complete
- [ ] Full batch script produces all ~100 result files
- [ ] `git commit -m 'Day 6: IS 800 integration + full batch complete'`

---

## Day 7 — Generate All Figures

**Goal:** 65-75 publication-quality figures for the thesis.

### Morning prompts

**Prompt 7.1 — Matplotlib style setup**

**Prompt 7.2 — Truss geometry plotter**

**Prompt 7.3 — Convergence plots**

### Afternoon prompts

**Prompt 7.4 — Pareto plots (enhanced)**

**Prompt 7.5 — Algorithm comparison dashboard**

**Prompt 7.6 — Parametric sensitivity plots**

### End-of-day checklist

- [ ] Style config applied everywhere
- [ ] Truss geometry plots (12 figures)
- [ ] Convergence plots (12 figures)
- [ ] Pareto plots (6 figures)
- [ ] Comparison dashboard (2 figures)
- [ ] Parametric sensitivity plots (4 figures)
- [ ] Total: ~40 figures in figures/ folder
- [ ] All figures also saved as SVG
- [ ] `git commit -m 'Day 7: all figures generated'`

---

## Days 8-10 — Write the Thesis

**Goal:** 142 pages of thesis written with Cursor/Claude assistance.

### Writing strategy — Important

Do NOT ask Claude to 'write my thesis.' The output will be generic and plagiarism flags will trigger.

Instead, do this loop for every section:
1. Write a 3-sentence outline of what the section says — in YOUR words.
2. Hand Claude the outline + any relevant code/data/figures.
3. Ask Claude to EXPAND the outline into formal academic prose.
4. Review the output. Rewrite 30-50% of it in your own voice to make it yours.
5. Check for plagiarism against Turnitin before committing.

### Day 8 — Chapters 1 & 2 (53 pages)

**8.1 — Write Chapter 1 outline yourself**

Before asking Claude anything, write a pen-and-paper outline:
- 1.1 Background: why do trusses matter? (3 bullet points)
- 1.2 Motivation: why optimize? (3 bullet points)
- 1.3 Problem statement: 2 sentences
- 1.4 Objectives: 5 objectives from your planning doc
- 1.5 Scope: 5 bullets on what's in and out
- 1.6 Thesis organization: 5 sentences (one per chapter)

**8.2 — Prompt for Chapter 1 drafting**

**8.3 — Chapter 2 Literature Review strategy**

Literature review is the biggest chapter (38 pages) but also the most formulaic. Break it into 10 sections and draft each separately.

### Day 9 — Chapters 3 & 4 (83 pages)

**9.1 — Chapter 3 (Methodology) is mostly equations and code descriptions**

This chapter writes itself once you have the code.

**9.2 — Chapter 4 (Results) writes ITSELF from figures**

For each figure in figures/ folder, describe what you see. You'll do this ~70 times (one per figure). Each only takes 2 minutes.

### Day 10 — Chapter 5, Abstract, Formatting, Submit

**10.1 — Chapter 5 Conclusion**

**10.2 — Abstract**

**10.3 — Format, compile, submit**

- [ ] Compile all chapters into one .docx file
- [ ] Insert all figures at their referenced positions
- [ ] Number all equations and tables
- [ ] Generate TOC, list of figures, list of tables
- [ ] Check page numbers, running headers
- [ ] Insert declaration, acknowledgment, certificate pages (template from IIT BHU)
- [ ] Run through Turnitin. Target: < 10% similarity
- [ ] Iterate on high-similarity sections by rewriting
- [ ] Final PDF export
- [ ] Submit to Pathak for review
- [ ] `git commit -m 'Day 10: thesis submitted'`

---

## Appendix A — Cursor Cheat Sheets

### A.1 Prompt patterns that always work

- **Pattern 1: 'Write a module'** — describe inputs, outputs, and constraints
- **Pattern 2: 'Fix this error'** — paste the full traceback
- **Pattern 3: 'Explain this code'** — ask for step-by-step breakdown
- **Pattern 4: 'Refactor for X'** — specify the goal (readability, speed, etc.)

### A.2 Common errors and quick fixes

- `ModuleNotFoundError` — forgot to activate venv or install a package
- `numpy.linalg.LinAlgError: Singular matrix` — stiffness matrix is wrong (check boundary conditions)
- `ValueError: shapes not aligned` — array dimension mismatch in assembly

### A.3 When to stop asking Cursor and think yourself

- If you've been in the same error loop for 30 minutes — step away, check units/units/units.
- If Cursor keeps suggesting slightly different wrong fixes — `git checkout` to before the error started, and try a different approach.
- If the output feels too good to be true (e.g., weight = 0.001 lb) — it IS too good. You have a bug.
- If a concept doesn't make sense after 2 explanations — ask Cursor to use an analogy: 'explain X using a non-engineering analogy, then connect back to engineering.'

---

## Appendix B — The Final Checklist

Print this page. Tape it on your wall. Check items off as you go.

### Day 0 — Setup
- [ ] Python 3.10+ installed
- [ ] Cursor IDE installed
- [ ] Git initialized in truss_optimization/
- [ ] venv created and activated
- [ ] requirements.txt installed
- [ ] Folder scaffold generated

### Day 1 — FEM
- [ ] truss_element.py working
- [ ] assembly.py working
- [ ] solver.py working
- [ ] test_fem_3bar.py PASSES

### Day 2 — Benchmarks
- [ ] 10-bar encoded and tested
- [ ] 25-bar encoded and tested
- [ ] 72-bar encoded and tested
- [ ] 200-bar encoded and tested

### Day 3 — GA
- [ ] TrussOptimizationProblem working
- [ ] run_ga() working
- [ ] 10-bar weight in 5000-5200 lb range
- [ ] 10-seed statistical run complete

### Day 4 — PSO
- [ ] run_pso() working
- [ ] PSO + GA complete on all 4 benchmarks
- [ ] ~60 pickle files in results/

### Day 5 — NSGA-II
- [ ] run_nsga2() working
- [ ] Pareto fronts for all 4 benchmarks
- [ ] Hypervolume metrics computed

### Day 6 — IS 800 + Batch
- [ ] IS 800 checks implemented
- [ ] Parametric sensitivity complete
- [ ] Full batch runs finished (~100 result files)

### Day 7 — Figures
- [ ] ~40-70 figures in figures/ folder
- [ ] All figures also saved as SVG

### Day 8 — Writing (Ch 1 + 2)
- [ ] Chapter 1 draft complete (15 pages)
- [ ] Chapter 2 draft complete (38 pages)

### Day 9 — Writing (Ch 3 + 4)
- [ ] Chapter 3 draft complete (35 pages)
- [ ] Chapter 4 draft complete (48 pages)

### Day 10 — Final
- [ ] Chapter 5 complete (6 pages)
- [ ] Abstract written
- [ ] TOC, lists, front-matter done
- [ ] Turnitin < 10%
- [ ] PDF exported
- [ ] Submitted to Pathak

---

*— End of Cursor Playbook —*
*Now go build it.*
