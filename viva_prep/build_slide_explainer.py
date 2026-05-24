"""
Build the definitive slide-by-slide viva explainer.
One section per slide: what's on it, what to SAY (verbatim), why it's there,
the likely question, and the answer. Output:
  /home/user/thesis/viva_prep/Slide_By_Slide_Explainer.docx
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

INK = RGBColor(0x0B, 0x12, 0x20)
MUTED = RGBColor(0x5B, 0x6B, 0x85)
ACCENT = RGBColor(0x0A, 0x66, 0xFF)
GREEN = RGBColor(0x10, 0x8A, 0x5B)
WARN = RGBColor(0xC0, 0x52, 0x10)


def shade(cell, hexfill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexfill)
    tcPr.append(shd)


def hr(p):
    pPr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "6")
    b.set(qn("w:space"), "1"); b.set(qn("w:color"), "D6DCE6")
    bdr.append(b); pPr.append(bdr)


def H1(doc, t):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(22); p.paragraph_format.space_after = Pt(6)
    r = p.add_run(t); r.font.name = "Calibri"; r.font.size = Pt(24); r.bold = True; r.font.color.rgb = INK
    hr(p); return p


def slidehdr(doc, num, total, title):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(20); p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"SLIDE {num} / {total}")
    r.font.name = "Calibri"; r.font.size = Pt(10); r.bold = True; r.font.color.rgb = ACCENT
    p2 = doc.add_paragraph(); p2.paragraph_format.space_after = Pt(4)
    r2 = p2.add_run(title); r2.font.name = "Calibri"; r2.font.size = Pt(16); r2.bold = True; r2.font.color.rgb = INK
    hr(p2)


def label(doc, text, color=ACCENT):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(1)
    r = p.add_run(text); r.font.name = "Calibri"; r.font.size = Pt(10.5); r.bold = True; r.font.color.rgb = color
    return p


def body(doc, text, italic=False, color=INK, size=11, after=5):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(after)
    r = p.add_run(text); r.font.name = "Calibri"; r.font.size = Pt(size); r.italic = italic; r.font.color.rgb = color
    return p


def say(doc, text):
    """A 'what to say' verbatim block — indented, accent left border."""
    table = doc.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    shade(cell, "F3F7FF")
    cell.paragraphs[0].text = ""
    r = cell.paragraphs[0].add_run("“" + text + "”")
    r.font.name = "Calibri"; r.font.size = Pt(11); r.italic = True; r.font.color.rgb = INK
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders"); left = OxmlElement("w:left")
    left.set(qn("w:val"), "single"); left.set(qn("w:sz"), "24"); left.set(qn("w:color"), "0A66FF")
    b.append(left); tcPr.append(b)
    doc.add_paragraph()


def bullet(doc, text, prefix=None):
    p = doc.add_paragraph(style="List Bullet"); p.paragraph_format.space_after = Pt(2)
    if prefix:
        r = p.add_run(prefix + "  "); r.font.name = "Calibri"; r.font.size = Pt(11); r.bold = True; r.font.color.rgb = INK
    r2 = p.add_run(text); r2.font.name = "Calibri"; r2.font.size = Pt(11); r2.font.color.rgb = INK


def qa(doc, q, a):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(1)
    r = p.add_run("Likely Q:  "); r.font.name = "Calibri"; r.font.size = Pt(10.5); r.bold = True; r.font.color.rgb = WARN
    r2 = p.add_run(q); r2.font.name = "Calibri"; r2.font.size = Pt(10.5); r2.italic = True; r2.font.color.rgb = INK
    p2 = doc.add_paragraph(); p2.paragraph_format.left_indent = Inches(0.25); p2.paragraph_format.space_after = Pt(6)
    r3 = p2.add_run("Answer:  "); r3.font.name = "Calibri"; r3.font.size = Pt(10.5); r3.bold = True; r3.font.color.rgb = GREEN
    r4 = p2.add_run(a); r4.font.name = "Calibri"; r4.font.size = Pt(10.5); r4.font.color.rgb = INK


def timing(doc, t):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(2)
    r = p.add_run("⏱  Time on this slide: " + t)
    r.font.name = "Calibri"; r.font.size = Pt(10); r.bold = True; r.font.color.rgb = MUTED


def pb(doc):
    doc.add_page_break()


# ============================================================
doc = Document()
for s in doc.sections:
    s.top_margin = Cm(1.8); s.bottom_margin = Cm(1.8); s.left_margin = Cm(2.0); s.right_margin = Cm(2.0)
doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(11)

TOTAL = 34

# ---------- COVER ----------
p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(50)
r = p.add_run("VIVA DECK — SLIDE-BY-SLIDE EXPLAINER")
r.font.name = "Calibri"; r.font.size = Pt(11); r.bold = True; r.font.color.rgb = MUTED
p = doc.add_paragraph()
r = p.add_run("Every Slide, Fully Explained")
r.font.name = "Calibri"; r.font.size = Pt(30); r.bold = True; r.font.color.rgb = INK
p = doc.add_paragraph()
r = p.add_run("AI-Powered Framework for Multi-Objective Steel Truss Optimization\n"
              "Read only this. For each of the 34 slides: what's on screen, exactly "
              "what to say, why it's there, and the question that will follow.")
r.font.name = "Calibri"; r.font.size = Pt(13); r.italic = True; r.font.color.rgb = MUTED
p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(40); hr(p)
p = doc.add_paragraph()
r = p.add_run("Aryan Gupta  •  Roll 21064030  •  IDD Civil Engineering (Y5)\n"
              "Supervisor: Dr. Krishna Kant Pathak  •  IIT (BHU) Varanasi")
r.font.name = "Calibri"; r.font.size = Pt(12); r.font.color.rgb = INK

label(doc, "HOW TO READ THIS")
body(doc, "Each slide has five parts: (1) what the examiner sees, (2) the blue box "
          "is your verbatim script — say it almost word-for-word, (3) why the slide "
          "exists in the story, (4) the most likely question, (5) the answer. The "
          "total spoken time adds up to about 18-19 minutes, leaving buffer for questions.")
pb(doc)

# ============================================================
# SLIDE 1 — TITLE
# ============================================================
slidehdr(doc, 1, TOTAL, "Title slide")
label(doc, "ON SCREEN")
body(doc, "Thesis title, IIT (BHU) logo (top-right), your name Aryan Gupta, roll 21064030, "
          "IDD Civil Engineering Year 5, supervisor Dr. Krishna Kant Pathak, department and institute.")
label(doc, "WHAT TO SAY")
say(doc, "Good morning, respected committee. I am Aryan Gupta, IDD Civil Engineering, "
         "fifth year, roll number 21064030. My thesis is on an AI-powered framework for "
         "multi-objective steel truss optimization, integrating evolutionary algorithms, "
         "neural surrogates, reinforcement learning, and agent-assisted design, all under "
         "IS 800:2007. My supervisor is Dr. Krishna Kant Pathak. Let me walk you through it.")
label(doc, "WHY IT'S HERE")
body(doc, "Formal opening. Sets identity and scope. The long subtitle signals breadth — "
          "five techniques unified under one Indian-code-compliant framework.")
qa(doc, "What does 'multi-objective' mean here?",
   "Two competing objectives — minimum weight and minimum displacement. Most of the work "
   "is single-objective (weight), and NSGA-II handles the multi-objective Pareto front.")
timing(doc, "20-25 seconds")
pb(doc)

# ============================================================
# SLIDE 2 — OUTLINE
# ============================================================
slidehdr(doc, 2, TOTAL, "Outline")
label(doc, "ON SCREEN")
body(doc, "Six parts: 1 Problem, 2 Framework, 3 Results, 4 Analysis, 5 Impact, 6 Conclusion.")
label(doc, "WHAT TO SAY")
say(doc, "The talk has six parts. First the problem and why it matters. Then the framework "
         "I built. Then the results across four benchmarks. Then a deeper analysis. Then the "
         "practical impact. And finally limitations, future work, and your questions.")
label(doc, "WHY IT'S HERE")
body(doc, "Gives the committee a map so they know where you are at any moment. Do NOT read "
          "the whole thing — just name the six parts in one breath.")
timing(doc, "15 seconds")
pb(doc)

# ============================================================
# SLIDE 3 — PART 1 COVER
# ============================================================
slidehdr(doc, 3, TOTAL, "Part 1 divider — The Problem")
label(doc, "ON SCREEN")
body(doc, "Large 'The problem'. Subtitle: steel-truss design in India carries 10-30% surplus "
          "material; can AI reduce that gap under code-compliant constraints?")
label(doc, "WHAT TO SAY")
say(doc, "Let me start with the problem. Steel-truss design in India typically carries 10 to "
         "30 percent surplus material. The question driving this thesis is whether modern AI "
         "can reduce that gap while staying fully code-compliant.")
label(doc, "WHY IT'S HERE")
body(doc, "Section divider. One sentence, then move on. Don't linger on dividers.")
timing(doc, "15-20 seconds")
pb(doc)

# ============================================================
# SLIDE 4 — THE DESIGN GAP
# ============================================================
slidehdr(doc, 4, TOTAL, "The design gap — motivation")
label(doc, "ON SCREEN")
body(doc, "Bullets on India's truss construction, surplus steel, embodied CO2, the scale of impact, "
          "and the speed problem. A motivation figure on the right showing the design gap.")
label(doc, "WHAT TO SAY")
say(doc, "India builds thousands of industrial sheds, transmission towers, and overpasses each "
         "year under IS 800. Classical trial-and-check design typically carries 10 to 30 percent "
         "surplus steel versus the theoretical feasible optimum. Steel embodies about 1.9 kg of "
         "CO2 per kg, and construction is around 11 percent of global emissions, so even a 15 "
         "percent cut across the country is a sector-level impact. Optimization tools exist, but "
         "a full GA run takes 30 to 40 minutes per benchmark — too slow for iterative design in "
         "a project meeting.")
label(doc, "WHY IT'S HERE")
body(doc, "Establishes that the problem is real, large, and economically and environmentally "
          "meaningful — and that existing tools are too slow. This justifies the whole thesis.")
qa(doc, "Where does the 10-30% figure come from?",
   "It's a widely-cited range in structural optimization literature comparing trial-and-check "
   "designs to optimized minimum-weight designs. I cite it in Chapter 1.")
qa(doc, "Is 1.9 kg CO2 per kg of steel accurate?",
   "Yes — that's the typical embodied-carbon figure for primary steel from blast-furnace "
   "routes; recycled steel is lower. The point is the order of magnitude.")
timing(doc, "90 seconds")
pb(doc)

# ============================================================
# SLIDE 5 — RESEARCH QUESTION
# ============================================================
slidehdr(doc, 5, TOTAL, "The research question")
label(doc, "ON SCREEN")
body(doc, "Large text: Can an AI design agent warm-start classical metaheuristics to converge "
          "faster on truss sizing, while staying IS 800-compliant? Tested on 10/25/72/200-bar.")
label(doc, "WHAT TO SAY")
say(doc, "So the precise research question is this: can an AI design agent warm-start classical "
         "metaheuristics, so that they converge faster on truss sizing problems, while staying "
         "IS 800-compliant? I measure this against four classical benchmarks — the 10-bar, "
         "25-bar, 72-bar, and 200-bar trusses.")
label(doc, "WHY IT'S HERE")
body(doc, "This is the single most important slide of Part 1 — it states the testable hypothesis. "
          "Everything afterward is evidence for or against it. Say it slowly and clearly.")
qa(doc, "Why these four benchmarks specifically?",
   "They are the canonical sizing benchmarks in the truss-optimization literature — every "
   "major paper reports them, so I can validate directly against published optima.")
timing(doc, "30 seconds")
pb(doc)

# ============================================================
# SLIDE 6 — FIVE OBJECTIVES
# ============================================================
slidehdr(doc, 6, TOTAL, "Five research objectives")
label(doc, "ON SCREEN")
body(doc, "Table of O1-O5 each with a result pill: O1 reproduce optima within 2% (achieved <0.01%), "
          "O2 Pareto fronts >=20 points (achieved 24-32), O3 >=50x surrogate speedup at R2>0.98 "
          "(achieved 132x), O4 PPO within 5% (mixed, +16%), O5 agent warm-start cuts gens >=20% "
          "(achieved -36.8% on 10-bar).")
label(doc, "WHAT TO SAY")
say(doc, "I set five quantitative objectives. O1: reproduce classical optima within two percent — "
         "achieved under 0.01 percent. O2: NSGA-II fronts with at least 20 points — achieved 24 to "
         "32. O3: at least 50 times surrogate speedup at R-squared above 0.98 — achieved 132 times. "
         "O4: PPO within five percent of the best classical solver — this one is mixed, plus 16 "
         "percent, and I'll explain why. O5: agent warm-start cuts generations by at least 20 "
         "percent — achieved minus 36.8 percent on 10-bar. Four of five fully met; the fifth is "
         "mixed and I report it honestly.")
label(doc, "WHY IT'S HERE")
body(doc, "Shows you set measurable targets in advance and met them — the hallmark of rigorous "
          "work. The honest 'mixed' on O4 builds credibility.")
qa(doc, "Why did PPO not meet its target?",
   "Reinforcement learning is data-hungry and the reward landscape here is highly constrained. "
   "For one-shot benchmark solving, GA and PSO simply win. PPO pays off for repeated solving of "
   "similar problems, which is future work F4.")
timing(doc, "60 seconds")
pb(doc)

# ============================================================
# SLIDE 7 — FIVE CONTRIBUTIONS
# ============================================================
slidehdr(doc, 7, TOTAL, "Five contributions in one view (C1-C5)")
label(doc, "ON SCREEN")
body(doc, "Five cards with big accent numbers: C1 0.004%, C2 132x, C3 549 lb, C4 -36.8%, C5 <10s. "
          "(This is the slide whose numbers were overlapping — now fixed with the compact card layout.)")
label(doc, "WHAT TO SAY")
say(doc, "Here are the five contributions I will defend. C1: end-to-end validation on the 10-bar "
         "to 0.004 percent of literature. C2: a 132 times surrogate speedup at R-squared 0.9993 on "
         "weight. C3: a rigorously feasible 72-bar baseline of 549 pounds — an observation, because "
         "it differs from the soft-penalty literature, pending an independent FEM cross-check. C4: "
         "agent warm-start cuts generations by 36.8 percent on 10-bar, p-value 0.0046. C5: the full "
         "IS 800 pipeline runs end-to-end in under 10 seconds on a laptop CPU, with no API key "
         "required.")
label(doc, "WHY IT'S HERE")
body(doc, "The thesis in five numbers. If you memorize nothing else, memorize these five. Every "
          "results slide later maps back to one of these Cs.")
qa(doc, "Which contribution is the most novel?",
   "C4, the agent warm-start — to my knowledge the first rigorous A/B test of an AI-seeded GA on "
   "truss benchmarks. C2 is the most practically valuable because it makes interactive use possible.")
timing(doc, "60 seconds")
pb(doc)

# ============================================================
# SLIDE 8 — PART 2 COVER
# ============================================================
slidehdr(doc, 8, TOTAL, "Part 2 divider — The Framework")
label(doc, "ON SCREEN")
body(doc, "Large 'The framework'. Subtitle: an 8-layer stack: benchmarks -> FEM -> IS 800 -> "
          "GA/PSO/NSGA-II -> surrogate -> PPO -> agent -> UI.")
label(doc, "WHAT TO SAY")
say(doc, "Now, how I built it. The framework is an eight-layer stack — benchmarks, FEM, IS 800 "
         "checks, three optimizers, a neural surrogate, a PPO agent, the design agent, and a UI.")
timing(doc, "15 seconds")
pb(doc)

# ============================================================
# SLIDE 9 — PROBLEM FORMULATION
# ============================================================
slidehdr(doc, 9, TOTAL, "Problem formulation")
label(doc, "ON SCREEN")
body(doc, "The optimization statement: minimize W(A) = sum of rho*L*A, subject to stress (IS 800 "
          "6.2/6.3/7.1), displacement (5.6.1), slenderness <=180 (3.8), and area bounds. Four "
          "benchmark pills. 'Hard constraints only — no soft penalties.'")
label(doc, "WHAT TO SAY")
say(doc, "Here is the formal problem. I minimize weight, which is a linear sum of areas weighted "
         "by density and length, subject to four constraint families: IS 800 stress checks from "
         "sections 6.2, 6.3, and 7.1; displacement serviceability from 5.6.1; the slenderness "
         "limit of 180 from 3.8; and area side limits. Crucially, all constraints are HARD — there "
         "is no soft penalty. Infeasible designs are always dominated by any feasible design in the "
         "tournament.")
label(doc, "WHY IT'S HERE")
body(doc, "Defines exactly what you optimize and under what rules. The 'hard constraints' line is "
          "the setup for the 72-bar observation later — flag it now so it isn't a surprise.")
qa(doc, "What is the difference between hard and soft constraints?",
   "Soft: add a penalty to the objective when violated, so slightly-infeasible designs can still "
   "score well. Hard: feasibility is tracked separately and any feasible design always beats any "
   "infeasible one. I use hard — it's stricter and matches real code compliance.")
qa(doc, "Why is weight linear but the problem still hard?",
   "The objective is linear, but the constraints — especially buckling and the feasibility "
   "boundary — are non-smooth and non-convex. That's what makes it a hard search problem.")
timing(doc, "80 seconds")
pb(doc)

# ============================================================
# SLIDE 10 — SYSTEM ARCHITECTURE
# ============================================================
slidehdr(doc, 10, TOTAL, "System architecture")
label(doc, "ON SCREEN")
body(doc, "The eight-layer block diagram: Benchmark registry -> Classical optimiser (GA/PSO/NSGA-II) "
          "-> IS 800 compliance; FEM engine -> Neural surrogate MLP -> PPO agent; design agent "
          "(Claude API + cache) -> FastAPI backend -> Streamlit UI.")
label(doc, "WHAT TO SAY")
say(doc, "This single figure is the whole dataflow. The benchmark registry defines the truss. The "
         "FEM engine is the ground truth — areas in, physics out. The neural surrogate sits beside "
         "it as a fast approximation. The optimizers — GA, PSO, NSGA-II — call either FEM or the "
         "surrogate, and every design is checked against IS 800. The design agent seeds the initial "
         "population. The PPO agent has its own training loop. Everything is exposed through a "
         "FastAPI backend and a Streamlit UI.")
label(doc, "WHY IT'S HERE")
body(doc, "Gives the committee the mental model for the rest of Part 2. Point to the flow with your "
          "hand; do NOT read every box aloud.")
qa(doc, "Which layer did you build yourself versus use a library?",
   "FEM, the IS 800 module, the benchmark encodings, the surrogate training pipeline, the agent "
   "prompt and cache, and the UI are mine. The optimizers come from pymoo, the MLP and PPO from "
   "PyTorch and Stable-Baselines3 — standard libraries I configured and wired together.")
timing(doc, "60 seconds")
pb(doc)

# ============================================================
# SLIDE 11 — FEM + IS 800
# ============================================================
slidehdr(doc, 11, TOTAL, "Layer 2 & 3 — FEM kernel and IS 800 compliance")
label(doc, "ON SCREEN")
body(doc, "Left: FEM steps (global stiffness assembly K = sum B^T k B, partition method, solve "
          "K_ff u = F, axial stress recovery) + truss element figure. Right: IS 800 clauses (tension "
          "yield 6.2, rupture 6.3, buckling 7.1, slenderness 3.8, serviceability 5.6.1) + checks figure.")
label(doc, "WHAT TO SAY")
say(doc, "Layers 2 and 3. The FEM kernel is the standard direct stiffness method: assemble the "
         "global stiffness matrix, enforce supports by the partition method, solve K-f-f times u "
         "equals F using a dense solver, and recover axial stresses. It's validated to machine "
         "precision against the 3-bar canonical problem. On top of that, the IS 800 layer checks "
         "five clauses: tension yielding, tension rupture, compression buckling via Perry-Robertson "
         "curve a, the slenderness limit, and serviceability deflection.")
label(doc, "WHY IT'S HERE")
body(doc, "Shows the engineering rigour underneath the AI. Examiners — being civil engineers — care "
          "most about this slide. Know every formula.")
qa(doc, "Why the partition method and not a penalty for supports?",
   "Partition removes fixed DoFs entirely, giving a clean, well-conditioned reduced system. Penalty "
   "methods add a large artificial stiffness which can hurt conditioning.")
qa(doc, "Why only buckling curve a?",
   "Curve a is for hot-rolled sections buckling about the strong axis — the simplest single-curve "
   "default. Selecting the correct curve per section type is a stated limitation.")
qa(doc, "What is the Perry-Robertson formula doing?",
   "It reduces the compression design strength as slenderness rises, accounting for imperfections "
   "— bridging full yield for short columns and Euler buckling for long ones.")
timing(doc, "90 seconds")
pb(doc)

# ============================================================
# SLIDE 12 — THREE OPTIMIZERS
# ============================================================
slidehdr(doc, 12, TOTAL, "Layer 4 — three evolutionary baselines")
label(doc, "ON SCREEN")
body(doc, "Three columns: GA (pop 100, gens 500, SBX, polynomial mutation), PSO (swarm 50, "
          "Clerc constriction), NSGA-II (pop 100, gens 300, non-dominated sort + crowding). "
          "Each with a flowchart figure.")
label(doc, "WHAT TO SAY")
say(doc, "Layer 4 — three evolutionary baselines, all from pymoo. GA: population 100, 500 "
         "generations, SBX crossover, polynomial mutation. PSO: swarm of 50, the standard Clerc "
         "constriction parameters. NSGA-II: the multi-objective version, returning a Pareto front "
         "of weight versus displacement. GA and PSO are two single-objective baselines from "
         "different families — including both shows the framework isn't tied to one optimizer. "
         "All three use feasibility-aware tournaments — no soft penalties.")
label(doc, "WHY IT'S HERE")
body(doc, "These are textbook methods — examiners assume you know them. Keep it brief; spend your "
          "time on the novel layers. This is plumbing, not the contribution.")
qa(doc, "Why not gradient-based optimization?",
   "The constraints are non-smooth — the buckling curve has kinks and feasibility is a step "
   "function — so gradients are unreliable and the feasible region is non-convex. Metaheuristics "
   "only need to rank designs, not differentiate them.")
qa(doc, "Difference between GA and PSO in one line?",
   "GA evolves by mating and mutation — big stochastic jumps. PSO drifts particles toward personal "
   "and global best — smooth, smaller steps.")
timing(doc, "70 seconds")
pb(doc)

# ============================================================
# SLIDE 13 — SURROGATE
# ============================================================
slidehdr(doc, 13, TOTAL, "Layer 5 — neural surrogate")
label(doc, "ON SCREEN")
body(doc, "MLP architecture [n_A -> 256 -> 128 -> 64 -> 3], ReLU, dropout 0.2; trained on 10,000 "
          "LHS samples, Adam, 200 epochs; outputs weight/stress/displacement; hybrid mode with "
          "FEM fallback. Architecture figure.")
label(doc, "WHAT TO SAY")
say(doc, "Layer 5, the neural surrogate — and this is where the speedup comes from. A GA run needs "
         "about 50,000 FEM evaluations, which is slow. So I trained a small MLP — 256, 128, 64 "
         "hidden units — on 10,000 Latin-hypercube-sampled designs that FEM had already evaluated. "
         "The network learns to predict weight, stress, and displacement directly. In the GA inner "
         "loop it replaces FEM, bringing a full run from 12.4 seconds to 94 milliseconds — 132 "
         "times faster, R-squared 0.9993 on weight. Because stress and displacement are noisier, I "
         "run a hybrid mode: the surrogate screens, and borderline designs fall back to real FEM.")
label(doc, "WHY IT'S HERE")
body(doc, "This is C2 — the practical contribution. The key insight to convey: the training data is "
          "self-generated by FEM, not downloaded.")
qa(doc, "Where did the training dataset come from?",
   "It's self-generated. I draw 10,000 area vectors by Latin Hypercube Sampling and evaluate each "
   "with my own FEM. That input-output table is the dataset — nothing is downloaded.")
qa(doc, "Why is weight R2 0.9993 but stress only 0.81?",
   "Weight is linear in area — trivial to learn. Stress depends on which bar is critical, which can "
   "switch discontinuously as areas change, so it's harder. That's why stress is a screen, not the "
   "objective.")
qa(doc, "Why 10,000 samples?",
   "A sweep showed R2 on weight saturates above 10,000 — fewer hurts accuracy, more gives "
   "diminishing returns.")
timing(doc, "80 seconds")
pb(doc)

# ============================================================
# SLIDE 14 — PPO + AGENT
# ============================================================
slidehdr(doc, 14, TOTAL, "Layers 6 & 7 — RL agent and design-agent warm-start")
label(doc, "ON SCREEN")
body(doc, "Left: PPO MDP (state = areas + max stress/disp; action = multiplicative adjustment "
          "[0.5,2.0]; reward = -W - lambda*infeasibility) + MDP figure. Right: design agent "
          "(structured prompt -> Claude returns k=8 designs + reasoning; injected as seeds; cached) "
          "+ pipeline figure.")
label(doc, "WHAT TO SAY")
say(doc, "Layers 6 and 7. The PPO agent treats sizing as a Markov decision process — the state is "
         "the current area vector plus the worst stress and displacement, the action scales each "
         "area between half and double, and the reward is negative weight minus an infeasibility "
         "penalty. It's trained for 150,000 timesteps. On the right is the design agent — the novel "
         "piece. I send a structured prompt describing the geometry, loads, material, and IS 800 "
         "constraints, and the agent returns eight candidate designs with reasoning. These seed the "
         "GA's initial population. Every response is cached, so the pipeline rebuilds offline with "
         "no API key.")
label(doc, "WHY IT'S HERE")
body(doc, "Introduces the two AI layers. Be crisp that the design agent is a SEED, not the "
          "optimizer. This sets up C4 in the results.")
qa(doc, "Is the agent doing the optimization?",
   "No. The agent proposes eight starting designs for generation zero. The GA does all the "
   "optimization. The agent is a smart initializer, not an optimizer.")
qa(doc, "What model is the agent?",
   "Zero-shot Claude via the public API at temperature 0.3 — not fine-tuned. I deliberately test "
   "the weakest version of the idea; fine-tuning is future work F5.")
timing(doc, "90 seconds")
pb(doc)

# ============================================================
# SLIDE 15 — PART 3 COVER
# ============================================================
slidehdr(doc, 15, TOTAL, "Part 3 divider — The Results")
label(doc, "ON SCREEN")
body(doc, "Large 'The results'. Subtitle: four benchmarks, five headline findings, one observation "
          "on the 72-bar literature.")
label(doc, "WHAT TO SAY")
say(doc, "Now the proof — four benchmarks, five headline findings, and one important observation on "
         "the 72-bar literature.")
timing(doc, "15 seconds")
pb(doc)

# ============================================================
# SLIDE 16 — 10-BAR
# ============================================================
slidehdr(doc, 16, TOTAL, "10-bar planar — end-to-end validation (C1)")
label(doc, "ON SCREEN")
body(doc, "Convergence curve on the left; table on the right: literature 5060.85, PSO 5061.05, "
          "GA 5062.78, NSGA-II 5081.51. 'PSO error = 0.004%.'")
label(doc, "WHAT TO SAY")
say(doc, "The 10-bar cantilever is the most-cited truss benchmark. The published optimum is "
         "5060.85 pounds. With PSO over five seeds I reach 5061.05 pounds — an error of 0.004 "
         "percent. GA over ten seeds reaches 5062.78. This is the end-to-end signature that my "
         "FEM, IS 800, and optimizer stack are all working correctly together. This is contribution "
         "C1.")
label(doc, "WHY IT'S HERE")
body(doc, "Validation against ground truth. If the framework nails the most-studied benchmark in "
          "the world to 0.004 percent, the committee trusts every later number.")
qa(doc, "Why is your number slightly different from 5060.85?",
   "Different optimizer and seed converge to within a fraction of a percent of the same global "
   "optimum. 0.004 percent is well inside the spread reported across 20-plus papers on this "
   "benchmark.")
timing(doc, "70 seconds")
pb(doc)

# ============================================================
# SLIDE 17 — BENCHMARK MATRIX
# ============================================================
slidehdr(doc, 17, TOTAL, "25-bar, 72-bar, 200-bar — full benchmark matrix")
label(doc, "ON SCREEN")
body(doc, "Table: 10-bar PSO/GA ~5061; 25-bar ~545 vs lit 545.16; 72-bar ~549 vs soft-penalty lit "
          "379.62 (dagger); 200-bar 315.89 vs lit 25445. Footnote on the dagger.")
label(doc, "WHAT TO SAY")
say(doc, "The full matrix. 10-bar within 0.004 percent. 25-bar within 0.03 percent of literature. "
         "72-bar converges to about 549 pounds — and the dagger marks the soft-penalty literature "
         "value of 380 pounds, which I discuss next. The 200-bar figures use a different problem "
         "variant with different loading, specified in Appendix C.")
label(doc, "WHY IT'S HERE")
body(doc, "Shows breadth — the framework handles 2D and 3D, small and large. Sets up the 72-bar "
          "discussion. Don't over-explain the 200-bar units; just flag the variant.")
qa(doc, "Why does the 200-bar differ so much from 25445?",
   "The 25445-pound value is a different 200-bar variant with separate loads and grouping. The "
   "benchmark name covers multiple variants; mine is the Lee & Geem 2004 reduced-load variant, "
   "documented in Appendix C.")
timing(doc, "90 seconds")
pb(doc)

# ============================================================
# SLIDE 18 — 72-BAR OBSERVATION
# ============================================================
slidehdr(doc, 18, TOTAL, "The 72-bar hard-constraint baseline (C3)")
label(doc, "ON SCREEN")
body(doc, "Hard-vs-soft comparison figure; text explaining ~549 lb under hard constraints vs ~380 lb "
          "soft-penalty literature; 'observation — future work F7'.")
label(doc, "WHAT TO SAY")
say(doc, "This is the one result that needs careful explanation. Under my hard-constraint FEM, the "
         "72-bar optimum converges to about 549 pounds, with cross-seed spread under one percent. I "
         "could not reproduce the soft-penalty values of around 380 pounds from Camp and Bichon 2004 "
         "and Bekdas 2015 as feasible. The gap most likely reflects a difference in constraint-"
         "handling convention, not an error in the published work. I flag it as observation C3 and "
         "propose an independent FEM cross-check using OpenSeesPy or SAP2000 as future work F7.")
label(doc, "WHY IT'S HERE")
body(doc, "The most likely target for hostile questioning. Owning it — framing it as an honest "
          "observation rather than a claim that the literature is wrong — is what protects you.")
qa(doc, "So are the published papers wrong?",
   "Not necessarily. They use soft penalties that tolerate small violations, so their reported "
   "optima can sit at or just past the constraint boundary. I use hard constraints. Both are valid "
   "choices — my point is to provide a rigorously feasible baseline, not to discredit prior work.")
qa(doc, "How do you know your 549 is right then?",
   "Cross-seed spread is under one percent and every constraint is strictly satisfied under FEM. "
   "The independent cross-check in F7 would confirm it against a third-party solver.")
timing(doc, "90 seconds")
pb(doc)

# ============================================================
# SLIDE 19 — SURROGATE PARITY
# ============================================================
slidehdr(doc, 19, TOTAL, "Surrogate parity and wall-clock speedup (C2)")
label(doc, "ON SCREEN")
body(doc, "Left: parity plot (surrogate vs FEM, tight on y=x). Right: wall-clock bar (12.4s -> 0.094s). "
          "Pills: weight R2 0.9993, stress R2 0.81, displacement R2 0.87.")
label(doc, "WHAT TO SAY")
say(doc, "Contribution C2. On the left, the parity plot — surrogate prediction against FEM truth on "
         "a held-out test split, tight on the y-equals-x line, R-squared 0.9993 on weight. On the "
         "right, wall-clock — 12.4 seconds with FEM down to 94 milliseconds with the surrogate, a "
         "132 times speedup. Stress and displacement heads are noisier, R-squared 0.81 and 0.87, so "
         "I use them as feasibility screens with FEM fallback on the constraint boundary.")
label(doc, "WHY IT'S HERE")
body(doc, "Visual proof of C2. The parity plot is the single most convincing image — it shows the "
          "surrogate is essentially indistinguishable from FEM on weight.")
qa(doc, "Doesn't the lower stress R2 make the result unsafe?",
   "Only if used naively. Weight, the objective, is at 0.9993. Stress and displacement are screens "
   "— borderline cases are re-checked with real FEM, so final feasibility is always FEM-verified.")
timing(doc, "80 seconds")
pb(doc)

# ============================================================
# SLIDE 20 — AGENT EFFECT MAP
# ============================================================
slidehdr(doc, 20, TOTAL, "Agent warm-start effect-map (C4)")
label(doc, "ON SCREEN")
body(doc, "Bar chart of % change in generations; table: 10-bar -36.8% (p=0.0046), 25-bar -76.6% "
          "(p=0.25, n=3), 72-bar +4.8% (p=0.81). Note on the redundant bar set {2,5,6,10}.")
label(doc, "WHAT TO SAY")
say(doc, "Contribution C4, and the part I'm proudest of. On the 10-bar, agent warm-start cuts "
         "generations-to-convergence by 36.8 percent, p-value 0.0046 — statistically significant. "
         "On 25-bar the magnitude is even larger, minus 76.6 percent, but with only three seeds it's "
         "underpowered, p 0.25. On 72-bar, no detectable effect. The interpretation: the agent helps "
         "most when it can surface an architectural insight. On 10-bar, the agent correctly "
         "identifies that bars 2, 5, 6, and 10 are nearly redundant — a known result the GA would "
         "otherwise rediscover by trial and error. On 72-bar there's no analogous insight, so the "
         "effect vanishes.")
label(doc, "WHY IT'S HERE")
body(doc, "The headline novelty. The qualified, honest framing — works when insight exists, null "
          "otherwise — is the defensible scientific claim. Never overclaim here.")
qa(doc, "Could the GA find the redundant bars without the agent?",
   "Yes, eventually. The agent doesn't discover anything unreachable — it just provides the insight "
   "earlier, cutting the generations the GA spends rediscovering it. That earlier start is the "
   "measured speedup.")
qa(doc, "Is one significant benchmark enough to claim the effect?",
   "I claim it for the regime where an architectural insight exists, evidenced significantly on "
   "10-bar. I explicitly do not claim universal benefit — 72-bar shows the null case. That honesty "
   "is the contribution.")
timing(doc, "90 seconds")
pb(doc)

# ============================================================
# SLIDE 21 — NSGA-II PARETO
# ============================================================
slidehdr(doc, 21, TOTAL, "NSGA-II Pareto front (O2)")
label(doc, "ON SCREEN")
body(doc, "Pareto front scatter in weight vs max-displacement space, 24-32 non-dominated points.")
label(doc, "WHAT TO SAY")
say(doc, "Here is the 10-bar Pareto front — weight against maximum displacement. Each point is a "
         "non-dominated design; 24 to 32 per seed, satisfying objective O2 with margin. The "
         "single-objective optimum sits at the rightmost corner; lighter designs to the left all "
         "deflect more. This is the menu of trade-offs an engineer chooses from.")
label(doc, "WHY IT'S HERE")
body(doc, "Demonstrates the multi-objective capability and answers O2. Shows the framework gives "
          "design choices, not just one answer.")
qa(doc, "How do you pick one design from the front?",
   "That's an engineering decision — based on budget and stiffness requirements. The framework "
   "presents the trade-off; the engineer picks the knee point or a code-driven displacement target.")
timing(doc, "50 seconds")
pb(doc)

# ============================================================
# SLIDE 22 — PART 4 COVER
# ============================================================
slidehdr(doc, 22, TOTAL, "Part 4 divider — Deeper Analysis")
label(doc, "ON SCREEN")
body(doc, "Large 'Deeper analysis'. Subtitle: where the surrogate is trustworthy, how fast each "
          "optimizer converges, and what the agent actually said.")
label(doc, "WHAT TO SAY")
say(doc, "Three deeper questions now: how trustworthy is the surrogate, how fast does each optimizer "
         "converge, and what does the agent actually say.")
timing(doc, "15 seconds")
pb(doc)

# ============================================================
# SLIDE 23 — MC DROPOUT
# ============================================================
slidehdr(doc, 23, TOTAL, "Surrogate uncertainty via MC-dropout")
label(doc, "ON SCREEN")
body(doc, "Calibration figure; bullets: T=40 stochastic passes, calibration test vs 0.95 target, "
          "weight head 0.96 (calibrated), displacement head 0.82 (under-confident, safe).")
label(doc, "WHAT TO SAY")
say(doc, "Is a 132 times speedup safe to ship? I test it with MC-dropout — keeping dropout active at "
         "inference, running 40 stochastic forward passes, and using the variance as an uncertainty "
         "estimate. The calibration target is 95 percent of true values inside two sigma. The weight "
         "head measures 96 percent — well calibrated. The displacement head is 82 percent — "
         "under-confident, which is the safe direction. The recommendation: use the weight head as "
         "the objective and keep FEM for the final feasibility check.")
label(doc, "WHY IT'S HERE")
body(doc, "Shows you didn't just trust the surrogate blindly — you quantified its reliability. This "
          "is the kind of rigour that impresses a technical committee.")
qa(doc, "What is MC-dropout in one sentence?",
   "Monte-Carlo dropout: keep dropout on at inference and run many passes; the spread of "
   "predictions approximates the model's uncertainty.")
timing(doc, "70 seconds")
pb(doc)

# ============================================================
# SLIDE 24 — CONVERGENCE RATE
# ============================================================
slidehdr(doc, 24, TOTAL, "Convergence-rate characterisation")
label(doc, "ON SCREEN")
body(doc, "Exponential-fit figure W(t)=W_inf + A e^(-t/tau); table: PSO tau 8.7, GA 23.5, agent+GA "
          "14.9 generations.")
label(doc, "WHAT TO SAY")
say(doc, "I fit an exponential to each seed's convergence curve, where tau is the time constant — "
         "how many generations to halve the remaining gap to the optimum. PSO converges fastest at "
         "8.7 generations. GA is 23.5. GA with agent warm-start is 14.9 — shifted roughly halfway "
         "toward PSO. So the agent warm-start mechanism makes GA behave more like PSO in speed, "
         "which is exactly consistent with contribution C4.")
label(doc, "WHY IT'S HERE")
body(doc, "Explains the MECHANISM behind C4 quantitatively — not just 'it's faster' but 'here is the "
          "time constant shifting'. Strong supporting evidence.")
qa(doc, "Why fit an exponential specifically?",
   "Convergence of these metaheuristics is approximately geometric toward the optimum, so an "
   "exponential is the natural model and tau gives a single comparable speed number per run.")
timing(doc, "70 seconds")
pb(doc)

# ============================================================
# SLIDE 25 — HARD VS SOFT
# ============================================================
slidehdr(doc, 25, TOTAL, "Hard- vs soft-constraint feasibility frontier")
label(doc, "ON SCREEN")
body(doc, "Pareto front shift figure (no IS 800 vs full IS 800); caption on 549 vs 380 lb and "
          "constraint-handling sensitivity; F7.")
label(doc, "WHAT TO SAY")
say(doc, "This is the analysis behind the 72-bar observation. The same problem under hard versus "
         "soft constraints produces very different fronts. Under hard IS 800, the feasible mass sits "
         "around 549 pounds; under a representative soft penalty it drops to around 380. The point is "
         "not that one is right and one wrong — it is that reported optima in the truss-sizing "
         "literature can be highly sensitive to the constraint-handling convention. For Indian "
         "practice, this means a soft-penalty optimum should not be deployed without a hard-"
         "feasibility cross-check.")
label(doc, "WHY IT'S HERE")
body(doc, "Elevates the 72-bar gap from 'my number disagrees' to a genuine engineering insight about "
          "the field. This is what turns a discrepancy into a contribution.")
qa(doc, "Is this sensitivity a known issue in the literature?",
   "Constraint handling is known to affect results, but a side-by-side hard-vs-soft ablation on the "
   "same FEM, quantifying the magnitude, is what I add here.")
timing(doc, "90 seconds")
pb(doc)

# ============================================================
# SLIDE 26 — PART 5 COVER
# ============================================================
slidehdr(doc, 26, TOTAL, "Part 5 divider — Impact")
label(doc, "ON SCREEN")
body(doc, "Large 'Impact'. Subtitle: reproducibility, civil-engineering practice, live demo.")
label(doc, "WHAT TO SAY")
say(doc, "Now the practical impact — for Indian civil engineering, for reproducibility, and a live "
         "demo.")
timing(doc, "10 seconds")
pb(doc)

# ============================================================
# SLIDE 27 — PRACTICAL IMPLICATIONS
# ============================================================
slidehdr(doc, 27, TOTAL, "Practical implications for Indian civil engineering")
label(doc, "ON SCREEN")
body(doc, "Three columns: interactive design (30-40 min -> <20 s), constraint-handling sensitivity, "
          "zero-Python design aid via Streamlit.")
label(doc, "WHAT TO SAY")
say(doc, "Three practical implications. First, the surrogate brings a 30-to-40-minute GA run down to "
         "under 20 seconds, enabling parametric studies during a review meeting instead of overnight "
         "batches. Second, the hard-versus-soft sensitivity should be flagged to anyone using "
         "commercial soft-penalty solvers — run a hard-feasibility check before sign-off. Third, the "
         "Streamlit UI exposes the whole framework without requiring Python literacy — usable by a "
         "junior engineer, not just a researcher.")
label(doc, "WHY IT'S HERE")
body(doc, "Connects the technical work to real-world value. Examiners want to know 'so what' — this "
          "answers it concretely for the Indian context.")
timing(doc, "70 seconds")
pb(doc)

# ============================================================
# SLIDE 28 — REPRODUCIBILITY
# ============================================================
slidehdr(doc, 28, TOTAL, "Reproducibility contract")
label(doc, "ON SCREEN")
body(doc, "Pinned versions, fixed seed set, 89 refs / 27 figures / 14 CSVs / 21 pickles checked in, "
          "cached agent responses, green CI, single-command PDF rebuild.")
label(doc, "WHAT TO SAY")
say(doc, "Reproducibility is a first-class deliverable. Every dependency is pinned. Seeds are fixed "
         "and logged. All 89 references, 27 figures, 14 CSVs, and 21 optimization histories are "
         "checked in. Every agent response is cached, so the whole thing rebuilds offline with no "
         "API key. Continuous integration runs the fast test suite green on every push, and a single "
         "Tectonic command rebuilds both the 125-page thesis and this deck.")
label(doc, "WHY IT'S HERE")
body(doc, "Reproducibility is increasingly valued in examinations. This slide says 'anyone can "
          "verify everything I claim' — a strong credibility signal.")
qa(doc, "Doesn't using an agent break reproducibility?",
   "No — I cache every response by prompt hash. The cache is committed, so replays are deterministic "
   "and need no API key.")
timing(doc, "60 seconds")
pb(doc)

# ============================================================
# SLIDE 29 — LIVE DEMO
# ============================================================
slidehdr(doc, 29, TOTAL, "Live demo surface")
label(doc, "ON SCREEN")
body(doc, "Three interfaces: Streamlit UI (localhost:8501), FastAPI (localhost:8000 + Swagger), "
          "thesis PDF + deck.")
label(doc, "WHAT TO SAY")
say(doc, "The same framework, three interfaces. The Streamlit UI lets a user pick a benchmark, an "
         "algorithm, a seed, and the agent warm-start toggle, then shows the live convergence plot, "
         "the IS 800 report, and the cross-section chart. The FastAPI backend exposes the same stack "
         "programmatically, with Swagger docs. And the thesis and this deck both build from the same "
         "tagged repository. If the committee wishes, I can run a 10-bar optimization live right now.")
label(doc, "WHY IT'S HERE")
body(doc, "Offers the live demo. IMPORTANT: have the server already running before viva. If asked, "
          "run a baseline 10-bar, then toggle the agent and show the faster convergence.")
qa(doc, "Can a user enter their own custom truss?",
   "Not through this UI yet — the four benchmarks are encoded in the registry. But a new benchmark "
   "is a short Python file of nodes, connectivity, loads, and supports; the framework is modular. "
   "Custom-truss UI is future work.")
timing(doc, "60 seconds (longer if running the live demo)")
pb(doc)

# ============================================================
# SLIDE 30 — PART 6 COVER
# ============================================================
slidehdr(doc, 30, TOTAL, "Part 6 divider — Conclusion")
label(doc, "ON SCREEN")
body(doc, "Large 'Conclusion'. Subtitle: limitations, future work, and a concrete ask.")
label(doc, "WHAT TO SAY")
say(doc, "Finally — limitations, future work, and a summary.")
timing(doc, "10 seconds")
pb(doc)

# ============================================================
# SLIDE 31 — LIMITATIONS
# ============================================================
slidehdr(doc, 31, TOTAL, "What I am NOT claiming — limitations")
label(doc, "ON SCREEN")
body(doc, "Bullets: linear-elastic small-displacement FEM; sizing-only not topology; NSGA-II 3 seeds; "
          "PPO single-instance no transfer; single buckling curve; agent zero-shot not fine-tuned.")
label(doc, "WHAT TO SAY")
say(doc, "Let me be explicit about what I am NOT claiming. The FEM is linear elastic and small-"
         "displacement — no non-linearity or dynamics. The optimization is sizing only, not topology "
         "— connectivity is fixed. The NSGA-II 3D runs use three seeds, so O2's statistical power is "
         "weaker than O1's. The PPO agent is trained on one instance with no transfer. The IS 800 "
         "module uses a single buckling curve. And the design agent is zero-shot, not fine-tuned. "
         "None of these are load-bearing for contributions C1 through C5 — but they set the boundary "
         "of what I will defend.")
label(doc, "WHY IT'S HERE")
body(doc, "Stating limitations clearly and confidently is a credibility multiplier. It pre-empts "
          "hostile questions and shows scientific maturity. Do not apologize — state them as "
          "deliberate scope decisions.")
qa(doc, "Why didn't you do topology optimization?",
   "Topology turns it into a mixed integer-continuous problem and is a distinct research area. "
   "Combining my agent warm-start with topology is a natural follow-on — future work F2.")
timing(doc, "90 seconds")
pb(doc)

# ============================================================
# SLIDE 32 — FUTURE WORK
# ============================================================
slidehdr(doc, 32, TOTAL, "Future work (F1-F7)")
label(doc, "ON SCREEN")
body(doc, "F1 nonlinear/dynamic FEM -> IS 1893; F2 topology via ground-structure/SIMP; F3 GNN "
          "surrogate; F4 cross-instance PPO transfer; F5 fine-tuned domain agent; F6 PGCIL "
          "transmission-tower validation; F7 independent FEM cross-check of 72-bar.")
label(doc, "WHAT TO SAY")
say(doc, "Seven directions of future work. F1: non-linear and dynamic FEM for IS 1893 seismic. F2: "
         "topology optimization. F3: a graph-neural-network surrogate for variable topology. F4: "
         "cross-instance PPO transfer. F5: a fine-tuned domain-specific agent on IS 800 and IS 875. "
         "F6: real-world validation on a PGCIL transmission tower. And F7: an independent FEM cross-"
         "check of the 72-bar discrepancy using OpenSeesPy or SAP2000.")
label(doc, "WHY IT'S HERE")
body(doc, "Shows the work opens doors rather than closing them. F7 directly addresses the 72-bar "
          "question — point to it if pressed on that result.")
timing(doc, "70 seconds")
pb(doc)

# ============================================================
# SLIDE 33 — SUMMARY
# ============================================================
slidehdr(doc, 33, TOTAL, "Summary")
label(doc, "ON SCREEN")
body(doc, "C1-C5 restated; '125 pages, 27 figures, 89 references; 35-slide deck; laptop CPU; tag "
          "phase-10-complete'.")
label(doc, "WHAT TO SAY")
say(doc, "To summarize. C1: 10-bar to literature at 0.004 percent — the full stack validated. C2: "
         "132 times surrogate speedup at R-squared 0.9993 — interactive use is practical. C3: a "
         "rigorously feasible 72-bar baseline at 549 pounds, with the gap versus soft-penalty "
         "literature flagged for an independent cross-check. C4: agent warm-start saves 36.8 percent "
         "of generations on 10-bar at p 0.0046, diminishing on harder 3D problems. C5: an IS 800-"
         "compliant design aid running under 10 seconds on a laptop CPU, with no API key needed to "
         "rebuild. The thesis is 125 pages, 27 figures, 89 references; everything is tagged "
         "phase-10-complete.")
label(doc, "WHY IT'S HERE")
body(doc, "Last chance to plant the five numbers in the committee's memory before questions. Hit "
          "all five Cs cleanly.")
timing(doc, "70 seconds")
pb(doc)

# ============================================================
# SLIDE 34 — THANK YOU
# ============================================================
slidehdr(doc, 34, TOTAL, "Thank you / Questions")
label(doc, "ON SCREEN")
body(doc, "'Questions?' with your name, roll, supervisor, repository, and demo URL.")
label(doc, "WHAT TO SAY")
say(doc, "Thank you for your attention. I would be happy to take your questions.")
label(doc, "WHY IT'S HERE")
body(doc, "Clean close. Pause, smile, take a sip of water, and wait for the first question. Repeat "
          "each question back in your own words before answering — it buys thinking time and "
          "confirms understanding.")
timing(doc, "10 seconds, then Q&A")
pb(doc)

# ============================================================
# CLOSING — TIMING + RULES
# ============================================================
H1(doc, "Timing Map & Golden Rules")
label(doc, "TIME BUDGET (target 18-19 min, ~1-2 min buffer)")
bullet(doc, "Slides 1-2 (open + outline): ~0:40", prefix="Intro")
bullet(doc, "Slides 3-7 (the problem): ~3:00", prefix="Part 1")
bullet(doc, "Slides 8-14 (the framework): ~5:00", prefix="Part 2")
bullet(doc, "Slides 15-21 (the results): ~5:00", prefix="Part 3")
bullet(doc, "Slides 22-25 (deeper analysis): ~3:00", prefix="Part 4")
bullet(doc, "Slides 26-29 (impact): ~2:00", prefix="Part 5")
bullet(doc, "Slides 30-34 (conclusion): ~2:30", prefix="Part 6")

label(doc, "GOLDEN RULES")
bullet(doc, "Never compress Part 1 (the problem) — examiners anchor every later question to it.")
bullet(doc, "If running long, compress Part 4 (analysis) — it's supporting evidence, not core claims.")
bullet(doc, "Spend the LEAST time on slide 12 (the three optimizers) — they're textbook plumbing.")
bullet(doc, "Spend the MOST care on slides 7, 18, 20 (contributions, 72-bar, agent effect) — these draw questions.")
bullet(doc, "Have the demo server running BEFORE the viva. Never start it live.")
bullet(doc, "State limitations (slide 31) confidently — they are scope decisions, not failures.")
bullet(doc, "Memorize the five C-numbers: 0.004% / 132x & R2 0.9993 / 549 lb vs 380 / -36.8% p=0.0046 / <10 s.")

p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(16); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("— Read this once the night before, once the morning of. You're ready. —")
r.font.name = "Calibri"; r.font.size = Pt(12); r.italic = True; r.font.color.rgb = MUTED

out = "/home/user/thesis/viva_prep/Slide_By_Slide_Explainer.docx"
doc.save(out)
print("Wrote", out)
