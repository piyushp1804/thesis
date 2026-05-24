"""
Definitive slide-by-slide viva explainer — WITH embedded slide images and
expanded detail. For each of the 34 slides:
  - the slide thumbnail (layout reference)
  - ON SCREEN, WHAT TO SAY (verbatim), DEEPER CONTEXT (the real teaching),
    WHY IT'S HERE, LIKELY Q&A (expanded), TIMING.
Output: /home/user/thesis/viva_prep/Slide_By_Slide_Explainer.docx
"""

import os
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
PURPLE = RGBColor(0x6A, 0x2C, 0x91)

IMG_DIR = "/home/user/thesis/viva_prep/slide_imgs"


def shade(cell, hexfill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexfill)
    tcPr.append(shd)


def hr(p):
    pPr = p._p.get_or_add_pPr(); bdr = OxmlElement("w:pBdr"); b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "6"); b.set(qn("w:space"), "1"); b.set(qn("w:color"), "D6DCE6")
    bdr.append(b); pPr.append(bdr)


def H1(doc, t):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(22); p.paragraph_format.space_after = Pt(6)
    r = p.add_run(t); r.font.name = "Calibri"; r.font.size = Pt(24); r.bold = True; r.font.color.rgb = INK
    hr(p)


def slidehdr(doc, num, total, title):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(14); p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"SLIDE {num} / {total}")
    r.font.name = "Calibri"; r.font.size = Pt(10); r.bold = True; r.font.color.rgb = ACCENT
    p2 = doc.add_paragraph(); p2.paragraph_format.space_after = Pt(4)
    r2 = p2.add_run(title); r2.font.name = "Calibri"; r2.font.size = Pt(16); r2.bold = True; r2.font.color.rgb = INK
    hr(p2)


def slide_img(doc, num):
    path = os.path.join(IMG_DIR, f"slide_{num:02d}.png")
    if os.path.exists(path):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run()
        run.add_picture(path, width=Inches(5.8))
        # subtle border via caption
        cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cr = cap.add_run("↑ slide layout (text wording shown is from an earlier build; your rebuilt deck reads 'agent')")
        cr.font.name = "Calibri"; cr.font.size = Pt(8); cr.italic = True; cr.font.color.rgb = MUTED


def label(doc, text, color=ACCENT):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(1)
    r = p.add_run(text); r.font.name = "Calibri"; r.font.size = Pt(10.5); r.bold = True; r.font.color.rgb = color


def body(doc, text, italic=False, color=INK, size=11, after=5):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(after)
    r = p.add_run(text); r.font.name = "Calibri"; r.font.size = Pt(size); r.italic = italic; r.font.color.rgb = color


def say(doc, text):
    table = doc.add_table(rows=1, cols=1); cell = table.cell(0, 0); shade(cell, "F3F7FF")
    cell.paragraphs[0].text = ""
    r = cell.paragraphs[0].add_run("“" + text + "”")
    r.font.name = "Calibri"; r.font.size = Pt(11); r.italic = True; r.font.color.rgb = INK
    tcPr = cell._tc.get_or_add_tcPr(); b = OxmlElement("w:tcBorders"); left = OxmlElement("w:left")
    left.set(qn("w:val"), "single"); left.set(qn("w:sz"), "24"); left.set(qn("w:color"), "0A66FF")
    b.append(left); tcPr.append(b)
    doc.add_paragraph()


def deep(doc, text):
    """Deeper-context paragraph in a tinted box."""
    table = doc.add_table(rows=1, cols=1); cell = table.cell(0, 0); shade(cell, "FBF6FF")
    cell.paragraphs[0].text = ""
    r = cell.paragraphs[0].add_run(text)
    r.font.name = "Calibri"; r.font.size = Pt(10.5); r.font.color.rgb = INK
    tcPr = cell._tc.get_or_add_tcPr(); b = OxmlElement("w:tcBorders"); left = OxmlElement("w:left")
    left.set(qn("w:val"), "single"); left.set(qn("w:sz"), "24"); left.set(qn("w:color"), "6A2C91")
    b.append(left); tcPr.append(b)
    doc.add_paragraph()


def bullet(doc, text, prefix=None):
    p = doc.add_paragraph(style="List Bullet"); p.paragraph_format.space_after = Pt(2)
    if prefix:
        r = p.add_run(prefix + "  "); r.font.name = "Calibri"; r.font.size = Pt(10.5); r.bold = True; r.font.color.rgb = INK
    r2 = p.add_run(text); r2.font.name = "Calibri"; r2.font.size = Pt(10.5); r2.font.color.rgb = INK


def qa(doc, q, a):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(1)
    r = p.add_run("Likely Q:  "); r.font.name = "Calibri"; r.font.size = Pt(10.5); r.bold = True; r.font.color.rgb = WARN
    r2 = p.add_run(q); r2.font.name = "Calibri"; r2.font.size = Pt(10.5); r2.italic = True; r2.font.color.rgb = INK
    p2 = doc.add_paragraph(); p2.paragraph_format.left_indent = Inches(0.25); p2.paragraph_format.space_after = Pt(6)
    r3 = p2.add_run("Answer:  "); r3.font.name = "Calibri"; r3.font.size = Pt(10.5); r3.bold = True; r3.font.color.rgb = GREEN
    r4 = p2.add_run(a); r4.font.name = "Calibri"; r4.font.size = Pt(10.5); r4.font.color.rgb = INK


def timing(doc, t):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(2)
    r = p.add_run("Time on this slide: " + t); r.font.name = "Calibri"; r.font.size = Pt(10); r.bold = True; r.font.color.rgb = MUTED


def pb(doc):
    doc.add_page_break()


doc = Document()
for s in doc.sections:
    s.top_margin = Cm(1.6); s.bottom_margin = Cm(1.6); s.left_margin = Cm(1.8); s.right_margin = Cm(1.8)
doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(11)

TOTAL = 34

# ---------- COVER ----------
p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(40)
r = p.add_run("VIVA DECK — SLIDE-BY-SLIDE EXPLAINER (ILLUSTRATED, EXPANDED)")
r.font.name = "Calibri"; r.font.size = Pt(11); r.bold = True; r.font.color.rgb = MUTED
p = doc.add_paragraph()
r = p.add_run("Every Slide, Pictured and Fully Explained"); r.font.name = "Calibri"; r.font.size = Pt(28); r.bold = True; r.font.color.rgb = INK
p = doc.add_paragraph()
r = p.add_run("AI-Powered Framework for Multi-Objective Steel Truss Optimization\n"
              "For each of the 34 slides: the slide picture, exactly what to say, the deeper "
              "concept behind it, why it's there, and the questions that will follow. Read only "
              "this and you are ready.")
r.font.name = "Calibri"; r.font.size = Pt(13); r.italic = True; r.font.color.rgb = MUTED
p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(24); hr(p)
p = doc.add_paragraph()
r = p.add_run("Aryan Gupta  •  Roll 21064030  •  IDD Civil & Structural Engineering (Y5)\n"
              "Supervisor: Dr. Krishna Kant Pathak  •  IIT (BHU) Varanasi")
r.font.name = "Calibri"; r.font.size = Pt(12); r.font.color.rgb = INK

label(doc, "ABOUT THE THUMBNAILS", WARN)
body(doc, "The slide pictures are rendered from an earlier build of the deck, so the wording in "
          "the image may still say 'LLM' and the contributions slide may look cramped. Your "
          "freshly-rebuilt deck fixes all of that (it reads 'agent', the contributions numbers no "
          "longer overlap, your name and the IIT BHU logo are on the title). Use the pictures for "
          "LAYOUT — where each element sits — and trust the TEXT in this document for the current "
          "wording.", size=10.5)

label(doc, "THE FIVE NUMBERS TO MEMORIZE", PURPLE)
body(doc, "C1 = 0.004% (10-bar vs literature)   •   C2 = 132× speedup & R²=0.9993   •   "
          "C3 = 549 lb hard vs 380 lb soft (72-bar)   •   C4 = −36.8% generations, p=0.0046   •   "
          "C5 = <10 s end-to-end. If you forget everything else, hold these.", size=10.5)
pb(doc)


# Helper to assemble a slide entry
def slide(num, title, on_screen, what, deeper, why, qas, time, deeper2=None):
    slidehdr(doc, num, TOTAL, title)
    slide_img(doc, num)
    label(doc, "ON SCREEN"); body(doc, on_screen)
    label(doc, "WHAT TO SAY"); say(doc, what)
    label(doc, "DEEPER CONTEXT — the real understanding", PURPLE); deep(doc, deeper)
    if deeper2:
        deep(doc, deeper2)
    label(doc, "WHY IT'S HERE"); body(doc, why)
    if qas:
        label(doc, "LIKELY QUESTIONS")
        for q, a in qas:
            qa(doc, q, a)
    timing(doc, time)
    pb(doc)


# ============================================================
slide(1, "Title slide",
    "Thesis title, IIT (BHU) logo (top-right/centre), your name Aryan Gupta, roll 21064030, "
    "IDD Civil & Structural Engineering Year 5, supervisor Dr. Krishna Kant Pathak, department, institute.",
    "Good morning, respected committee. I am Aryan Gupta, IDD Civil and Structural Engineering, "
    "fifth year, roll number 21064030. My thesis is on an AI-powered framework for multi-objective "
    "steel truss optimization, integrating evolutionary algorithms, neural surrogates, reinforcement "
    "learning, and agent-assisted design, all under IS 800:2007. My supervisor is Dr. Krishna Kant "
    "Pathak. Let me walk you through it.",
    "‘Multi-objective’ means you optimize for more than one goal at once. Here the two goals are "
    "minimum WEIGHT and minimum DISPLACEMENT (sag). These fight each other — a lighter truss sags "
    "more. Most of the thesis is single-objective (weight only), and NSGA-II handles the genuinely "
    "multi-objective case by producing a Pareto front. ‘Agent-assisted design’ is your novel piece: "
    "an AI design agent (Claude under the hood) proposes engineering-reasonable starting designs "
    "that warm-start the optimizer.",
    "Formal opening. The long subtitle signals breadth: five techniques (GA/PSO/NSGA-II + surrogate "
    "+ RL + agent) unified under one Indian-code-compliant framework. Say it confidently in one breath.",
    [("What does multi-objective mean here?",
      "Two competing objectives — minimum weight and minimum maximum-displacement. There's no single "
      "best; you get a Pareto front of trade-offs, which NSGA-II computes."),
     ("Why all these techniques together?",
      "Each plays a distinct role: the optimizers search, the surrogate accelerates, the RL agent is "
      "an alternative learned solver, and the design agent warm-starts. The contribution is composing "
      "them into one validated, reproducible, code-compliant system.")],
    "20–25 seconds")

slide(2, "Outline",
    "Six parts listed: 1 Problem, 2 Framework, 3 Results, 4 Analysis, 5 Impact, 6 Conclusion.",
    "The talk has six parts. First the problem and why it matters. Then the framework I built. Then "
    "the results across four benchmarks. Then a deeper analysis. Then practical impact. And finally "
    "limitations, future work, and your questions.",
    "An outline slide is a navigation aid — it lets the committee know where you are at any moment, "
    "so when they ask a question they can place it. The six-part arc is a classic research-talk "
    "structure: motivate → build → prove → analyze → apply → conclude. Don't read every word; name "
    "the six parts and move on.",
    "Sets expectations and a mental map. Spending more than ~15 seconds here wastes your tight budget.",
    [("(Rarely questioned — it's an outline.)",
      "If asked how long: 'About 18 minutes of talk, leaving time for questions.'")],
    "15 seconds")

slide(3, "Part 1 divider — The Problem",
    "Large 'The problem'. Subtitle: steel-truss design in India carries 10–30% surplus material; can "
    "AI reduce that gap under code-compliant constraints?",
    "Let me start with the problem. Steel-truss design in India typically carries 10 to 30 percent "
    "surplus material. The question driving this thesis is whether modern AI can reduce that gap while "
    "staying fully code-compliant.",
    "A section divider exists to give the audience a breath and a signpost. The single most important "
    "word here is ‘code-compliant’ — it distinguishes your work from purely academic optimization that "
    "ignores real design codes. Plant that idea now; it pays off when you defend the 72-bar result.",
    "Pure signpost. One sentence, then move to the meat.",
    [],
    "15–20 seconds")

slide(4, "The design gap — motivation",
    "Bullets: India's truss construction volume; 10–30% surplus steel; steel embodied CO₂ ~1.9 kg/kg; "
    "construction ~11% of global emissions; existing optimizers too slow (~40 min/run). Motivation "
    "figure on the right.",
    "India builds thousands of industrial sheds, transmission towers, and overpasses each year under "
    "IS 800. Classical trial-and-check design typically carries 10 to 30 percent surplus steel versus "
    "the theoretical feasible optimum. Steel embodies about 1.9 kg of CO₂ per kg, and construction is "
    "around 11 percent of global emissions, so even a 15 percent cut across the country is a "
    "sector-level impact. Optimization tools exist, but a full GA run takes 30 to 40 minutes per "
    "benchmark — too slow for iterative design in a project meeting.",
    "Two distinct problems are being motivated here, and you should be able to separate them. PROBLEM "
    "ONE is waste: trial-and-check design over-sizes members because engineers add safety margins on "
    "top of code margins, leaving 10–30% surplus steel. PROBLEM TWO is speed: the research tools that "
    "could remove that waste are too slow to use during a live design session. Your thesis attacks "
    "both — the surrogate solves speed (C2), and the whole optimized pipeline solves waste. The CO₂ "
    "figure matters because steel is carbon-intensive: cutting tonnage cuts embodied carbon almost "
    "one-to-one.",
    "Establishes that the problem is real, large, economically and environmentally meaningful, and "
    "that current tools fall short on speed. This justifies the entire thesis.",
    [("Where does 10–30% come from?",
      "It's a widely-cited range in structural-optimization literature comparing trial-and-check "
      "designs against optimized minimum-weight designs; cited in Chapter 1."),
     ("Is 1.9 kg CO₂ per kg steel right?",
      "Yes — typical embodied-carbon for primary (blast-furnace) steel; recycled steel is lower. The "
      "order of magnitude is the point."),
     ("Why is 40 minutes a problem if you only design once?",
      "Real design is iterative — you change loads, spans, materials and re-run dozens of times. At 40 "
      "minutes a run that's an overnight job; the surrogate makes it interactive.")],
    "90 seconds")

slide(5, "The research question",
    "Large text: Can an AI design agent warm-start classical metaheuristics to converge faster on "
    "truss sizing, while staying IS 800-compliant? Tested on 10/25/72/200-bar.",
    "So the precise research question is this: can an AI design agent warm-start classical "
    "metaheuristics, so that they converge faster on truss sizing problems, while staying IS "
    "800-compliant? I measure this against four classical benchmarks — the 10-bar, 25-bar, 72-bar, "
    "and 200-bar trusses.",
    "A good research question is FALSIFIABLE — it can be proven wrong by data. Yours has three "
    "measurable parts: (1) ‘warm-start’ — does seeding the optimizer with agent designs help; "
    "(2) ‘converge faster’ — measured in generations-to-convergence, a hard number; (3) ‘IS "
    "800-compliant’ — every result must pass the code checks. Because all three are measurable, the "
    "whole thesis becomes evidence for or against a single sentence. The four benchmarks are the "
    "‘test set’ — they're the canonical sizing problems every paper reports, so you can compare "
    "directly with published numbers.",
    "The most important slide of Part 1 — it states the testable hypothesis. Everything afterwards is "
    "evidence. Say it slowly.",
    [("Why these four benchmarks?",
      "They are THE canonical truss-sizing benchmarks — 10-bar (2D, small), 25-bar (3D), 72-bar (3D "
      "tower), 200-bar (large 2D). Every major paper reports them, so I can validate against published "
      "optima directly."),
     ("Why ‘warm-start’ and not let the agent design the whole thing?",
      "LLMs are excellent at plausible initialization but poor at exact numerical optimization. So I "
      "use the agent only where it's strong — proposing good starting points — and let the proven "
      "optimizer do the precise search.")],
    "30 seconds")

slide(6, "Five research objectives (O1–O5)",
    "Table of O1–O5 each with a result pill: O1 reproduce optima within 2% (achieved <0.01%), O2 "
    "Pareto fronts ≥20 points (24–32), O3 ≥50× surrogate speedup at R²>0.98 (132×), O4 PPO within 5% "
    "(mixed, +16%), O5 agent warm-start cuts gens ≥20% (−36.8% on 10-bar).",
    "I set five quantitative objectives. O1: reproduce classical optima within two percent — achieved "
    "under 0.01 percent. O2: NSGA-II fronts with at least 20 points — achieved 24 to 32. O3: at least "
    "50 times surrogate speedup at R-squared above 0.98 — achieved 132 times. O4: PPO within five "
    "percent of the best classical solver — this one is mixed, plus 16 percent, and I'll explain why. "
    "O5: agent warm-start cuts generations by at least 20 percent — achieved minus 36.8 percent on "
    "10-bar. Four of five fully met; the fifth is mixed and I report it honestly.",
    "Notice the structure: each objective is a PRE-REGISTERED, falsifiable target with a pass/fail "
    "threshold set BEFORE seeing results. This is what separates rigorous engineering from "
    "storytelling — you committed to numbers and then measured against them. The one ‘mixed’ result "
    "(O4, PPO) is deliberately not hidden. Examiners trust a candidate who reports a partial failure "
    "far more than one who claims everything worked perfectly. When you reach O4, own it: RL is the "
    "right tool for repeated solving, not one-shot benchmarks.",
    "Shows you set measurable targets in advance and met four of five. The honest ‘mixed’ on O4 builds "
    "credibility for everything else.",
    [("Why did PPO (O4) not meet target?",
      "RL is data-hungry and the reward landscape is highly constrained. For one-shot benchmark "
      "solving, GA and PSO simply win. PPO pays off when you re-solve many similar problems — once "
      "trained it designs in one forward pass. That's future work F4."),
     ("Isn't a +16% miss a failure?",
      "For the stated objective, yes — and I report it as such. But it's a scientifically useful "
      "negative result: it maps where learned solvers do and don't beat classical search.")],
    "60 seconds")

slide(7, "Five contributions in one view (C1–C5)",
    "Five cards with big accent numbers: C1 0.004%, C2 132×, C3 549 lb, C4 −36.8%, C5 <10s. (This is "
    "the slide whose numbers were overlapping in the old build — your rebuild fixes it with the "
    "compact card layout.)",
    "Here are the five contributions I will defend. C1: end-to-end validation on the 10-bar to 0.004 "
    "percent of literature. C2: a 132 times surrogate speedup at R-squared 0.9993 on weight. C3: a "
    "rigorously feasible 72-bar baseline of 549 pounds — an observation, because it differs from the "
    "soft-penalty literature, pending an independent FEM cross-check. C4: agent warm-start cuts "
    "generations by 36.8 percent on 10-bar, p-value 0.0046. C5: the full IS 800 pipeline runs "
    "end-to-end in under 10 seconds on a laptop CPU, with no API key required.",
    "These five numbers ARE your thesis. Memorize them cold — examiners will quote them back and ask "
    "you to defend each. Know what each is measured against: C1 vs Sunar & Belegundu 1991; C2 is "
    "wall-clock for a full 500-generation GA run (12.4 s → 0.094 s) and R² on the weight head of the "
    "surrogate; C3 is your hard-constraint optimum vs the soft-penalty literature value; C4 is a "
    "Mann-Whitney/Wilcoxon test over seeds; C5 is the Streamlit demo runtime. Each maps to a later "
    "results slide, so this slide is the table of contents for Part 3.",
    "The thesis in five numbers and the spine for all of Part 3. Hit all five cleanly.",
    [("Which contribution is most novel?",
      "C4, the agent warm-start — to my knowledge the first rigorous A/B test of an AI-seeded GA on "
      "truss benchmarks with a significance test. C2 is the most practically valuable because it makes "
      "interactive use possible."),
     ("Are all five independent?",
      "Largely yes. C1 (validation), C2 (surrogate), C3 (hard-constraint baseline), and C5 "
      "(reproducible pipeline) stand even without the agent. C4 is the novel agent result layered on "
      "top.")],
    "60 seconds")

slide(8, "Part 2 divider — The Framework",
    "Large 'The framework'. Subtitle: an 8-layer stack: benchmarks → FEM → IS 800 → GA/PSO/NSGA-II → "
    "surrogate → PPO → agent → UI.",
    "Now, how I built it. The framework is an eight-layer stack — benchmarks, FEM, IS 800 checks, "
    "three optimizers, a neural surrogate, a PPO agent, the design agent, and a UI.",
    "Thinking of it as a layered stack helps the committee (and you) reason about dependencies: each "
    "layer only talks to the one below it. The FEM is the bedrock — everything ultimately calls it for "
    "ground truth. The surrogate is a faster stand-in for FEM. The optimizers sit on top and call "
    "either. The agent sits at the very top, seeding the optimizer. This clean separation is also why "
    "the system is reproducible and testable layer by layer.",
    "Section signpost that primes the architecture diagram on the next slide.",
    [],
    "15 seconds")

slide(9, "Problem formulation",
    "Optimization statement: minimize W(A)=Σ ρ L A, subject to stress (IS 800 6.2/6.3/7.1), "
    "displacement (5.6.1), slenderness ≤180 (3.8), area bounds. Four benchmark pills. 'Hard "
    "constraints only — no soft penalties.'",
    "Here is the formal problem. I minimize weight, which is a linear sum of areas weighted by density "
    "and length, subject to four constraint families: IS 800 stress checks from sections 6.2, 6.3, and "
    "7.1; displacement serviceability from 5.6.1; the slenderness limit of 180 from 3.8; and area side "
    "limits. Crucially, all constraints are HARD — there is no soft penalty. Infeasible designs are "
    "always dominated by any feasible design in the tournament.",
    "The objective W(A)=Σ ρ·Lᵢ·Aᵢ is LINEAR in the areas — that's why the surrogate predicts weight "
    "almost perfectly later (R²=0.9993). The difficulty is entirely in the constraints. ‘Hard vs "
    "soft’ is the single most important concept on this slide: a SOFT penalty adds a cost when you "
    "violate a limit, so a slightly-illegal design can still ‘win’ if it's light enough — its reported "
    "optimum may not actually be buildable. A HARD constraint tracks feasibility separately so a "
    "feasible design ALWAYS beats an infeasible one regardless of weight. Real code compliance is "
    "hard, not soft — which is exactly why your 72-bar number differs from the literature.",
    "Defines precisely what you optimize and under what rules. The ‘hard constraints’ line is the "
    "setup for the 72-bar observation — flag it now so it's not a surprise later.",
    [("Hard vs soft constraints — explain?",
      "Soft: violation adds a penalty to the objective, so a slightly-infeasible design can score "
      "well. Hard: feasibility is separate; any feasible design beats any infeasible one. I use hard — "
      "stricter, and it matches real code compliance."),
     ("Weight is linear — so why is it hard?",
      "The objective is linear, but the constraints (buckling, the feasibility boundary) are "
      "non-smooth and the feasible region is non-convex. That's what makes the search hard and rules "
      "out gradient methods."),
     ("Why slenderness ≤ 180?",
      "IS 800 §3.8 caps the slenderness ratio of compression members at 180 to prevent excessively "
      "slender, buckling-prone members — a serviceability/robustness limit independent of the stress "
      "check.")],
    "80 seconds")

slide(10, "System architecture",
    "Eight-layer block diagram: Benchmark registry → Classical optimiser (GA/PSO/NSGA-II) → IS 800 "
    "compliance; FEM engine → Neural surrogate MLP → PPO agent; design agent (Claude API + cache) → "
    "FastAPI backend → Streamlit UI.",
    "This single figure is the whole dataflow. The benchmark registry defines the truss. The FEM "
    "engine is the ground truth — areas in, physics out. The neural surrogate sits beside it as a fast "
    "approximation. The optimizers — GA, PSO, NSGA-II — call either FEM or the surrogate, and every "
    "design is checked against IS 800. The design agent seeds the initial population. The PPO agent "
    "has its own training loop. Everything is exposed through a FastAPI backend and a Streamlit UI.",
    "Trace one design through the diagram to truly own it: the optimizer proposes an area vector → it "
    "goes to FEM (or the surrogate) which returns weight, stress, displacement → the IS 800 module "
    "marks it feasible or not → the optimizer uses that to select survivors → repeat. The agent only "
    "touches generation zero (the initial population). The UI/API are thin wrappers so a non-Python "
    "user can drive the whole stack. Knowing this flow lets you answer almost any ‘how does X talk to "
    "Y’ question.",
    "Gives the committee the mental model for the rest of Part 2. Point to the flow; don't read every "
    "box.",
    [("Which layers did you build vs use a library?",
      "FEM, the IS 800 module, the benchmark encodings, the surrogate training pipeline, the agent "
      "prompt and cache, and the UI are mine. Optimizers are pymoo; the MLP and PPO use PyTorch and "
      "Stable-Baselines3 — standard libraries I configured and wired together."),
     ("Where is the bottleneck?",
      "The FEM solve inside the optimizer inner loop — 50,000 calls per run. That's exactly what the "
      "surrogate removes.")],
    "60 seconds")

slide(11, "Layer 2 & 3 — FEM kernel and IS 800 compliance",
    "Left: FEM steps (K = Σ Bᵀ k B assembly, partition method, solve K_ff u = F, axial-stress "
    "recovery) + truss-element figure. Right: IS 800 clauses (tension yield 6.2, rupture 6.3, buckling "
    "7.1, slenderness 3.8, serviceability 5.6.1) + checks figure.",
    "Layers 2 and 3. The FEM kernel is the standard direct stiffness method: assemble the global "
    "stiffness matrix, enforce supports by the partition method, solve K-f-f times u equals F using a "
    "dense solver, and recover axial stresses. It's validated to machine precision against the 3-bar "
    "canonical problem. On top of that, the IS 800 layer checks five clauses: tension yielding, "
    "tension rupture, compression buckling via Perry-Robertson curve a, the slenderness limit, and "
    "serviceability deflection.",
    "FEM in four steps you must be able to recite: (1) each bar gets a small stiffness matrix k=EA/L; "
    "(2) assemble all of them into the global K; (3) delete the rows/columns of supported joints "
    "(partition method) so the system is solvable; (4) solve K·u=F for displacements, then back out "
    "each bar's axial force and stress. ‘Machine precision’ means the error vs the analytical 3-bar "
    "answer is ~10⁻²¹ — essentially zero, so FEM is never the source of any discrepancy. On the code "
    "side, the key subtlety is BUCKLING: a compression member can fail below its yield stress by "
    "buckling sideways. Perry-Robertson (IS 800 curve a) reduces the allowable compressive strength as "
    "the member gets more slender, smoothly interpolating between full yield (short members) and Euler "
    "buckling (long members).",
    "Shows the engineering rigour beneath the AI. Examiners — civil engineers — care most about this "
    "slide. Know every formula on it.",
    [("Why the partition method, not a penalty for supports?",
      "Partition removes fixed DoFs entirely, giving a clean well-conditioned reduced system. Penalty "
      "adds a huge artificial stiffness which hurts numerical conditioning."),
     ("Why only buckling curve a?",
      "Curve a is for hot-rolled sections buckling about the strong axis — the simplest single-curve "
      "default. Selecting the correct curve per section type is a stated limitation."),
     ("Why dense solve, not sparse?",
      "For ≤200 bars the stiffness matrix is small; dense factorization beats sparse overhead. For "
      "10,000-bar real structures you'd switch to sparse."),
     ("What does Perry-Robertson actually compute?",
      "A reduction factor χ on the yield strength as a function of non-dimensional slenderness, "
      "accounting for initial imperfections — giving the design compressive strength f_cd.")],
    "90 seconds")

slide(12, "Layer 4 — three evolutionary baselines",
    "Three columns: GA (pop 100, gens 500, SBX, polynomial mutation), PSO (swarm 50, Clerc "
    "constriction), NSGA-II (pop 100, gens 300, non-dominated sort + crowding). Each with a flowchart.",
    "Layer 4 — three evolutionary baselines, all from pymoo. GA: population 100, 500 generations, SBX "
    "crossover, polynomial mutation. PSO: swarm of 50, the standard Clerc constriction parameters. "
    "NSGA-II: the multi-objective version, returning a Pareto front of weight versus displacement. GA "
    "and PSO are two single-objective baselines from different families — including both shows the "
    "framework isn't tied to one optimizer. All three use feasibility-aware tournaments — no soft "
    "penalties.",
    "These are population-based METAHEURISTICS — they keep a set of candidate designs and improve them "
    "iteratively without needing gradients. GA mimics evolution: select good parents, recombine "
    "(crossover), mutate, repeat. PSO mimics a swarm: each particle drifts toward its own best and the "
    "swarm's best. ‘Generation’ = one full cycle of score→select→breed; 100 designs × 500 generations "
    "= 50,000 evaluations per run. You run multiple random SEEDS because these are stochastic — one "
    "run might get lucky — so you report mean ± std over 5–10 seeds. Keep this slide BRIEF in the "
    "talk: these are textbook methods, not your contribution.",
    "Textbook methods the committee assumes you know. Keep it short — spend your time on the novel "
    "layers (surrogate, agent).",
    [("Why not gradient-based optimization?",
      "The constraints are non-smooth — the buckling curve has kinks and feasibility is a step "
      "function — so gradients are unreliable and the feasible region is non-convex. Metaheuristics "
      "only need to RANK designs, not differentiate."),
     ("Difference between GA and PSO in one line?",
      "GA evolves by mating and mutation — big stochastic jumps. PSO drifts particles toward personal "
      "and global best — smooth, smaller steps."),
     ("What is a generation, concretely?",
      "One cycle: score all 100 designs (one FEM/surrogate call each), select the best, recombine and "
      "mutate to make 100 children, repeat. 500 such cycles per run."),
     ("Why multiple seeds?",
      "The algorithms are random; a single run can be lucky or unlucky. 5–10 fixed seeds give mean and "
      "standard deviation, so the claims are statistically honest and reproducible.")],
    "70 seconds")

slide(13, "Layer 5 — neural surrogate",
    "MLP architecture [n_A → 256 → 128 → 64 → 3], ReLU, dropout 0.2; trained on 10,000 LHS samples, "
    "Adam, 200 epochs; outputs weight/stress/displacement; hybrid mode with FEM fallback. Architecture "
    "figure.",
    "Layer 5, the neural surrogate — and this is where the speedup comes from. A GA run needs about "
    "50,000 FEM evaluations, which is slow. So I trained a small MLP — 256, 128, 64 hidden units — on "
    "10,000 Latin-hypercube-sampled designs that FEM had already evaluated. The network learns to "
    "predict weight, stress, and displacement directly. In the GA inner loop it replaces FEM, bringing "
    "a full run from 12.4 seconds to 94 milliseconds — 132 times faster, R-squared 0.9993 on weight. "
    "Because stress and displacement are noisier, I run a hybrid mode: the surrogate screens, and "
    "borderline designs fall back to real FEM.",
    "The crucial thing to understand: the training data is SELF-GENERATED — there is no downloaded "
    "dataset. You pick 10,000 area vectors by Latin Hypercube Sampling (which spreads samples evenly "
    "across the design space, far better than uniform random), run your own FEM on each, and that "
    "input→output table IS the dataset. The MLP then learns the mapping ‘areas → weight, stress, "
    "displacement’. Weight is easy (R²=0.9993) because it's LINEAR in area. Stress/displacement are "
    "harder (R²≈0.81/0.87) because they depend on WHICH bar is critical, which can switch "
    "discontinuously as areas change. That's why the surrogate gives the objective (weight) directly "
    "but only SCREENS feasibility, deferring borderline cases to exact FEM — fast in the safe "
    "interior, exact at the constraint boundary.",
    "This is C2 — the practical contribution that makes interactive use possible. The key teaching "
    "point: FEM is both the ground truth AND the source of the training data.",
    [("Where did the training dataset come from?",
      "Self-generated. I draw 10,000 area vectors by Latin Hypercube Sampling and evaluate each with "
      "my own FEM. That table is the dataset — nothing is downloaded."),
     ("Why is weight R² 0.9993 but stress only 0.81?",
      "Weight is linear in area — trivial to learn. Stress depends on which bar is critical, which "
      "switches discontinuously, so it's harder. Hence stress is a screen, not the objective."),
     ("Why 10,000 samples?",
      "A sweep showed weight R² saturates above 10,000 — fewer hurts accuracy, more gives diminishing "
      "returns."),
     ("Why an MLP and not a Gaussian process?",
      "GPs are excellent below ~1,000 samples but scale cubically and break at 10,000. An MLP scales "
      "to large training sets and interpolates smoothly — the right tool here."),
     ("Is the 132× safe given approximate stress?",
      "Yes, via hybrid mode: weight (R²=0.9993) drives the objective; feasibility is screened and "
      "FEM-verified on borderline designs. Final answers are always FEM-checked.")],
    "85 seconds")

slide(14, "Layers 6 & 7 — RL agent and design-agent warm-start",
    "Left: PPO MDP (state = areas + max stress/disp; action = multiplicative adjustment [0.5,2.0]; "
    "reward = −W − λ·infeasibility) + MDP figure. Right: design agent (structured prompt → returns k=8 "
    "designs + reasoning; injected as seeds; cached) + pipeline figure.",
    "Layers 6 and 7. The PPO agent treats sizing as a Markov decision process — the state is the "
    "current area vector plus the worst stress and displacement, the action scales each area between "
    "half and double, and the reward is negative weight minus an infeasibility penalty. It's trained "
    "for 150,000 timesteps. On the right is the design agent — the novel piece. I send a structured "
    "prompt describing the geometry, loads, material, and IS 800 constraints, and the agent returns "
    "eight candidate designs with reasoning. These seed the GA's initial population. Every response is "
    "cached, so the pipeline rebuilds offline with no API key.",
    "Two very different uses of ‘AI’ here — keep them distinct. The PPO RL AGENT learns a POLICY by "
    "trial and error: given a truss state, what change makes it lighter while staying feasible? Like "
    "learning a video game through reward. The DESIGN AGENT is a language model that, prompted with "
    "the engineering problem, proposes sensible starting designs (thicker bars where forces are high). "
    "It does NOT optimize — it only seeds generation zero; the GA does all the real search. Caching "
    "every agent response by prompt-hash is what keeps the thesis reproducible and API-key-free: "
    "re-runs read the cache, not the live model.",
    "Introduces the two AI layers. Be crisp that the design agent is a SEED, not the optimizer. This "
    "sets up C4.",
    [("Is the design agent doing the optimization?",
      "No. It proposes eight starting designs for generation zero; the GA does all the optimization. "
      "It's a smart initializer, not an optimizer."),
     ("What model is the agent and is it fine-tuned?",
      "Zero-shot Claude via the public API at temperature 0.3 — not fine-tuned. I deliberately test "
      "the weakest version of the idea; fine-tuning is future work F5."),
     ("Why does PPO use multiplicative actions?",
      "Scaling each area by a factor in [0.5, 2.0] keeps moves proportional and bounded, which "
      "stabilizes learning better than additive changes across very different area magnitudes."),
     ("Doesn't an LLM make results non-reproducible?",
      "It would, but every response is cached by prompt hash and committed, so replays are "
      "deterministic and need no API key.")],
    "90 seconds")

slide(15, "Part 3 divider — The Results",
    "Large 'The results'. Subtitle: four benchmarks, five headline findings, one observation on the "
    "72-bar literature.",
    "Now the proof — four benchmarks, five headline findings, and one important observation on the "
    "72-bar literature.",
    "This is the heart of the talk and where most questions come from. The phrase ‘one observation’ "
    "is doing diplomatic work — it pre-frames the 72-bar discrepancy as a careful finding rather than "
    "a claim that prior papers are wrong. Setting that tone now defuses the hardest question later.",
    "Signpost into the evidence. Brief.",
    [],
    "15 seconds")

slide(16, "10-bar planar — end-to-end validation (C1)",
    "Convergence curve on the left; table on the right: literature 5060.85, PSO 5061.05, GA 5062.78, "
    "NSGA-II 5081.51. 'PSO error = 0.004%.'",
    "The 10-bar cantilever is the most-cited truss benchmark. The published optimum is 5060.85 pounds. "
    "With PSO over five seeds I reach 5061.05 pounds — an error of 0.004 percent. GA over ten seeds "
    "reaches 5062.78. This is the end-to-end signature that my FEM, IS 800, and optimizer stack are "
    "all working correctly together. This is contribution C1.",
    "Why this single number matters so much: matching the world's most-studied truss benchmark to "
    "0.004% simultaneously validates THREE things at once — your FEM is correct, your constraint "
    "handling is correct, and your optimizer converges to the true optimum. If any one were broken, "
    "you couldn't hit the published number. The convergence curve (weight vs generation) visually "
    "shows the optimizer descending and flattening at the literature line — the classic "
    "‘converged’ shape. PSO beats GA here because the 10-bar landscape is smooth and low-dimensional, "
    "which favours PSO's continuous drift.",
    "Validation against ground truth. Nailing the most-studied benchmark to 0.004% earns trust for "
    "every later number.",
    [("Why 5061.05 and not exactly 5060.85?",
      "A different optimizer and seed converge to within a fraction of a percent of the same global "
      "optimum. 0.004% is well inside the spread reported across 20+ papers."),
     ("Why does PSO beat GA here?",
      "The 10-bar feasible landscape is smooth and only 10-dimensional — PSO's continuous drift "
      "converges faster there; GA's advantage shows on rugged, higher-dimensional problems.")],
    "70 seconds")

slide(17, "25/72/200-bar — full benchmark matrix",
    "Table: 10-bar ~5061; 25-bar ~545 vs lit 545.16; 72-bar ~549 vs soft-penalty lit 379.62 (dagger); "
    "200-bar 315.89 vs lit 25445. Footnote on the dagger.",
    "The full matrix. 10-bar within 0.004 percent. 25-bar within 0.03 percent of literature. 72-bar "
    "converges to about 549 pounds — and the dagger marks the soft-penalty literature value of 380 "
    "pounds, which I discuss next. The 200-bar figures use a different problem variant with different "
    "loading, specified in Appendix C.",
    "This slide proves BREADTH: 2D and 3D, small (10 bars) to large (200 bars). The 25-bar match "
    "(0.03%) reinforces that the stack generalizes beyond the one benchmark. Two numbers will draw "
    "questions and you must not be flustered: the 72-bar (549 vs 380) and the 200-bar (315.89 vs "
    "25445). The 200-bar gap is simply a DIFFERENT PROBLEM VARIANT — the same ‘200-bar’ name covers "
    "several versions with different loads/units, and yours is the Lee & Geem 2004 variant documented "
    "in Appendix C. The 72-bar gap is the constraint-handling story (next slide).",
    "Demonstrates the framework handles the whole benchmark family. Sets up the 72-bar discussion; "
    "don't over-explain 200-bar units — just name the variant.",
    [("Why does 200-bar differ so much from 25,445?",
      "Different variant. The 25,445-lb value is a different 200-bar problem with separate loads and "
      "grouping; mine is the Lee & Geem 2004 reduced-load variant, documented in Appendix C."),
     ("Are the 25-bar and 10-bar matches independent confirmations?",
      "Yes — two different benchmarks, both within 0.03% and 0.004%, are independent evidence the "
      "stack is correct, not a fluke on one problem.")],
    "90 seconds")

slide(18, "The 72-bar hard-constraint baseline (C3)",
    "Hard-vs-soft comparison figure; text: ~549 lb under hard constraints vs ~380 lb soft-penalty "
    "literature; 'observation — future work F7'.",
    "This is the one result that needs careful explanation. Under my hard-constraint FEM, the 72-bar "
    "optimum converges to about 549 pounds, with cross-seed spread under one percent. I could not "
    "reproduce the soft-penalty values of around 380 pounds from Camp and Bichon 2004 and Bekdas 2015 "
    "as feasible. The gap most likely reflects a difference in constraint-handling convention, not an "
    "error in the published work. I flag it as observation C3 and propose an independent FEM "
    "cross-check using OpenSeesPy or SAP2000 as future work F7.",
    "Here is the careful logic to internalize so you stay calm: a soft-penalty optimizer can report a "
    "design that is SLIGHTLY infeasible because the small penalty is outweighed by the weight saving — "
    "so 380 lb may be a design that marginally violates a constraint. Your hard-constraint search only "
    "accepts STRICTLY feasible designs, landing at 549 lb. Neither is ‘wrong’ — they answer different "
    "questions. Your contribution is providing a RIGOROUSLY FEASIBLE baseline and quantifying how much "
    "the constraint convention changes the reported optimum (~45%). The diplomatic framing — "
    "‘observation, pending cross-check’ — is essential: never say the published papers are wrong.",
    "The single most likely target for a hostile question. Owning it as an honest observation, not a "
    "claim of error, is what protects you.",
    [("So are the published papers wrong?",
      "Not necessarily. They use soft penalties that tolerate small violations, so their optima can "
      "sit at or just past the constraint boundary. I use hard constraints. Both are valid — my point "
      "is to provide a rigorously feasible baseline, not to discredit prior work."),
     ("How do you know your 549 is correct?",
      "Cross-seed spread under 1% and every IS 800 constraint strictly satisfied under FEM. F7 "
      "proposes confirming it against a third-party solver (OpenSeesPy/SAP2000)."),
     ("Could it be a bug in your encoding?",
      "Possible in principle, which is exactly why I flag an independent cross-check rather than "
      "claiming a literature match. But the 10-bar and 25-bar match to <0.03%, so the core stack is "
      "validated.")],
    "90 seconds")

slide(19, "Surrogate parity and wall-clock speedup (C2)",
    "Left: parity plot (surrogate vs FEM, tight on y=x). Right: wall-clock bar (12.4 s → 0.094 s). "
    "Pills: weight R² 0.9993, stress 0.81, displacement 0.87.",
    "Contribution C2. On the left, the parity plot — surrogate prediction against FEM truth on a "
    "held-out test split, tight on the y-equals-x line, R-squared 0.9993 on weight. On the right, "
    "wall-clock — 12.4 seconds with FEM down to 94 milliseconds with the surrogate, a 132 times "
    "speedup. Stress and displacement heads are noisier, R-squared 0.81 and 0.87, so I use them as "
    "feasibility screens with FEM fallback on the constraint boundary.",
    "A parity plot puts predicted value on one axis and true value on the other; perfect prediction "
    "lies exactly on the 45° line. The tight cluster on y=x is the visual proof that the surrogate is "
    "essentially indistinguishable from FEM on weight. The 132× is END-TO-END wall-clock for a full "
    "500-generation run, not a per-call microbenchmark — that's the honest, conservative way to report "
    "it. The lower stress/displacement R² is not a weakness once you explain hybrid mode: weight "
    "(the objective) is near-perfect, and the noisier heads are only used to flag which designs need "
    "an exact FEM feasibility check.",
    "Visual proof of C2. The parity plot is your most convincing single image.",
    [("Doesn't lower stress R² make it unsafe?",
      "Only if used naively. Weight (the objective) is 0.9993; stress/displacement are screens, and "
      "borderline designs are re-checked with exact FEM. Final feasibility is always FEM-verified."),
     ("Is 132× cherry-picked?",
      "No — it's end-to-end wall-clock for a full GA run including constraint checks, the most "
      "conservative way to measure. Per-call speedup is even higher.")],
    "80 seconds")

slide(20, "Agent warm-start effect-map (C4)",
    "Bar chart of % change in generations; table: 10-bar −36.8% (p=0.0046), 25-bar −76.6% (p=0.25, "
    "n=3), 72-bar +4.8% (p=0.81). Note on redundant bar set {2,5,6,10}.",
    "Contribution C4, and the part I'm proudest of. On the 10-bar, agent warm-start cuts "
    "generations-to-convergence by 36.8 percent, p-value 0.0046 — statistically significant. On 25-bar "
    "the magnitude is even larger, minus 76.6 percent, but with only three seeds it's underpowered, "
    "p 0.25. On 72-bar, no detectable effect. The interpretation: the agent helps most when it can "
    "surface an architectural insight. On 10-bar, the agent correctly identifies that bars 2, 5, 6, "
    "and 10 are nearly redundant — a known result the GA would otherwise rediscover by trial and "
    "error. On 72-bar there's no analogous insight, so the effect vanishes.",
    "This is the novel scientific claim, and its strength is its HONESTY. The ‘effect-map’ shows the "
    "agent helping a lot on 10-bar, maybe on 25-bar (but too few seeds to be sure — p=0.25 means not "
    "significant), and not at all on 72-bar. The mechanism is the key insight: the agent encodes "
    "engineering ‘common sense’ from its training, so when the problem has a recognizable structural "
    "shortcut (10-bar's redundant members), it hands that to the GA immediately, saving the "
    "generations the GA would spend rediscovering it. When there's no such shortcut (72-bar), the "
    "agent adds nothing. The honest, qualified claim — ‘helps when insight exists’ — is far stronger "
    "than ‘always helps’, which would be false and easily attacked. p-value 0.0046 means there's only "
    "a 0.46% chance the 10-bar improvement is random luck.",
    "The headline novelty. The qualified framing is the defensible claim. Never overclaim.",
    [("Could the GA find the redundant bars without the agent?",
      "Yes, eventually. The agent doesn't discover anything unreachable — it provides the insight "
      "EARLIER, cutting the generations the GA spends rediscovering it. That earlier start is the "
      "measured speedup."),
     ("Is one significant benchmark enough?",
      "I claim it only for the regime where an architectural insight exists, shown significantly on "
      "10-bar. I explicitly do NOT claim universal benefit — 72-bar is the null case. That honesty is "
      "the contribution."),
     ("What does p = 0.0046 mean?",
      "Under a Mann-Whitney/Wilcoxon test, there's about a 0.46% probability the observed 10-bar "
      "improvement arose by chance — strong evidence it's real."),
     ("Why is 25-bar not significant despite −76.6%?",
      "Only three seeds per arm — too small a sample for significance (p=0.25), even though the effect "
      "size looks large. I report it as underpowered, not proven.")],
    "90 seconds")

slide(21, "NSGA-II Pareto front (O2)",
    "Pareto-front scatter in weight vs max-displacement space, 24–32 non-dominated points.",
    "Here is the 10-bar Pareto front — weight against maximum displacement. Each point is a "
    "non-dominated design; 24 to 32 per seed, satisfying objective O2 with margin. The "
    "single-objective optimum sits at the rightmost corner; lighter designs to the left all deflect "
    "more. This is the menu of trade-offs an engineer chooses from.",
    "‘Non-dominated’ means no other design is better in BOTH objectives at once — you can't reduce "
    "weight without increasing displacement, or vice versa. The set of all such designs is the Pareto "
    "front. NSGA-II finds it using two ideas: non-dominated SORTING (rank designs by how many "
    "dominate them) and CROWDING DISTANCE (prefer designs in sparse regions so the front is evenly "
    "spread, not clustered). The practical value: instead of one answer, the engineer sees the whole "
    "weight-vs-stiffness trade-off and picks the point that fits their budget and serviceability "
    "limit.",
    "Demonstrates multi-objective capability and answers O2 — the framework gives design choices, not "
    "just one number.",
    [("How do you pick one design from the front?",
      "An engineering decision based on budget and stiffness needs — often the ‘knee’ of the curve, or "
      "the lightest design meeting a code displacement limit. The framework presents the trade-off; "
      "the engineer chooses."),
     ("What is crowding distance?",
      "A tie-breaker that measures how isolated a design is in objective space; NSGA-II favours "
      "isolated points so the Pareto front stays evenly spread.")],
    "50 seconds")

slide(22, "Part 4 divider — Deeper Analysis",
    "Large 'Deeper analysis'. Subtitle: where the surrogate is trustworthy, how fast each optimizer "
    "converges, and what the agent actually said.",
    "Three deeper questions now: how trustworthy is the surrogate, how fast does each optimizer "
    "converge, and what does the agent actually say.",
    "Part 4 is where you show scientific maturity — not just ‘it works’, but ‘here's how reliable it "
    "is and why’. If you're short on time in the talk, this is the part to compress; the core claims "
    "live in Part 3.",
    "Signpost. Brief.",
    [],
    "15 seconds")

slide(23, "Surrogate uncertainty via MC-dropout",
    "Calibration figure; bullets: T=40 stochastic passes, calibration vs 0.95 target, weight head 0.96 "
    "(calibrated), displacement head 0.82 (under-confident, safe).",
    "Is a 132 times speedup safe to ship? I test it with MC-dropout — keeping dropout active at "
    "inference, running 40 stochastic forward passes, and using the variance as an uncertainty "
    "estimate. The calibration target is 95 percent of true values inside two sigma. The weight head "
    "measures 96 percent — well calibrated. The displacement head is 82 percent — under-confident, "
    "which is the safe direction. The recommendation: use the weight head as the objective and keep "
    "FEM for the final feasibility check.",
    "Normally dropout (randomly switching off neurons) is only used during TRAINING to prevent "
    "overfitting. MC-dropout keeps it ON at inference and runs the network many times — each pass "
    "gives a slightly different answer, and the SPREAD of those answers estimates how uncertain the "
    "model is. ‘Calibrated’ means the model's stated uncertainty matches reality: if it says ‘95% "
    "confident’, the truth really lands in that band 95% of the time. Your weight head at 96% is "
    "essentially perfectly calibrated. The displacement head at 82% is UNDER-confident — it claims "
    "more uncertainty than it has, which is the SAFE direction (it over-warns rather than "
    "under-warns).",
    "Shows you didn't trust the surrogate blindly — you quantified its reliability. The kind of rigour "
    "that impresses a technical committee.",
    [("What is MC-dropout in one sentence?",
      "Keep dropout active at inference and run many forward passes; the spread of predictions "
      "approximates the model's uncertainty (Gal & Ghahramani 2016)."),
     ("Why is under-confidence ‘safe’?",
      "It over-estimates uncertainty, so the system errs toward calling FEM more often than strictly "
      "necessary — it won't wrongly pass an unsafe design.")],
    "70 seconds")

slide(24, "Convergence-rate characterisation",
    "Exponential-fit figure W(t)=W∞ + A·e^(−t/τ); table: PSO τ 8.7, GA 23.5, agent+GA 14.9 generations.",
    "I fit an exponential to each seed's convergence curve, where tau is the time constant — how many "
    "generations to halve the remaining gap to the optimum. PSO converges fastest at 8.7 generations. "
    "GA is 23.5. GA with agent warm-start is 14.9 — shifted roughly halfway toward PSO. So the agent "
    "warm-start mechanism makes GA behave more like PSO in speed, which is exactly consistent with "
    "contribution C4.",
    "The time constant τ is a single number summarizing convergence SPEED: a smaller τ means the "
    "optimizer closes the gap to the optimum faster. Fitting the same exponential model to every run "
    "lets you compare speeds on equal footing. The headline insight: agent warm-start drags GA's τ "
    "from 23.5 down to 14.9 — about halfway to PSO's 8.7. This is the MECHANISM behind C4 made "
    "quantitative: the agent isn't changing where GA ends up, it's making GA get there faster by "
    "starting it closer.",
    "Explains the mechanism behind C4 quantitatively — not just ‘faster’ but ‘here's the time constant "
    "shifting’. Strong supporting evidence.",
    [("Why fit an exponential?",
      "Metaheuristic convergence toward an optimum is approximately geometric, so an exponential is "
      "the natural model and τ gives one comparable speed number per run."),
     ("Does warm-start change the final optimum or just the speed?",
      "Just the speed — both reach the same optimum; the agent reduces the generations needed to get "
      "there.")],
    "70 seconds")

slide(25, "Hard- vs soft-constraint feasibility frontier",
    "Pareto-front shift figure (no IS 800 vs full IS 800); caption on 549 vs 380 lb and "
    "constraint-handling sensitivity; F7.",
    "This is the analysis behind the 72-bar observation. The same problem under hard versus soft "
    "constraints produces very different fronts. Under hard IS 800, the feasible mass sits around 549 "
    "pounds; under a representative soft penalty it drops to around 380. The point is not that one is "
    "right and one wrong — it is that reported optima in the truss-sizing literature can be highly "
    "sensitive to the constraint-handling convention. For Indian practice, this means a soft-penalty "
    "optimum should not be deployed without a hard-feasibility cross-check.",
    "This slide turns the 72-bar ‘discrepancy’ into a genuine engineering CONTRIBUTION. By running the "
    "identical problem under both conventions on the same FEM, you ISOLATE the effect of constraint "
    "handling — about a 45% swing in reported weight. That's a cautionary result the whole field "
    "(and Indian designers using commercial soft-penalty solvers) should heed: a published light "
    "optimum may be marginally infeasible. You're not just reporting your number; you're explaining "
    "WHY the literature numbers differ and what that means for safe practice.",
    "Elevates the 72-bar gap from ‘my number disagrees’ to a real insight about the field. This is "
    "what makes the discrepancy a contribution rather than a weakness.",
    [("Is this constraint sensitivity already known?",
      "That constraint handling affects results is known, but a same-FEM, side-by-side hard-vs-soft "
      "ablation quantifying the ~45% magnitude on a standard benchmark is what I add."),
     ("What should a practicing engineer take from this?",
      "If using a soft-penalty commercial solver, run an independent hard-feasibility check before "
      "sign-off — the reported optimum may sit just past a code limit.")],
    "90 seconds")

slide(26, "Part 5 divider — Impact",
    "Large 'Impact'. Subtitle: reproducibility, civil-engineering practice, live demo.",
    "Now the practical impact — for Indian civil engineering, for reproducibility, and a live demo.",
    "Part 5 answers the examiner's unspoken ‘so what?’. You've proven the framework works; now show it "
    "matters in practice and that anyone can verify it.",
    "Signpost. Brief.",
    [],
    "10 seconds")

slide(27, "Practical implications for Indian civil engineering",
    "Three columns: interactive design (30–40 min → <20 s), constraint-handling sensitivity, "
    "zero-Python design aid via Streamlit.",
    "Three practical implications. First, the surrogate brings a 30-to-40-minute GA run down to under "
    "20 seconds, enabling parametric studies during a review meeting instead of overnight batches. "
    "Second, the hard-versus-soft sensitivity should be flagged to anyone using commercial "
    "soft-penalty solvers — run a hard-feasibility check before sign-off. Third, the Streamlit UI "
    "exposes the whole framework without requiring Python literacy — usable by a junior engineer, not "
    "just a researcher.",
    "These three implications map directly to your three strongest results: C2 (speed → interactive "
    "design), C3 (hard-vs-soft → safety awareness), and C5 (UI → accessibility). Tying impact back to "
    "specific contributions shows the work isn't academic for its own sake — each finding has a "
    "concrete user and use-case in the Indian construction context.",
    "Connects the technical work to real-world value — answers ‘so what’ concretely.",
    [("Who is the actual user?",
      "A practising structural engineer at a consulting firm or PSU (e.g., transmission-tower design) "
      "who needs fast, code-compliant sizing without writing code.")],
    "70 seconds")

slide(28, "Reproducibility contract",
    "Pinned versions, fixed seed set, 89 refs / 27 figures / 14 CSVs / 21 pickles checked in, cached "
    "agent responses, green CI, single-command PDF rebuild.",
    "Reproducibility is a first-class deliverable. Every dependency is pinned. Seeds are fixed and "
    "logged. All 89 references, 27 figures, 14 CSVs, and 21 optimization histories are checked in. "
    "Every agent response is cached, so the whole thing rebuilds offline with no API key. Continuous "
    "integration runs the fast test suite green on every push, and a single Tectonic command rebuilds "
    "both the 125-page thesis and this deck.",
    "Reproducibility means a stranger can re-run your work and get the SAME numbers. The threats to "
    "that are: unpinned library versions (results drift), unlogged random seeds (stochastic runs "
    "differ), and external API calls (the model changes or needs a key). You closed all three — pinned "
    "versions, fixed seeds, cached agent responses. This is increasingly what examiners and reviewers "
    "look for; it signals the work is trustworthy, not a one-off lucky run.",
    "Reproducibility is increasingly valued. This slide says ‘anyone can verify everything I claim’ — "
    "a strong credibility signal.",
    [("Doesn't using an agent break reproducibility?",
      "No — every response is cached by prompt hash and committed, so replays are deterministic and "
      "need no API key."),
     ("What does pinning versions achieve?",
      "It freezes the exact library versions (NumPy, pymoo, PyTorch, etc.) so numerical results don't "
      "drift when libraries update.")],
    "60 seconds")

slide(29, "Live demo surface",
    "Three interfaces: Streamlit UI (localhost:8501), FastAPI (localhost:8000 + Swagger), thesis PDF + "
    "deck.",
    "The same framework, three interfaces. The Streamlit UI lets a user pick a benchmark, an "
    "algorithm, a seed, and the agent warm-start toggle, then shows the live convergence plot, the IS "
    "800 report, and the cross-section chart. The FastAPI backend exposes the same stack "
    "programmatically, with Swagger docs. And the thesis and this deck both build from the same tagged "
    "repository. If the committee wishes, I can run a 10-bar optimization live right now.",
    "The demo flow to rehearse: pick 10-bar + GA, run WITHOUT the agent (note generations/time), then "
    "tick the agent warm-start and run AGAIN — the convergence curve drops faster and finishes in "
    "fewer generations. That live before/after IS contribution C4 happening in real time. CRITICAL: "
    "have the server already running before the viva and do one warm-up run; never start it live or "
    "debug in front of the committee. The user does NOT enter a custom truss — they pick from the four "
    "encoded benchmarks; a new benchmark is a short Python file (future-work UI).",
    "Offers the live demo. Have the server running beforehand. If anything fails, skip to slides — the "
    "convergence figures show the same thing.",
    [("Can a user enter their own custom truss?",
      "Not through this UI yet — the four benchmarks are encoded in the registry. A new benchmark is a "
      "short Python file of nodes, connectivity, loads, supports; the framework is modular. Custom-truss "
      "UI is future work."),
     ("What exactly does the user control?",
      "Benchmark, algorithm, seed, population, generations, and the agent warm-start toggle. Geometry, "
      "loads, and material come fixed from the published benchmark.")],
    "60 seconds (longer if running the live demo)")

slide(30, "Part 6 divider — Conclusion",
    "Large 'Conclusion'. Subtitle: limitations, future work, and a concrete ask.",
    "Finally — limitations, future work, and a summary.",
    "Closing the arc. The strongest conclusions state limitations openly THEN summarize strengths — it "
    "leaves the committee with confidence rather than doubt.",
    "Signpost. Brief.",
    [],
    "10 seconds")

slide(31, "What I am NOT claiming — limitations",
    "Bullets: linear-elastic small-displacement FEM; sizing-only not topology; NSGA-II 3 seeds; PPO "
    "single-instance no transfer; single buckling curve; agent zero-shot not fine-tuned.",
    "Let me be explicit about what I am NOT claiming. The FEM is linear elastic and small-displacement "
    "— no non-linearity or dynamics. The optimization is sizing only, not topology — connectivity is "
    "fixed. The NSGA-II 3D runs use three seeds, so O2's statistical power is weaker than O1's. The "
    "PPO agent is trained on one instance with no transfer. The IS 800 module uses a single buckling "
    "curve. And the design agent is zero-shot, not fine-tuned. None of these are load-bearing for "
    "contributions C1 through C5 — but they set the boundary of what I will defend.",
    "Stating limitations clearly is a CREDIBILITY MULTIPLIER, not a confession of weakness. Each limit "
    "is a deliberate SCOPE decision: linear-elastic FEM is the standard, correct choice for sizing "
    "(the design stays below yield by definition); sizing-only keeps the problem well-defined; "
    "zero-shot agent tests the weakest version of your idea so any positive result is conservative. "
    "The crucial sentence is ‘none of these are load-bearing for C1–C5’ — i.e., admitting them costs "
    "you nothing because your actual claims don't depend on them. Deliver this slide with confidence, "
    "never apology.",
    "Pre-empts hostile questions and shows scientific maturity. Do not apologize — these are scope "
    "decisions.",
    [("Why didn't you do topology optimization?",
      "Topology turns it into a mixed integer-continuous problem and is a distinct research area. "
      "Combining the agent warm-start with topology is a natural follow-on — future work F2."),
     ("Why linear-elastic FEM?",
      "For sizing against IS 800 working capacities the design stays below yield, so linear-elastic "
      "captures the relevant regime. Non-linear/dynamic FEM (for seismic) is future work F1."),
     ("Why zero-shot and not fine-tuned agent?",
      "To test the weakest version of the idea. If even un-fine-tuned the agent gives a 36.8% speedup, "
      "fine-tuning is upside, not a dependency. F5.")],
    "90 seconds")

slide(32, "Future work (F1–F7)",
    "F1 nonlinear/dynamic FEM → IS 1893; F2 topology via ground-structure/SIMP; F3 GNN surrogate; F4 "
    "cross-instance PPO transfer; F5 fine-tuned domain agent; F6 PGCIL transmission-tower validation; "
    "F7 independent FEM cross-check of 72-bar.",
    "Seven directions of future work. F1: non-linear and dynamic FEM for IS 1893 seismic. F2: topology "
    "optimization. F3: a graph-neural-network surrogate for variable topology. F4: cross-instance PPO "
    "transfer. F5: a fine-tuned domain-specific agent on IS 800 and IS 875. F6: real-world validation "
    "on a PGCIL transmission tower. And F7: an independent FEM cross-check of the 72-bar discrepancy "
    "using OpenSeesPy or SAP2000.",
    "A strong future-work slide shows the work OPENS doors rather than closing them, and that you "
    "understand the field's next steps. Notice F7 directly addresses the 72-bar question — so if "
    "pressed on that result, you can point here and say ‘and here is exactly how I'd resolve it’. F6 "
    "(PGCIL transmission tower) shows a credible path from benchmark to real Indian infrastructure.",
    "Shows the work is a foundation, not a dead end. F7 is your escape hatch for the 72-bar question.",
    [("Which future direction is most important?",
      "F2 (topology) for impact, and F7 (the 72-bar cross-check) for closure on the one open "
      "observation in this thesis.")],
    "70 seconds")

slide(33, "Summary",
    "C1–C5 restated; '125 pages, 27 figures, 89 references; 35-slide deck; laptop CPU; tag "
    "phase-10-complete'.",
    "To summarize. C1: 10-bar to literature at 0.004 percent — the full stack validated. C2: 132 times "
    "surrogate speedup at R-squared 0.9993 — interactive use is practical. C3: a rigorously feasible "
    "72-bar baseline at 549 pounds, with the gap versus soft-penalty literature flagged for an "
    "independent cross-check. C4: agent warm-start saves 36.8 percent of generations on 10-bar at "
    "p 0.0046, diminishing on harder 3D problems. C5: an IS 800-compliant design aid running under 10 "
    "seconds on a laptop CPU, with no API key needed to rebuild. The thesis is 125 pages, 27 figures, "
    "89 references; everything is tagged phase-10-complete.",
    "The summary is your last chance to plant the five numbers before questions begin — so hit all "
    "five C's cleanly and in order. Saying them again here means that even if a committee member "
    "drifted earlier, they leave with the headline results fresh. End on the reproducibility tag — it "
    "closes on strength.",
    "Reinforces the five contributions one final time. Deliver crisply.",
    [],
    "70 seconds")

slide(34, "Thank you / Questions",
    "'Questions?' with your name, roll, supervisor, repository, and demo URL.",
    "Thank you for your attention. I would be happy to take your questions.",
    "How you handle Q&A matters as much as the talk. Technique: when a question comes, REPEAT it back "
    "in your own words first — this buys thinking time and confirms you understood. If you don't know "
    "something, say so and offer your best reasoning (‘I don't have that measurement, but I'd expect X "
    "because Y’). Defend limitations as deliberate scope, not failures. The committee wants to see "
    "calm, honest command of your own work — not omniscience.",
    "Clean close. Pause, smile, sip water, wait for the first question.",
    [("(General Q&A technique)",
      "Repeat the question, answer in 30–60 seconds, stop. Don't ramble. If unsure, reason aloud "
      "honestly rather than guessing a number.")],
    "10 seconds, then Q&A")

# ---------- closing ----------
H1(doc, "Timing Map & Golden Rules")
label(doc, "TIME BUDGET (target 18–19 min, ~1–2 min buffer)")
for t, x in [("Intro","Slides 1–2: ~0:40"),("Part 1","Slides 3–7 (problem): ~3:00"),
             ("Part 2","Slides 8–14 (framework): ~5:00"),("Part 3","Slides 15–21 (results): ~5:00"),
             ("Part 4","Slides 22–25 (analysis): ~3:00"),("Part 5","Slides 26–29 (impact): ~2:00"),
             ("Part 6","Slides 30–34 (conclusion): ~2:30")]:
    bullet(doc, x, prefix=t)
label(doc, "GOLDEN RULES")
for r_ in [
    "Never compress Part 1 (the problem) — examiners anchor every later question to it.",
    "If running long, compress Part 4 (analysis) — it's supporting evidence, not core claims.",
    "Least time on slide 12 (three optimizers) — textbook plumbing.",
    "Most care on slides 7, 18, 20 (contributions, 72-bar, agent effect) — these draw questions.",
    "Have the demo server running BEFORE the viva. Never start it live.",
    "State limitations (slide 31) confidently — scope decisions, not failures.",
    "Memorize: 0.004% / 132× & R²0.9993 / 549 vs 380 / −36.8% p=0.0046 / <10 s.",
]:
    bullet(doc, r_)

p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(14); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("— Read once the night before, once the morning of. You're ready, Aryan. —")
r.font.name = "Calibri"; r.font.size = Pt(12); r.italic = True; r.font.color.rgb = MUTED

out = "/home/user/thesis/viva_prep/Slide_By_Slide_Explainer.docx"
doc.save(out)
print("Wrote", out, "size", os.path.getsize(out)//1024, "KB")
