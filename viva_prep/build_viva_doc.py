"""
Build a comprehensive viva-preparation Word document.
Output: /home/user/thesis/viva_prep/Viva_Complete_Guide.docx
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


# ---------- styling helpers ----------

INK = RGBColor(0x0B, 0x12, 0x20)
MUTED = RGBColor(0x5B, 0x6B, 0x85)
ACCENT = RGBColor(0x0A, 0x66, 0xFF)
ACCENT2 = RGBColor(0x18, 0xB9, 0x84)
WARN = RGBColor(0xE5, 0x70, 0x1F)


def set_cell_shading(cell, fill_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tc_pr.append(shd)


def add_horizontal_line(paragraph):
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "D6DCE6")
    p_bdr.append(bottom)
    p_pr.append(p_bdr)


def H1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(24)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(26)
    run.font.bold = True
    run.font.color.rgb = INK
    add_horizontal_line(p)
    return p


def H2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(18)
    run.font.bold = True
    run.font.color.rgb = ACCENT
    return p


def H3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = INK
    return p


def para(doc, text, italic=False, bold=False, size=11, color=INK, after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    run.italic = italic
    run.bold = bold
    run.font.color.rgb = color
    return p


def rich_para(doc, segments, size=11, after=6):
    """segments = list of (text, bold, italic, color)"""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    for text, bold, italic, color in segments:
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        run.font.color.rgb = color
    return p


def bullet(doc, text, level=0, bold_prefix=None, size=11):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.25 + 0.25 * level)
    p.paragraph_format.space_after = Pt(3)
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.font.name = "Calibri"
        r.font.size = Pt(size)
        r.bold = True
        r.font.color.rgb = INK
        r2 = p.add_run("  " + text)
        r2.font.name = "Calibri"
        r2.font.size = Pt(size)
        r2.font.color.rgb = INK
    else:
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(size)
        r.font.color.rgb = INK
    return p


def callout(doc, title, body, color_hex="EAF1FF", border_hex="0A66FF"):
    table = doc.add_table(rows=1, cols=1)
    table.autofit = True
    cell = table.cell(0, 0)
    set_cell_shading(cell, color_hex)
    cell.paragraphs[0].text = ""
    p = cell.paragraphs[0]
    r = p.add_run(title)
    r.font.name = "Calibri"
    r.font.size = Pt(11)
    r.bold = True
    r.font.color.rgb = RGBColor(0x0A, 0x66, 0xFF)
    p2 = cell.add_paragraph()
    r2 = p2.add_run(body)
    r2.font.name = "Calibri"
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = INK
    # add left border accent
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "24")
    left.set(qn("w:color"), border_hex)
    tcBorders.append(left)
    tcPr.append(tcBorders)
    doc.add_paragraph()


def code_block(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


def qna(doc, q, a):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("Q.  ")
    r.font.name = "Calibri"
    r.font.size = Pt(11)
    r.bold = True
    r.font.color.rgb = ACCENT
    r2 = p.add_run(q)
    r2.font.name = "Calibri"
    r2.font.size = Pt(11)
    r2.bold = True
    r2.font.color.rgb = INK

    p2 = doc.add_paragraph()
    p2.paragraph_format.space_after = Pt(10)
    p2.paragraph_format.left_indent = Inches(0.3)
    r3 = p2.add_run("A.  ")
    r3.font.name = "Calibri"
    r3.font.size = Pt(11)
    r3.bold = True
    r3.font.color.rgb = ACCENT2
    r4 = p2.add_run(a)
    r4.font.name = "Calibri"
    r4.font.size = Pt(11)
    r4.font.color.rgb = INK


def page_break(doc):
    doc.add_page_break()


# ============================================================
# BUILD THE DOCUMENT
# ============================================================

doc = Document()

# page margins
for section in doc.sections:
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)

# Default font
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

# ============================================================
# COVER
# ============================================================

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(60)
r = p.add_run("M.TECH / IDD THESIS VIVA — COMPLETE PREPARATION GUIDE")
r.font.name = "Calibri"
r.font.size = Pt(10)
r.font.color.rgb = MUTED
r.bold = True

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(8)
p.paragraph_format.space_after = Pt(6)
r = p.add_run("AI-Powered Framework for\nMulti-Objective Steel Truss Optimization")
r.font.name = "Calibri"
r.font.size = Pt(32)
r.bold = True
r.font.color.rgb = INK

p = doc.add_paragraph()
r = p.add_run(
    "From absolute basics to viva-ready confidence: every concept, "
    "every result, every likely examiner question — all in one place."
)
r.font.name = "Calibri"
r.font.size = Pt(13)
r.italic = True
r.font.color.rgb = MUTED

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(60)
add_horizontal_line(p)

p = doc.add_paragraph()
r = p.add_run("Aryan  •  Roll 21064030  •  IDD Civil (Y5)")
r.font.name = "Calibri"
r.font.size = Pt(13)
r.bold = True
r.font.color.rgb = INK

p = doc.add_paragraph()
r = p.add_run(
    "Supervisor: Dr. Krishna Kant Pathak\n"
    "Department of Civil Engineering\n"
    "Indian Institute of Technology (BHU) Varanasi"
)
r.font.name = "Calibri"
r.font.size = Pt(11)
r.font.color.rgb = MUTED

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(40)
r = p.add_run("HOW TO USE THIS DOCUMENT")
r.font.name = "Calibri"
r.font.size = Pt(10)
r.bold = True
r.font.color.rgb = ACCENT

para(
    doc,
    "Read it cover-to-cover at least once. Then re-read Sections 8–13 (the "
    "slide-by-slide walkthrough) the night before viva. Section 14 is a "
    "rapid-fire Q&A bank — practice answering aloud. Section 15 is the "
    "20-minute speaking script you can rehearse verbatim.",
    size=11,
)

page_break(doc)

# ============================================================
# TABLE OF CONTENTS
# ============================================================
H1(doc, "Table of Contents")

toc_items = [
    ("1.  The big picture: what is this thesis about?", "p. 3"),
    ("2.  The civil-engineering problem (start here)", "p. 5"),
    ("3.  What is FEM and why we need it", "p. 8"),
    ("4.  What is IS 800:2007 — the Indian steel code", "p. 12"),
    ("5.  Optimization, the simplest possible explanation", "p. 15"),
    ("6.  The five search methods (GA, PSO, NSGA-II, Surrogate, PPO)", "p. 18"),
    ("7.  The LLM warm-start — the novel contribution", "p. 30"),
    ("8.  Slide-by-slide walkthrough — Part 1: The Problem", "p. 34"),
    ("9.  Slide-by-slide walkthrough — Part 2: The Framework", "p. 40"),
    ("10. Slide-by-slide walkthrough — Part 3: The Results", "p. 48"),
    ("11. Slide-by-slide walkthrough — Part 4: Deeper Analysis", "p. 56"),
    ("12. Slide-by-slide walkthrough — Part 5: Impact", "p. 60"),
    ("13. Slide-by-slide walkthrough — Part 6: Conclusion", "p. 63"),
    ("14. Expected viva questions — full answer bank (50+ questions)", "p. 66"),
    ("15. The 20-minute viva script (rehearse this verbatim)", "p. 78"),
    ("16. Glossary of every technical term used", "p. 85"),
    ("17. The five contributions (C1–C5) — memorize these numbers", "p. 90"),
    ("18. Final checklist — the day of the viva", "p. 92"),
]

for label, pageref in toc_items:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(label)
    r.font.name = "Calibri"
    r.font.size = Pt(11)
    r.font.color.rgb = INK
    r2 = p.add_run("   " + pageref)
    r2.font.name = "Calibri"
    r2.font.size = Pt(10)
    r2.font.color.rgb = MUTED
    r2.italic = True

page_break(doc)

# ============================================================
# SECTION 1: THE BIG PICTURE
# ============================================================
H1(doc, "1. The Big Picture: What Is This Thesis About?")

para(
    doc,
    "In one sentence: you built an end-to-end software framework that takes "
    "a steel truss design problem, applies several AI techniques to find the "
    "lightest design that still satisfies the Indian steel code (IS 800:2007), "
    "and proves on four standard benchmarks that the AI techniques work as "
    "claimed — including a novel idea where a Large Language Model (Claude) "
    "warm-starts the optimization to converge faster.",
)

H2(doc, "1.1  Why does this matter?")
bullet(doc, "India builds thousands of trusses every year — industrial sheds, transmission towers, highway overpasses. Most are designed by trial-and-check methods that over-use steel by 10–30%.")
bullet(doc, "Steel is responsible for ~1.9 kg of CO₂ per kg produced, and construction is ~11% of global emissions. Even small percentage savings, scaled across the country, become massive in absolute terms.")
bullet(doc, "Optimization research already exists, but most published methods are too slow (40+ minutes per run) or use ‘soft penalty’ tricks that wouldn't survive a real code check.")
bullet(doc, "This thesis gives a fast (under 10 seconds with the surrogate), hard-constraint, IS 800-compliant, reproducible framework — and it tests whether an LLM can help it converge faster.")

H2(doc, "1.2  The five contributions in one place")

table = doc.add_table(rows=6, cols=3)
table.style = "Light Grid Accent 1"
table.autofit = False
table.columns[0].width = Inches(0.6)
table.columns[1].width = Inches(2.6)
table.columns[2].width = Inches(3.2)

hdr = table.rows[0].cells
for cell, t in zip(hdr, ["#", "Contribution", "Headline number"]):
    cell.text = ""
    p = cell.paragraphs[0]
    r = p.add_run(t)
    r.font.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = INK
    set_cell_shading(cell, "EAF1FF")

contributions = [
    ("C1", "End-to-end stack validation on 10-bar truss vs. literature", "Error = 0.004% (PSO best = 5061.05 lb vs. 5060.85 lb)"),
    ("C2", "Neural surrogate wall-clock speedup, weight head accuracy", "132× faster than FEM, R² = 0.9993"),
    ("C3", "Rigorously-feasible 72-bar baseline under hard IS 800 constraints", "~549 lb (vs. soft-penalty literature ~380 lb — flagged as observation)"),
    ("C4", "LLM warm-start reduces generations to convergence (10-bar)", "−36.8% generations, p-value = 0.0046"),
    ("C5", "Complete IS 800 pipeline on laptop CPU, no API key required", "<10 s end-to-end with surrogate"),
]
for i, (lab, contrib, num) in enumerate(contributions, start=1):
    row = table.rows[i].cells
    row[0].text = lab
    row[1].text = contrib
    row[2].text = num
    for c in row:
        for p in c.paragraphs:
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(10.5)

doc.add_paragraph()

H2(doc, "1.3  One-line description of every layer in the framework")
bullet(doc, "FEM solver — the ‘physics calculator’; given areas, returns forces, stresses, displacements. Slow but exact.", bold_prefix="Layer 1.")
bullet(doc, "Benchmark definitions — the four canonical test problems (10/25/72/200-bar).", bold_prefix="Layer 2.")
bullet(doc, "IS 800:2007 compliance — the code checks (tension, compression buckling, slenderness, serviceability).", bold_prefix="Layer 3.")
bullet(doc, "GA / PSO / NSGA-II — three evolutionary optimizers from pymoo.", bold_prefix="Layer 4.")
bullet(doc, "Neural surrogate — MLP that replaces FEM in the inner loop for 132× speedup.", bold_prefix="Layer 5.")
bullet(doc, "PPO reinforcement learning agent — learns a sizing policy by trial and error.", bold_prefix="Layer 6.")
bullet(doc, "LLM warm-start designer — Claude proposes initial designs to seed the GA.", bold_prefix="Layer 7.")
bullet(doc, "Streamlit UI + FastAPI — interactive interface for non-Python users.", bold_prefix="Layer 8.")

callout(
    doc,
    "REMEMBER",
    "If the examiner asks ‘what is the one-line thesis statement’ — say: "
    "‘An LLM-warm-started, surrogate-accelerated, IS 800-compliant evolutionary "
    "framework for steel truss sizing, validated on four classical benchmarks.’",
)

page_break(doc)

# ============================================================
# SECTION 2: THE CIVIL ENGINEERING PROBLEM
# ============================================================
H1(doc, "2. The Civil-Engineering Problem (Start Here)")

para(doc,
     "Before any AI, you must be able to describe — clearly and simply — the "
     "actual engineering problem you solved. Examiners will probe this first.")

H2(doc, "2.1  What is a truss?")
para(doc,
     "A truss is a structural framework made of straight bars (members) "
     "connected at joints (nodes). Loads are applied at the joints, and "
     "each bar carries only axial force — pure tension or pure compression. "
     "No bending, no shear (in the idealized model).")
para(doc, "Common examples:")
bullet(doc, "Roof trusses on industrial sheds and warehouses.")
bullet(doc, "Transmission towers (electricity pylons).")
bullet(doc, "Bridge trusses (Howe, Pratt, Warren types).")
bullet(doc, "Cranes and lifting frames.")

H2(doc, "2.2  The design decision")
para(doc,
     "In ‘sizing optimization’, the geometry (where the joints are, which "
     "bars connect to which) is FIXED by the benchmark. The only thing you "
     "choose is the cross-sectional area of each bar — call it A₁, A₂, …, Aₙ.")
para(doc, "So the design variable is a vector of n areas. For each benchmark:")
bullet(doc, "10-bar planar:  10 design variables (one per bar)")
bullet(doc, "25-bar spatial: 8 design variables (bars are grouped by symmetry)")
bullet(doc, "72-bar tower:   16 design variables (grouped)")
bullet(doc, "200-bar tower:  29 design variables (grouped)")

H2(doc, "2.3  The objective function")
para(doc, "Minimize total weight:")
code_block(doc, "W(A) = Σᵢ  ρ · Lᵢ · Aᵢ\n\nwhere  ρ = density of steel,  Lᵢ = length of bar i,  Aᵢ = area of bar i.")
para(doc,
     "Density and lengths are constants for a given benchmark. So weight is "
     "just a weighted sum of the design variables. Linear in A. Very simple "
     "objective.")

H2(doc, "2.4  The constraints (what makes it hard)")
para(doc,
     "If only weight mattered, the answer would be ‘set every area to zero’. "
     "The constraints are what give the problem meaning:")

H3(doc, "Stress constraint")
para(doc,
     "Each bar has an allowable stress σ_all (depends on whether it's "
     "tension or compression — compression also depends on buckling). The "
     "actual stress in the bar, σᵢ, must satisfy |σᵢ| ≤ σ_all.")

H3(doc, "Displacement constraint")
para(doc,
     "Each free joint can move under load. We limit each displacement δⱼ to "
     "a maximum (e.g., 2 inches for the classical 10-bar problem; L/325 for "
     "real-world serviceability in IS 800).")

H3(doc, "Slenderness constraint")
para(doc,
     "IS 800 Section 3.8 limits how slender a compression member can be: "
     "slenderness ratio λ = L/r ≤ 180 (where r is the radius of gyration). "
     "A bar that is too long and too thin will buckle.")

H3(doc, "Side constraints")
para(doc,
     "Aᵢ ∈ [A_min, A_max] — a physical minimum (you can't manufacture an "
     "infinitely thin bar) and a maximum (manufacturing/economic limit).")

H2(doc, "2.5  Hard vs. soft constraints — a key distinction")
para(doc,
     "If a design violates a constraint, what do you do? Two philosophies:")

bullet(doc, "Add a big penalty to the weight: W_effective = W + penalty × violation. The optimizer ‘sees’ a single number, but infeasible designs can mathematically dominate feasible ones if the penalty is too soft. Many published papers use this.", bold_prefix="Soft penalty:")
bullet(doc, "Track feasibility separately. In any tournament, a feasible design always beats an infeasible one — regardless of weight. This is what pymoo does by default and what we use.", bold_prefix="Hard constraint:")

para(doc,
     "This distinction matters a lot for the 72-bar result. Published "
     "papers report ~380 lb using soft penalties. Our hard-constraint search "
     "converges to ~549 lb. The difference is the constraint-handling "
     "convention, not a bug.",
     italic=True, color=MUTED)

callout(
    doc, "EXAMINER TIP",
    "If asked ‘what is the design variable?’ — the answer is "
    "‘a vector of n cross-sectional areas, one per bar (or per symmetry group).’ "
    "Not coordinates, not connectivity — those are fixed. ONLY areas."
)

page_break(doc)

# ============================================================
# SECTION 3: FEM
# ============================================================
H1(doc, "3. What Is FEM and Why We Need It")

para(doc,
     "The Finite Element Method is how you turn ‘a set of areas’ into ‘a "
     "set of stresses and displacements’. It is the physics calculator at "
     "the bottom of the entire framework — every optimizer eventually calls FEM.")

H2(doc, "3.1  The intuition")
para(doc,
     "Imagine pulling on a chain of springs. Each spring's stretch depends "
     "on how stiff it is and how much force passes through it. If you know "
     "every spring's stiffness, you can solve a system of equations to find "
     "every stretch.")
para(doc,
     "A truss is just a 2D or 3D version of this: each bar is a spring with "
     "axial stiffness  k = EA/L  (where E is Young's modulus, A is area, "
     "L is length). FEM is the systematic procedure to assemble all the "
     "individual bar stiffnesses into one big system  K·u = F  and solve "
     "for the displacement vector u.")

H2(doc, "3.2  The four steps (this is what to say in viva)")
H3(doc, "Step 1: Element stiffness")
para(doc,
     "For each bar, write a small (4×4 in 2D, 6×6 in 3D) stiffness matrix "
     "in local coordinates: kₑ = (EA/L) × [[1,−1],[−1,1]]. Then rotate it "
     "into the global coordinate system using the bar's direction cosines.")

H3(doc, "Step 2: Assembly")
para(doc,
     "Add every element's stiffness matrix into the global stiffness matrix "
     "K, placing each bar's contribution at the rows/columns corresponding "
     "to its end nodes. Mathematically: K = Σₑ Bₑᵀ kₑ Bₑ, where Bₑ is a "
     "connectivity matrix.")

H3(doc, "Step 3: Apply boundary conditions")
para(doc,
     "Some joints are pinned (cannot move). For these, the corresponding "
     "rows and columns of K are removed (partition method). What's left is "
     "K_ff — the stiffness matrix of free degrees of freedom.")

H3(doc, "Step 4: Solve")
para(doc,
     "Solve K_ff · u_f = F_f for the free displacements u_f using "
     "numpy.linalg.solve. Then back-substitute to get reactions at supports "
     "and stress in each bar: σₑ = (E/L) × [−c, −s, c, s] · uₑ.")

H2(doc, "3.3  Why dense numpy, not sparse?")
para(doc,
     "For our benchmarks (≤ 200 bars), the stiffness matrix is small "
     "(hundreds of DoF). Dense factorization is faster than sparse "
     "overhead. For real-world 10,000-bar problems, you'd switch to sparse "
     "(scipy.sparse + spsolve).")

H2(doc, "3.4  Sign convention (memorize)")
bullet(doc, "Positive axial force = tension.")
bullet(doc, "Negative axial force = compression.")
bullet(doc, "Constraint is written as g(x) ≤ 0  (matches IS 800 LSD format).")

H2(doc, "3.5  Validation — how do we know FEM is correct?")
para(doc,
     "We validated on the 3-bar canonical problem (textbook closed-form "
     "solution). Computed node-3 vertical displacement = −2.928932×10⁻⁵ m. "
     "Expected from analytical solution = −2.928932×10⁻⁵ m. Error = 6.78×10⁻²¹ "
     "(machine precision). Equilibrium residual ‖Ku − F‖ = 2.27×10⁻¹³.")
para(doc, "21 pytest cases pass on the FEM module. FEM is not the source of any error in the thesis.",
     italic=True, color=MUTED)

H2(doc, "3.6  Limitations of our FEM kernel")
bullet(doc, "Linear elastic only — no plasticity, no yielding mid-analysis.")
bullet(doc, "Small displacement — no geometric non-linearity.")
bullet(doc, "Static — no dynamics, no seismic response, no time history.")
bullet(doc, "No post-buckling analysis — we check buckling capacity per IS 800 but don't simulate the post-buckling regime.")

callout(
    doc, "VIVA STORY",
    "‘Our FEM is the simplest competent choice for sizing optimization on "
    "small-to-medium trusses. It is linear elastic, small-displacement, "
    "validated to machine precision against the 3-bar analytical solution.’"
)

page_break(doc)

# ============================================================
# SECTION 4: IS 800:2007
# ============================================================
H1(doc, "4. What Is IS 800:2007 — The Indian Steel Code")

para(doc,
     "IS 800:2007 is the Bureau of Indian Standards code for general "
     "construction in steel. Any steel structure in India must comply. "
     "It is a Limit State Design (LSD) code — same philosophy as Eurocode 3 "
     "and the latest AISC.")

H2(doc, "4.1  The five clauses we implement")

table = doc.add_table(rows=6, cols=3)
table.style = "Light Grid Accent 1"
hdr = table.rows[0].cells
for cell, t in zip(hdr, ["Clause", "What it checks", "Formula"]):
    cell.text = ""
    p = cell.paragraphs[0]
    r = p.add_run(t)
    r.font.bold = True
    r.font.size = Pt(11)
    set_cell_shading(cell, "EAF1FF")

rows = [
    ("§6.2", "Tension yielding of gross section", "T_dg = A_g · f_y / γ_m0"),
    ("§6.3", "Tension rupture of net section", "T_dn = 0.9 · A_n · f_u / γ_m1"),
    ("§7.1", "Compression buckling (Perry-Robertson, curve a)", "P_d = A_g · f_cd"),
    ("§3.8", "Slenderness limit for compression members", "λ = KL/r ≤ 180"),
    ("§5.6.1", "Serviceability (deflection limit)", "δ ≤ L/325"),
]
for i, (cl, ck, fm) in enumerate(rows, start=1):
    rc = table.rows[i].cells
    rc[0].text = cl
    rc[1].text = ck
    rc[2].text = fm
    for c in rc:
        for p in c.paragraphs:
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(10.5)

doc.add_paragraph()

H2(doc, "4.2  The partial safety factors")
bullet(doc, "γ_m0 = 1.10 — yielding (gross section)")
bullet(doc, "γ_m1 = 1.25 — rupture (net section, less ductile mode)")
bullet(doc, "f_y = yield strength (typical: 250 MPa for E250 steel, 350 MPa for E350)")
bullet(doc, "f_u = ultimate strength (typical: 410 MPa for E250)")

H2(doc, "4.3  The buckling formula (Perry-Robertson) explained simply")
para(doc,
     "A column under compression doesn't simply yield — long thin ones "
     "buckle sideways before reaching yield. The Perry-Robertson formula "
     "(IS 800 uses curve ‘a’ for hot-rolled sections) accounts for "
     "imperfections and reduces the design strength as slenderness "
     "increases:")
code_block(doc,
           "f_cd = (f_y / γ_m0)  ×  χ\n"
           "χ = 1 / (φ + √(φ² − λ̄²))      ≤ 1\n"
           "φ = 0.5 [1 + α(λ̄ − 0.2) + λ̄²]\n"
           "λ̄ = non-dimensional slenderness = √(f_y / f_cc)\n"
           "f_cc = π²E / λ² (Euler stress)\n"
           "α = 0.21 for curve a")
para(doc,
     "Short columns: χ ≈ 1 (no reduction). Long columns: χ < 1 (strength "
     "reduced). At λ̄ → ∞, χ → 0 (Euler buckling limit).",
     italic=True, color=MUTED)

H2(doc, "4.4  Why ONLY curve ‘a’?")
para(doc,
     "IS 800 has four buckling curves (a, b, c, d) depending on cross-"
     "section type, residual stress, axis of buckling. We use curve ‘a’ "
     "throughout — the most favourable. This is a limitation: a fully "
     "general implementation would select the correct curve per "
     "section type. We note this in the limitations slide.")

H2(doc, "4.5  Hard-constraint enforcement")
para(doc,
     "Every IS 800 check is a hard constraint: if the design fails any "
     "clause, it is marked infeasible. The optimizer's tournament rule "
     "(in pymoo) guarantees that any feasible design beats any "
     "infeasible design — independent of weight. This is the cleanest, "
     "most defensible constraint handling.")

callout(
    doc, "WHY IS 800 MATTERS FOR YOUR STORY",
    "Most academic optimization papers use a soft-penalty approach because "
    "it's easier to code. A real engineer cannot deliver a soft-penalty "
    "design — the code is a hard requirement. By enforcing IS 800 as a hard "
    "constraint, you bridge the gap between academic optimization and "
    "Indian practice. That is a substantive contribution."
)

page_break(doc)

# ============================================================
# SECTION 5: OPTIMIZATION BASICS
# ============================================================
H1(doc, "5. Optimization, The Simplest Possible Explanation")

para(doc,
     "‘Optimization’ sounds intimidating but in our context it just means: "
     "find the set of 10 numbers (or 8, 16, 29) that gives the lowest "
     "weight while satisfying every IS 800 check.")

H2(doc, "5.1  Why can't you just try every combination?")
para(doc,
     "Each area can be anything from 0.1 to 30 in² (continuous). If you "
     "discretize at 0.1 in² steps, that's 300 options per bar. For 10 bars: "
     "300¹⁰ = 5.9 × 10²⁴ combinations. At 1 ms per FEM evaluation, that's "
     "1.87 × 10¹⁴ years. The universe is 1.4 × 10¹⁰ years old. Brute force "
     "is not an option.")

H2(doc, "5.2  Why not just use calculus / gradient descent?")
para(doc,
     "Gradient methods need a smooth, differentiable objective. Our "
     "objective is smooth, BUT the constraints (IS 800 buckling, "
     "slenderness, feasibility flag) are non-smooth, discontinuous, and "
     "the buckling curve has kinks. Also, the feasible region is "
     "non-convex — gradient methods will get stuck in local minima.")
para(doc,
     "Metaheuristics (GA, PSO) don't need gradients. They only need to "
     "RANK designs. That's why they dominate truss-sizing literature.")

H2(doc, "5.3  Single-objective vs. multi-objective")
para(doc,
     "Single-objective (GA, PSO): one number to minimize (weight). Output: "
     "one best design.")
para(doc,
     "Multi-objective (NSGA-II): multiple competing objectives (weight AND "
     "max displacement). Output: a Pareto front — a set of trade-off "
     "designs, none dominated by any other.")

H2(doc, "5.4  Evaluating an algorithm — what numbers do we report?")

bullet(doc, "Best fitness across all seeds — does the algorithm find the optimum?")
bullet(doc, "Mean fitness across seeds — is it consistent?")
bullet(doc, "Standard deviation — does it have low variance across runs?")
bullet(doc, "Generations to convergence — how fast does it reach the optimum?")
bullet(doc, "Wall-clock time — practical speed.")
bullet(doc, "Number of FEM calls — the dominant cost (FEM is expensive).")

H2(doc, "5.5  The seeds question (examiners love this)")
para(doc,
     "Metaheuristics are stochastic. A single run can luck out (or be "
     "unlucky). To make claims, you run each algorithm with multiple "
     "random seeds and report mean ± std. We use 5–10 seeds for the "
     "single-objective runs and 3 for NSGA-II (multi-objective is more "
     "expensive).")
para(doc,
     "Seed set: {42, 123, 456, 789, 2026, 7, 13, 91, 314, 271}. Fixed per "
     "algorithm. Logged per run. Anyone can reproduce.",
     italic=True, color=MUTED)

page_break(doc)

# ============================================================
# SECTION 6: THE FIVE SEARCH METHODS
# ============================================================
H1(doc, "6. The Five Search Methods")

para(doc,
     "These are the algorithms inside your framework. Master these five. "
     "If you can explain all five clearly, you have already passed.")

# --- 6.1 GA ---
H2(doc, "6.1  Genetic Algorithm (GA)")

H3(doc, "The 30-second pitch")
para(doc,
     "Mimic Darwinian evolution. Maintain a population of candidate designs. "
     "Score them. Pick winners. Mate winners to make children. Mutate. "
     "Repeat for 500 generations.")

H3(doc, "The six steps")
bullet(doc, "Generate N=100 random area vectors. Evaluate each via FEM/surrogate. Compute fitness (= weight if feasible, else worst).", bold_prefix="Initialize.")
bullet(doc, "Tournament — pick 3 random designs from the population, keep the best. Do this twice to get two parents.", bold_prefix="Select.")
bullet(doc, "Simulated Binary Crossover (SBX) with η_c = 15. Two parents → two children, each child is a stochastic blend of the parents.", bold_prefix="Crossover.")
bullet(doc, "Polynomial mutation with η_m = 20, probability 1/n. Slightly perturbs each gene with low probability.", bold_prefix="Mutate.")
bullet(doc, "Replace the old population with the new children. Elitism — keep the single best design from the old population.", bold_prefix="Replace.")
bullet(doc, "Stop after 500 generations OR when fitness improvement < 10⁻⁶ for 50 consecutive generations.", bold_prefix="Stop.")

H3(doc, "Hyperparameters in our implementation")
code_block(doc,
           "Population size  = 100\n"
           "Generations      = 500\n"
           "Crossover (SBX)  η_c = 15\n"
           "Mutation (poly)  η_m = 20\n"
           "Mutation prob    = 1/n  (one gene per child on average)\n"
           "Library          = pymoo 0.6.1.6\n"
           "Constraint mode  = feasibility-aware tournament")

H3(doc, "What SBX actually does (in case asked)")
para(doc,
     "Simulated Binary Crossover. Given parents p1, p2 and a random number "
     "u ∈ [0,1], compute β = (2u)^(1/(η_c+1)) if u ≤ 0.5, else "
     "β = (1/(2(1−u)))^(1/(η_c+1)). Then children = 0.5[(1+β)p1 + (1−β)p2] "
     "and 0.5[(1−β)p1 + (1+β)p2]. The η_c parameter controls spread: "
     "larger η_c = children closer to parents, smaller η_c = more "
     "exploration.")

# --- 6.2 PSO ---
H2(doc, "6.2  Particle Swarm Optimization (PSO)")

H3(doc, "The 30-second pitch")
para(doc,
     "A swarm of particles flies through the design space. Each particle "
     "remembers its personal best location and is pulled toward both that "
     "and the swarm's global best.")

H3(doc, "The update equations")
code_block(doc,
           "vᵢ ← ω·vᵢ  +  c₁·r₁·(pᵢ − xᵢ)  +  c₂·r₂·(g − xᵢ)\n"
           "xᵢ ← xᵢ + vᵢ\n\n"
           "ω  = inertia (0.729 in our runs)\n"
           "c₁ = cognitive coefficient (1.494 — pull toward personal best)\n"
           "c₂ = social coefficient (1.494 — pull toward global best)\n"
           "r₁, r₂ = random ∈ [0,1] per dimension\n"
           "pᵢ = particle i's personal-best position\n"
           "g  = swarm's global-best position")

H3(doc, "Why these specific constants?")
para(doc,
     "ω = 0.729, c₁ = c₂ = 1.494 is the ‘Clerc constriction’ setting "
     "(Clerc & Kennedy 2002). It is the most-cited standard parameter set "
     "and gives stable convergence without divergence. We did not tune.")

H3(doc, "Velocity clipping")
para(doc,
     "If velocity grows too large, particles ‘fly out’ of the design "
     "space. We clip velocity at ±20% of the area range per step. This is "
     "standard practice.")

H3(doc, "GA vs. PSO — when does each win?")
bullet(doc, "PSO converges faster on smooth, low-dimensional problems (10-bar — best is PSO at 5061.05 lb).")
bullet(doc, "GA is more robust on rugged, high-dimensional, multimodal problems.")
bullet(doc, "PSO can prematurely converge if ω is too low — it lacks ‘mutation’ to escape local optima.")

# --- 6.3 NSGA-II ---
H2(doc, "6.3  NSGA-II (Multi-Objective)")

H3(doc, "Why need a different algorithm?")
para(doc,
     "If you want to minimize BOTH weight AND max displacement, there is "
     "no single optimum. There is a curve of trade-offs (the Pareto "
     "front). GA returns one design. NSGA-II returns the whole curve.")

H3(doc, "The Pareto front, formally")
para(doc,
     "Design A DOMINATES design B if A is at least as good in every "
     "objective AND strictly better in at least one. The Pareto front is "
     "the set of designs not dominated by anything in the population.")

H3(doc, "Non-dominated sorting (NSGA-II's first trick)")
para(doc,
     "Sort the population into ‘fronts’:")
bullet(doc, "Front 1 = designs not dominated by anyone (current Pareto front).")
bullet(doc, "Front 2 = designs dominated only by Front 1.")
bullet(doc, "Front 3 = dominated by Fronts 1 and 2. And so on.")
para(doc,
     "When selecting parents for the next generation, prefer designs from "
     "earlier fronts.")

H3(doc, "Crowding distance (NSGA-II's second trick)")
para(doc,
     "Within a front, prefer designs in less-crowded regions of objective "
     "space. For each objective, sort and take the gap between neighbours. "
     "Sum the gaps across objectives = crowding distance. Larger = less "
     "crowded = preferred. This spreads the final front evenly.")

H3(doc, "Our NSGA-II setup")
code_block(doc,
           "Population size  = 100\n"
           "Generations      = 300\n"
           "Objectives       = (weight, max_displacement)\n"
           "Output           = Pareto front of 24–32 non-dominated designs\n"
           "Seeds            = 3 per benchmark (multi-objective is expensive)")

# --- 6.4 Neural Surrogate ---
H2(doc, "6.4  Neural Surrogate (MLP)")

H3(doc, "The problem it solves")
para(doc,
     "GA needs 100 individuals × 500 generations = 50,000 FEM calls per run. "
     "FEM takes ~1 ms per call on 10-bar. A neural surrogate that mimics "
     "FEM can do the same in ~0.01 ms — a 100× speedup. Wall-clock: 12.4 s "
     "→ 0.094 s.")

H3(doc, "Architecture")
code_block(doc,
           "Input  layer:  n_A neurons  (one per area)\n"
           "Hidden:        256 → 128 → 64  (ReLU activation, dropout p=0.2)\n"
           "Output layer:  3 neurons  (weight, max_stress, max_displacement)\n"
           "Loss:          MSE (mean squared error)\n"
           "Optimizer:     Adam, learning rate = 1e-3\n"
           "Epochs:        200\n"
           "Batch size:    128\n"
           "Library:       PyTorch 2.3.1")

H3(doc, "Training data — Latin Hypercube Sampling (LHS)")
para(doc,
     "We sample 10,000 area vectors using LHS. LHS partitions each "
     "dimension into 10,000 equal bins and ensures every bin is occupied "
     "exactly once — much better space-filling than uniform random "
     "sampling. Each sample is then evaluated with real FEM to create "
     "(input, output) training pairs.")

H3(doc, "Accuracy report")
bullet(doc, "Weight head: R² = 0.9993 (extremely accurate — weight is linear in area, easy to learn).")
bullet(doc, "Max stress head: R² = 0.81 (harder — non-linear, sensitive to which bar is critical).")
bullet(doc, "Max displacement head: R² = 0.87 (medium difficulty).")

H3(doc, "Hybrid mode (the trick we use)")
para(doc,
     "Surrogate is used as a first-pass evaluator inside the GA. But "
     "because stress/displacement heads are less accurate, we use them as "
     "FEASIBILITY SCREENS only. Any design that the surrogate flags as "
     "borderline (close to constraint boundary) is re-evaluated with real "
     "FEM. This gives full surrogate speed in the safe interior while "
     "keeping FEM accuracy on the constraint boundary.")

H3(doc, "MC-dropout uncertainty (Bayesian touch)")
para(doc,
     "We turn dropout ON during inference too. Run T=40 forward passes per "
     "input. The variance of predictions estimates the surrogate's "
     "uncertainty. Calibration test: how often does the true value fall "
     "inside μ ± 2σ? Target = 95%. Weight head: 96% (well-calibrated). "
     "Displacement head: 82% (under-confident, conservatively safe).")

# --- 6.5 PPO ---
H2(doc, "6.5  PPO Reinforcement Learning")

H3(doc, "The reframing")
para(doc,
     "Instead of SEARCHING the design space, train an AGENT that knows "
     "how to MODIFY any design to make it lighter and feasible.")

H3(doc, "The Markov Decision Process (MDP)")
bullet(doc, "Current area vector A ∈ ℝⁿ + current max stress σ_max + current max displacement δ_max.", bold_prefix="State sₜ:")
bullet(doc, "Multiplicative adjustment vector ∈ [0.5, 2.0]ⁿ — i.e., scale each area by a factor between 0.5× and 2×.", bold_prefix="Action aₜ:")
bullet(doc, "rₜ = −W(A_new) − λ · infeasibility_penalty. Lower weight → higher reward. Infeasibility → big negative reward.", bold_prefix="Reward rₜ:")
bullet(doc, "50 steps per episode. New random initial design at each reset.", bold_prefix="Episode:")

H3(doc, "What PPO actually does")
para(doc,
     "Proximal Policy Optimization (Schulman et al. 2017). Trains a neural "
     "network policy π_θ(a|s) that outputs an action distribution given a "
     "state. Updates policy by maximising expected reward, with a clipped "
     "surrogate objective:")
code_block(doc,
           "L^CLIP = E[ min( rₜ(θ)·Â_t,  clip(rₜ(θ), 1−ε, 1+ε)·Â_t ) ]\n\n"
           "rₜ(θ) = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)\n"
           "Â_t   = estimated advantage at step t\n"
           "ε     = 0.2 (clip range)")

H3(doc, "Why the clip?")
para(doc,
     "Without clipping, a single very-good batch could push the policy "
     "WAY off and the agent forgets prior learning (high variance, "
     "catastrophic updates). The clip caps the per-step change — hence "
     "‘proximal’. This is the single trick that makes PPO the default RL "
     "algorithm today.")

H3(doc, "Our PPO results — be honest")
para(doc,
     "PPO converges but is +16% off the best classical solver on 10-bar. "
     "We report this as a MIXED result (O4 in objectives). RL pays off "
     "when you'll re-solve many similar problems; for one-shot benchmarks, "
     "GA/PSO win. Stating this honestly is a strength, not a weakness.")

H3(doc, "Library")
code_block(doc,
           "Stable-Baselines3 PPO\n"
           "Policy network: MLP, two hidden layers, 64 units each\n"
           "Gymnasium env: custom TrussEnv wrapping our FEM evaluator\n"
           "Training: 150,000 timesteps")

page_break(doc)

# ============================================================
# SECTION 7: LLM WARM-START
# ============================================================
H1(doc, "7. The LLM Warm-Start — The Novel Contribution")

para(doc,
     "This is the most original part of the thesis. Examiners will want "
     "to know exactly what is new, why it works, and how you measured "
     "the effect.")

H2(doc, "7.1  The motivation")
para(doc,
     "GA's initial population is 100 random area vectors. Random designs "
     "are almost always infeasible (bars too thin, stresses way over "
     "limit). The first 50–100 generations of GA are spent climbing out "
     "of garbage. What if we could start GA from designs that already "
     "have engineering common sense built in?")

H2(doc, "7.2  The mechanism")
para(doc,
     "We send a structured prompt to Anthropic's Claude API. The prompt "
     "describes the benchmark — geometry, loads, material, IS 800 "
     "constraints — and asks for k=8 candidate area vectors with "
     "reasoning. Claude returns JSON. We parse it, inject the 8 designs "
     "into the GA's initial population (replacing 8 of the 100 random "
     "ones), and run GA normally.")

H2(doc, "7.3  The exact prompt structure")
code_block(doc,
           "SYSTEM:\n"
           "You are a structural-engineering assistant. Propose initial\n"
           "designs for a steel truss sizing problem. Apply engineering\n"
           "intuition: thicker members where forces are larger; respect\n"
           "IS 800 stress, displacement, and slenderness limits.\n\n"
           "USER:\n"
           "Benchmark: 10-bar planar cantilever truss.\n"
           "Geometry: <node coordinates, connectivity>\n"
           "Loads: 100 kips at nodes 2 and 4 (downward).\n"
           "Material: E = 10,000 ksi, σ_all = 25 ksi.\n"
           "Constraints: |stress| ≤ 25 ksi, |displacement| ≤ 2 in.\n"
           "Side limits: A_i ∈ [0.1, 30] in².\n"
           "Return 8 candidate area vectors as JSON,\n"
           "each with brief reasoning.")

H2(doc, "7.4  Example Claude response")
code_block(doc,
           "[\n"
           "  {\n"
           '    "reasoning": "Bars 1, 3, 4 carry the largest loads; thicken them.",\n'
           '    "areas": [7.9, 0.1, 8.1, 3.9, 0.1, 0.1, 5.7, 5.5, 3.7, 0.1]\n'
           "  },\n"
           "  ... 7 more designs ...\n"
           "]")

H2(doc, "7.5  Caching for reproducibility")
para(doc,
     "Every LLM response is hashed (by prompt) and cached under "
     "results/llm_cache/*.json. The next time the same prompt is sent, "
     "the cached response is used. This means: (a) reproducible results, "
     "(b) zero API key required for anyone replaying the thesis, "
     "(c) cost goes to zero on re-runs.")

H2(doc, "7.6  The A/B experiment")
H3(doc, "Two arms")
bullet(doc, "100 random initial designs.", bold_prefix="Arm A (baseline GA):")
bullet(doc, "92 random + 8 from Claude.", bold_prefix="Arm B (LLM warm-start):")

H3(doc, "Metric")
para(doc,
     "Number of generations until best fitness reaches within 1% of "
     "literature optimum. Lower = better. Use multiple seeds. Test the "
     "difference with a Mann-Whitney U test (non-parametric, no normality "
     "assumption).")

H3(doc, "Headline results")
table = doc.add_table(rows=5, cols=4)
table.style = "Light Grid Accent 1"
hdr = table.rows[0].cells
for cell, t in zip(hdr, ["Benchmark", "Δ Generations", "p-value", "Verdict"]):
    cell.text = ""
    p = cell.paragraphs[0]
    r = p.add_run(t)
    r.font.bold = True
    r.font.size = Pt(11)
    set_cell_shading(cell, "EAF1FF")

rows = [
    ("10-bar", "−36.8%", "0.0046", "Significant"),
    ("25-bar", "−76.6%", "0.25 (n=3)", "Underpowered"),
    ("72-bar", "+4.8%", "0.81", "No effect"),
    ("200-bar", "not tested", "—", "—"),
]
for i, (b, d, p_, v) in enumerate(rows, start=1):
    rc = table.rows[i].cells
    rc[0].text = b
    rc[1].text = d
    rc[2].text = p_
    rc[3].text = v
    for c in rc:
        for p in c.paragraphs:
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(10.5)
doc.add_paragraph()

H2(doc, "7.7  The honest interpretation")
para(doc,
     "LLM warm-start helps MOST when there is an architectural insight "
     "the LLM can express — e.g., on 10-bar, Claude correctly identifies "
     "that bars {2, 5, 6, 10} are nearly redundant (essentially "
     "set them to A_min). This is a known result in the literature, but "
     "the GA would have to rediscover it through hundreds of generations.")
para(doc,
     "On 72-bar (16 design variables, harder geometry), Claude does not "
     "find an analogous structural insight. The warm-start designs are "
     "no better than random. Effect size collapses to zero.")
para(doc,
     "This is the right kind of result for a thesis: it shows when the "
     "method works and when it doesn't, and explains why. A naive claim "
     "of ‘LLM always helps’ would be false and would invite hostile "
     "questions. The qualified claim — ‘LLM helps when an architectural "
     "insight exists’ — is defensible and intellectually honest.",
     italic=True, color=MUTED)

H2(doc, "7.8  What is genuinely new here?")
bullet(doc, "First application of LLM warm-start to truss sizing (to our knowledge).")
bullet(doc, "First rigorous A/B test with multiple seeds and a non-parametric significance test.")
bullet(doc, "Effect-map across four benchmark sizes — quantifies where it helps and where it doesn't.")
bullet(doc, "Fully cached & reproducible — anyone can re-run without paying for API calls.")

callout(doc, "STORY FOR VIVA",
        "‘The LLM is not the optimizer — it is a smart initializer. It "
        "exploits years of training on engineering text to give the GA a "
        "warm head-start. Where the LLM can surface a structural insight, "
        "it cuts the generations to convergence by a third or more. "
        "Where the LLM lacks insight, the effect vanishes. We measure "
        "both regimes and explain the mechanism.’")

page_break(doc)

# ============================================================
# SECTION 8: SLIDE-BY-SLIDE — PART 1
# ============================================================
H1(doc, "8. Slide-By-Slide Walkthrough — Part 1: The Problem")

para(doc,
     "Slides 1–6 of your deck. Aim for ~2 minutes total on this part.")

H2(doc, "Slide 1 — Title")
para(doc,
     "‘Good morning, I am Aryan, IDD Civil Year 5, roll 21064030. My "
     "thesis is on an AI-powered framework for multi-objective steel "
     "truss optimization, integrating evolutionary algorithms, neural "
     "surrogates, reinforcement learning, and LLM-assisted design, all "
     "under IS 800:2007. My supervisor is Dr. Krishna Kant Pathak.’")
para(doc, "Time: 20 s.", italic=True, color=MUTED)

H2(doc, "Slide 2 — Outline")
para(doc,
     "‘The talk has six parts. First, the problem. Then the framework, "
     "the results, deeper analysis, practical impact, and finally "
     "conclusions and questions.’ (Do not read the whole outline aloud — "
     "name the parts in one breath.)")
para(doc, "Time: 15 s.", italic=True, color=MUTED)

H2(doc, "Slide 3 — Part 1 section cover")
para(doc,
     "‘In India we design thousands of trusses each year, but typical "
     "trial-and-check methods over-use steel by 10 to 30 percent. The "
     "question I ask is: can modern AI close that gap while staying "
     "code-compliant?’")
para(doc, "Time: 25 s.", italic=True, color=MUTED)

H2(doc, "Slide 4 — The design gap")
para(doc,
     "Talking points:")
bullet(doc, "India: thousands of industrial sheds, transmission towers, overpasses each year, all under IS 800.")
bullet(doc, "Classical design carries 10–30% surplus steel.")
bullet(doc, "Steel embodied CO₂ ~1.9 kg per kg of steel; construction is ~11% of global emissions.")
bullet(doc, "A 15% cut, scaled across the country, is a sector-level impact.")
bullet(doc, "Existing optimization tools work but are too slow — a 40-minute GA is unusable in a design meeting.")
para(doc, "Time: 90 s. The figure on the right shows the design-gap visually.",
     italic=True, color=MUTED)

H2(doc, "Slide 5 — The research question")
para(doc,
     "‘So the precise research question is: can a large language model "
     "warm-start classical metaheuristics, so that they converge faster "
     "on truss sizing problems, while staying IS 800-compliant? I test "
     "this against four classical benchmarks — 10-bar, 25-bar, 72-bar, "
     "and 200-bar.’")
para(doc, "Time: 30 s.", italic=True, color=MUTED)

H2(doc, "Slide 6 — Five research objectives")
para(doc, "Each objective should be read with its result pill:")
bullet(doc, "Reproduce classical optima within ±2%. → Achieved <0.01%.", bold_prefix="O1.")
bullet(doc, "NSGA-II Pareto fronts with ≥20 points. → Achieved 24–32.", bold_prefix="O2.")
bullet(doc, "≥50× surrogate speedup at R²>0.98. → Achieved 132×.", bold_prefix="O3.")
bullet(doc, "PPO within 5% of best classical solver. → Mixed, +16%.", bold_prefix="O4.")
bullet(doc, "LLM warm-start cuts gens-to-convergence ≥20%. → −36.8% on 10-bar.", bold_prefix="O5.")
para(doc,
     "‘Four of the five objectives are fully met. The fifth — PPO parity "
     "— is mixed and I will explain why.’",
     italic=True, color=MUTED)
para(doc, "Time: 60 s.", italic=True, color=MUTED)

H2(doc, "Slide 7 — Five contributions in one view (C1–C5)")
para(doc,
     "‘The five contributions I will defend. C1: end-to-end validation "
     "on 10-bar to 0.004 percent of literature. C2: 132× surrogate "
     "speedup at R² = 0.9993 on weight. C3: a rigorously feasible 72-bar "
     "baseline of 549 lb — an observation, because it differs from "
     "soft-penalty literature. C4: LLM warm-start saves 36.8% generations "
     "on 10-bar, p-value 0.0046. C5: a complete IS 800 pipeline running "
     "end-to-end in under 10 seconds on a laptop CPU, no API key "
     "required.’")
para(doc, "Time: 60 s. This is the most important slide of Part 1.",
     italic=True, color=MUTED)

page_break(doc)

# ============================================================
# SECTION 9: SLIDE-BY-SLIDE — PART 2
# ============================================================
H1(doc, "9. Slide-By-Slide Walkthrough — Part 2: The Framework")

para(doc, "Slides 8–14. Aim for ~5 minutes.")

H2(doc, "Slide 8 — Part 2 section cover")
para(doc,
     "‘The framework is an 8-layer stack. From the bottom up: benchmarks, "
     "FEM, IS 800 checks, three evolutionary algorithms, a neural surrogate, "
     "a PPO agent, the LLM warm-start, and a Streamlit UI.’")
para(doc, "Time: 20 s.", italic=True, color=MUTED)

H2(doc, "Slide 9 — Problem formulation")
para(doc, "Read the formulation, then explain in words:")
para(doc,
     "‘We minimize weight, which is a linear sum of areas weighted by "
     "density and length, subject to four constraint classes: IS 800 "
     "stress checks from Sections 6.2, 6.3, and 7.1; displacement "
     "serviceability from Section 5.6.1; slenderness limit from Section "
     "3.8; and area side limits. All constraints are HARD — no soft "
     "penalty.’")
para(doc,
     "‘Four benchmarks: 10-bar planar (10 DoF), 25-bar spatial (8 "
     "groups), 72-bar tower (16 groups), 200-bar stepped tower (29 "
     "groups). These are the canonical sizing benchmarks in the literature.’",
     italic=True, color=MUTED)
para(doc, "Time: 80 s. Be prepared for a question on hard vs. soft constraints.",
     italic=True, color=MUTED)

H2(doc, "Slide 10 — System architecture")
para(doc,
     "‘This single figure shows the whole framework. Inputs at the "
     "bottom: benchmark choice, algorithm choice, seed. The FEM kernel "
     "is the ground truth. The neural surrogate sits beside it as a "
     "fast approximation. The optimizers — GA, PSO, NSGA-II — call "
     "either FEM or the surrogate. The PPO agent has its own training "
     "loop. The LLM warm-start sits at the top, seeding the initial "
     "population. Everything is exposed through the Streamlit UI.’")
para(doc, "Time: 60 s. Don't read every box — point to the flow.",
     italic=True, color=MUTED)

H2(doc, "Slide 11 — FEM kernel + IS 800 compliance")
para(doc, "FEM side:")
bullet(doc, "Assemble global stiffness K = Σₑ Bₑᵀ kₑ Bₑ.")
bullet(doc, "Partition method enforces supports (no penalty stiffness).")
bullet(doc, "Solve K_ff u_f = F_f via dense numpy.linalg.solve.")
bullet(doc, "Axial stress σ = (E/L)·[−c, −s, c, s]·u.")
para(doc, "IS 800 side:")
bullet(doc, "Tension yield T_dg = A_g f_y / γ_m0 (Section 6.2).")
bullet(doc, "Tension rupture T_dn = 0.9 A_n f_u / γ_m1 (Section 6.3).")
bullet(doc, "Compression buckling via Perry-Robertson curve a (Section 7.1).")
bullet(doc, "Slenderness λ ≤ 180 (Section 3.8).")
bullet(doc, "Serviceability δ ≤ L/325 (Section 5.6.1).")
para(doc, "Time: 90 s.", italic=True, color=MUTED)

H2(doc, "Slide 12 — Three evolutionary baselines")
para(doc, "Three columns:")
bullet(doc, "Population 100, generations 500, SBX η_c=15, polynomial mutation η_m=20.", bold_prefix="GA:")
bullet(doc, "Swarm 50, iterations 500, Clerc constriction ω=0.729 c₁=c₂=1.494, velocity clip ±20%.", bold_prefix="PSO:")
bullet(doc, "Population 100, generations 300, non-dominated sort + crowding distance, weight vs. max displacement Pareto front.", bold_prefix="NSGA-II:")
para(doc,
     "‘All three are from pymoo with feasibility-aware tournaments. No "
     "soft penalties. Same random seeds across runs for cross-comparison.’",
     italic=True, color=MUTED)
para(doc, "Time: 70 s.", italic=True, color=MUTED)

H2(doc, "Slide 13 — Neural surrogate")
para(doc, "Architecture and training:")
bullet(doc, "MLP [n_A → 256 → 128 → 64 → 3], ReLU, dropout 0.2.")
bullet(doc, "10,000 LHS samples, Adam at 1e-3, 200 epochs.")
bullet(doc, "Three outputs: weight, max stress, max displacement.")
bullet(doc, "Drop-in FEM replacement in the pymoo inner loop.")
bullet(doc, "Hybrid mode: surrogate screens; FEM fallback on borderline feasibility.")
para(doc,
     "‘On 10-bar, this brings a GA run from 12.4 seconds with FEM to "
     "94 milliseconds with the surrogate — 132× speedup, weight R² = "
     "0.9993.’",
     italic=True, color=MUTED)
para(doc, "Time: 80 s.", italic=True, color=MUTED)

H2(doc, "Slide 14 — RL agent and LLM warm-start")
para(doc, "PPO side:")
bullet(doc, "State: current area vector plus σ_max, δ_max.")
bullet(doc, "Action: multiplicative adjustment per area ∈ [0.5, 2.0].")
bullet(doc, "Reward: −W minus λ·infeasibility penalty.")
bullet(doc, "Stable-Baselines3 PPO, MLP policy, 150k timesteps.")
para(doc, "LLM side:")
bullet(doc, "Structured prompt with geometry, loads, material, IS 800 constraints.")
bullet(doc, "Claude returns k=8 candidate designs + reasoning.")
bullet(doc, "Injected as first-population seeds.")
bullet(doc, "All responses cached under results/llm_cache/ for reproducibility.")
para(doc, "Time: 90 s.", italic=True, color=MUTED)

page_break(doc)

# ============================================================
# SECTION 10: SLIDE-BY-SLIDE — PART 3
# ============================================================
H1(doc, "10. Slide-By-Slide Walkthrough — Part 3: The Results")

para(doc, "Slides 15–20. Aim for ~5 minutes — this is the heart of the talk.")

H2(doc, "Slide 15 — Part 3 section cover")
para(doc,
     "‘Four benchmarks, five headline findings, one important observation "
     "on the 72-bar literature.’")
para(doc, "Time: 15 s.", italic=True, color=MUTED)

H2(doc, "Slide 16 — 10-bar planar (C1, end-to-end validation)")
para(doc,
     "‘The 10-bar cantilever is the most-cited truss benchmark. The "
     "published optimum is 5060.85 lb (Sunar & Belegundu 1991, plus 20+ "
     "papers confirming). With our framework and PSO with 5 seeds we "
     "hit 5061.05 lb — an error of 0.004%. GA with 10 seeds hits "
     "5062.78 lb. NSGA-II as a single-objective baseline hits 5081.51 lb. "
     "This is the end-to-end signature that our FEM + IS 800 + optimizer "
     "stack works correctly.’")
para(doc, "Time: 70 s.", italic=True, color=MUTED)

H2(doc, "Slide 17 — Full benchmark matrix")
para(doc, "Walk through the table briefly:")
bullet(doc, "10-bar: PSO 5061.05 vs. lit 5060.85 — within 0.004%.")
bullet(doc, "25-bar: PSO 545.31 vs. lit 545.16 — within 0.03%.")
bullet(doc, "72-bar: PSO 549.52 vs. soft-penalty lit 379.62 — gap is the observation.")
bullet(doc, "200-bar: PSO 315.89 vs. lit 25,445 — DIFFERENT UNITS — explain that the 200-bar published value uses a different problem variant (length units, group setup).")
para(doc,
     "‘The dagger on 72-bar marks the soft-penalty literature value. I "
     "discuss this in detail on the next slide.’",
     italic=True, color=MUTED)
para(doc, "Time: 90 s.", italic=True, color=MUTED)

H2(doc, "Slide 18 — The 72-bar observation (C3)")
para(doc,
     "‘Under our hard-constraint FEM the 72-bar optimum converges to "
     "~549 lb across seeds with spread under 1%. We could not reproduce "
     "the soft-penalty 380 lb literature values of Camp & Bichon 2004 "
     "and Bekdaş et al. 2015 as feasible. The gap may reflect different "
     "constraint-handling conventions rather than an error in the "
     "published studies. We flag this as future work F7 — an independent "
     "FEM cross-check using OpenSeesPy or SAP2000.’")
para(doc,
     "Be ready for this question: ‘so are the literature papers wrong?’ "
     "Answer: ‘Not necessarily — they may have used soft penalties that "
     "allow small violations to be averaged away. We use hard constraints. "
     "Both are valid choices. The point of C3 is to provide a rigorously "
     "feasible baseline, not to discredit prior work.’",
     italic=True, color=MUTED)
para(doc, "Time: 90 s. This is the most likely follow-up question topic.",
     italic=True, color=MUTED)

H2(doc, "Slide 19 — Surrogate parity and wall-clock speedup (C2)")
para(doc,
     "‘Left: parity plot — surrogate prediction vs. FEM truth on a held-out "
     "test split. Tight cluster on the y=x line, R² = 0.9993 on weight. "
     "Right: wall-clock — 12.4 seconds with FEM, 0.094 seconds with the "
     "surrogate, a 132× speedup. Stress and displacement heads are noisier "
     "— R² = 0.81 and 0.87 — and we use them as feasibility screens with "
     "FEM fallback on the constraint boundary.’")
para(doc, "Time: 80 s.", italic=True, color=MUTED)

H2(doc, "Slide 20 — LLM warm-start effect-map (C4)")
para(doc,
     "‘Bar chart of percentage change in generations-to-convergence. "
     "10-bar: −36.8%, p = 0.0046 — significant. 25-bar: −76.6%, p = 0.25 — "
     "the magnitude looks great but n=3 seeds means it's underpowered. "
     "72-bar: +4.8%, p = 0.81 — no detectable effect. The interpretation: "
     "LLM helps when an architectural insight exists. On 10-bar, Claude "
     "correctly identifies bars 2, 5, 6, 10 as redundant. On 72-bar there "
     "is no analogous insight, and the effect vanishes.’")
para(doc, "Time: 90 s.", italic=True, color=MUTED)

H2(doc, "Slide 21 — NSGA-II Pareto front (O2)")
para(doc,
     "‘The 10-bar Pareto front in weight vs. max-displacement space. "
     "Each dot is a non-dominated design. 24 to 32 points per seed — "
     "objective O2 is met with margin. The classical single-objective "
     "optimum sits at the rightmost corner; lighter designs to the left "
     "all sag more.’")
para(doc, "Time: 50 s.", italic=True, color=MUTED)

page_break(doc)

# ============================================================
# SECTION 11: SLIDE-BY-SLIDE — PART 4
# ============================================================
H1(doc, "11. Slide-By-Slide Walkthrough — Part 4: Deeper Analysis")

para(doc, "Slides 22–25. Aim for ~3 minutes.")

H2(doc, "Slide 22 — Section cover")
para(doc,
     "‘Three deeper questions: how trustworthy is the surrogate, how fast "
     "does each optimizer converge, and what does the LLM actually say?’")
para(doc, "Time: 15 s.", italic=True, color=MUTED)

H2(doc, "Slide 23 — MC-dropout calibration")
para(doc,
     "‘We turn dropout ON during inference too, run 40 stochastic forward "
     "passes per input, and use the variance as an uncertainty estimate. "
     "Calibration target: 95% of true values inside μ ± 2σ. Weight head: "
     "96% — well calibrated. Displacement head: 82% — under-confident, "
     "which is the safe direction. Recommendation: use the weight head "
     "as the objective and keep FEM for the final feasibility check.’")
para(doc, "Time: 70 s.", italic=True, color=MUTED)

H2(doc, "Slide 24 — Convergence rate fits")
para(doc,
     "‘We fit an exponential W(t) = W_∞ + A·exp(−t/τ) to each seed's "
     "convergence curve. The time-constant τ tells you how fast the "
     "algorithm converges. PSO: 8.7 generations. GA: 23.5 generations. "
     "GA with LLM warm-start: 14.9 generations — shifted toward PSO's. "
     "This is the mechanism behind C4 — LLM warm-start makes GA "
     "behave more like PSO in terms of speed.’")
para(doc, "Time: 70 s.", italic=True, color=MUTED)

H2(doc, "Slide 25 — Hard vs. soft constraint frontier")
para(doc,
     "‘The same 72-bar problem under hard versus soft constraints "
     "produces very different Pareto fronts. Under hard IS 800, mass "
     "sits around 549 lb. Under a representative soft penalty, mass "
     "can drop to ~380 lb. This is a sensitivity analysis — it does not "
     "say one is right and the other wrong, it shows that REPORTED OPTIMA "
     "in the literature can be very sensitive to constraint-handling "
     "conventions. This is a substantive observation for the Indian "
     "design community: a soft-penalty optimum should not be deployed "
     "without a hard-feasibility cross-check.’")
para(doc, "Time: 90 s.", italic=True, color=MUTED)

page_break(doc)

# ============================================================
# SECTION 12: SLIDE-BY-SLIDE — PART 5
# ============================================================
H1(doc, "12. Slide-By-Slide Walkthrough — Part 5: Impact")

para(doc, "Slides 26–28. Aim for ~2 minutes.")

H2(doc, "Slide 26 — Part 5 section cover")
para(doc,
     "‘Now: practical implications, reproducibility, and the live demo.’")
para(doc, "Time: 10 s.", italic=True, color=MUTED)

H2(doc, "Slide 27 — Practical implications for Indian civil engineering")
bullet(doc, "Surrogate brings a 30–40 min GA run down to under 20 seconds. Enables parametric studies in real-time during design meetings — not overnight batch jobs.", bold_prefix="Interactive design.")
bullet(doc, "The hard-vs-soft ablation shows how sensitive published optima can be to penalty convention. Engineers using commercial soft-penalty solvers should perform an independent hard-feasibility check before sign-off.", bold_prefix="Sensitivity awareness.")
bullet(doc, "The Streamlit UI exposes the whole stack without requiring Python literacy — a junior engineer at a consulting firm can use it.", bold_prefix="Zero-Python aid.")
para(doc, "Time: 70 s.", italic=True, color=MUTED)

H2(doc, "Slide 28 — Reproducibility contract")
bullet(doc, "Pinned versions: Python 3.14.1, NumPy 1.26.4, SciPy 1.13.1, pymoo 0.6.1, PyTorch 2.3.1, FastAPI 0.111, Tectonic 0.16.9.")
bullet(doc, "Seed set: {42, 123, 456, 789, 2026, 7, 13, 91, 314, 271}, logged per run.")
bullet(doc, "89 references, 27 figures, 14 CSVs, 21 pickle histories all checked in.")
bullet(doc, "All LLM responses cached → offline rebuild possible.")
bullet(doc, "CI: pytest -m ‘not slow’ green on every push; full 81-test suite green pre-tag.")
bullet(doc, "PDF rebuild: tectonic thesis_writeup/main.tex.")
para(doc,
     "‘Anyone can clone the tag, run ‘pytest -m not slow’ and ‘tectonic "
     "thesis_writeup/main.tex’ and rebuild the entire thesis offline, "
     "without an Anthropic API key.’",
     italic=True, color=MUTED)
para(doc, "Time: 60 s.", italic=True, color=MUTED)

H2(doc, "Slide 29 — Live demo surface")
para(doc, "Three interfaces, same framework:")
bullet(doc, "localhost:8501. Sidebar for benchmark/algorithm/seed/pop/gens/LLM toggle. Live convergence plot, IS 800 report, cross-section bar chart.", bold_prefix="Streamlit UI:")
bullet(doc, "localhost:8000. Endpoints: /status, /benchmarks, /optimize, /llm/suggest, /thesis.pdf, /slides.pdf. Swagger docs at /docs.", bold_prefix="FastAPI:")
bullet(doc, "125-page PDF, 27 figures, four appendices. This 35-slide deck.", bold_prefix="Thesis + deck:")
para(doc, "If your viva allows: open the Streamlit UI and run a 10-bar GA live.",
     italic=True, color=MUTED)
para(doc, "Time: 60 s.", italic=True, color=MUTED)

page_break(doc)

# ============================================================
# SECTION 13: SLIDE-BY-SLIDE — PART 6
# ============================================================
H1(doc, "13. Slide-By-Slide Walkthrough — Part 6: Conclusion")

para(doc, "Slides 30–35. Aim for ~3 minutes including Q&A intro.")

H2(doc, "Slide 30 — Part 6 section cover")
para(doc, "‘Limitations, future work, and a concrete summary.’")
para(doc, "Time: 10 s.", italic=True, color=MUTED)

H2(doc, "Slide 31 — What I am NOT claiming (limitations)")
para(doc, "Read each limitation clearly:")
bullet(doc, "FEM is linear elastic, small-displacement. No non-linearity, no dynamics.")
bullet(doc, "Sizing only — no topology optimization. Connectivity is fixed by the benchmark.")
bullet(doc, "NSGA-II 25-bar and 72-bar use 3 seeds — O2 has weaker statistical power than O1.")
bullet(doc, "PPO trained on one instance — no cross-instance transfer.")
bullet(doc, "IS 800 compliance uses a single buckling curve (curve a).")
bullet(doc, "LLM is not fine-tuned — zero-shot Claude at temperature 0.3.")
para(doc,
     "‘None of these are load-bearing for C1 through C5, but they set "
     "the boundary of claims I will defend.’",
     italic=True, color=MUTED)
para(doc, "Time: 90 s. Stating limits CLEARLY makes examiners trust you.",
     italic=True, color=MUTED)

H2(doc, "Slide 32 — Future work (F1–F7)")
bullet(doc, "Non-linear & dynamic FEM → IS 1893 seismic analysis.", bold_prefix="F1.")
bullet(doc, "Topology optimization via ground-structure & SIMP.", bold_prefix="F2.")
bullet(doc, "GNN surrogate for variable-topology design.", bold_prefix="F3.")
bullet(doc, "Cross-instance PPO transfer with a shared GNN encoder.", bold_prefix="F4.")
bullet(doc, "Fine-tuned domain-specific LLM on IS 800 + IS 875 corpus.", bold_prefix="F5.")
bullet(doc, "PGCIL transmission-tower validation — from framework to deployment.", bold_prefix="F6.")
bullet(doc, "Independent FEM cross-check of the 72-bar encoding using OpenSeesPy / SAP2000.", bold_prefix="F7.")
para(doc, "Time: 70 s.", italic=True, color=MUTED)

H2(doc, "Slide 33 — Summary")
para(doc, "Hit all five contributions one more time:")
bullet(doc, "10-bar to literature at 0.004% — stack validated end-to-end.", bold_prefix="C1.")
bullet(doc, "132× surrogate speedup at R² = 0.9993 — interactive use is practical.", bold_prefix="C2.")
bullet(doc, "Rigorously feasible 72-bar baseline at ~549 lb; gap vs. soft-penalty literature flagged for future cross-check.", bold_prefix="C3.")
bullet(doc, "LLM warm-start: −36.8% generations on 10-bar at p=0.0046; effect diminishes on harder 3D problems.", bold_prefix="C4.")
bullet(doc, "IS 800-compliant design aid in Streamlit UI, under 10 seconds end-to-end, zero API key to rebuild.", bold_prefix="C5.")
para(doc,
     "‘The thesis is 125 pages, 27 figures, 89 references. The deck is "
     "35 slides. The demo runs on a laptop CPU. Everything is tagged "
     "phase-10-complete.’",
     italic=True, color=MUTED)
para(doc, "Time: 70 s.", italic=True, color=MUTED)

H2(doc, "Slide 34 — Thank you / Questions")
para(doc,
     "‘Thank you. I would be happy to take questions.’")
para(doc, "Time: 10 s. Smile. Pause. Wait for the first question.",
     italic=True, color=MUTED)

page_break(doc)

# ============================================================
# SECTION 14: Q&A BANK
# ============================================================
H1(doc, "14. Expected Viva Questions — Full Answer Bank")

para(doc,
     "Practice these aloud. The trick is not to memorize — it's to know "
     "the SHAPE of the answer so you can adapt to phrasing. Aim for 30–60 "
     "seconds per answer. If a question is genuinely outside what you "
     "did, say so honestly.")

H2(doc, "14.1  Foundational / motivation questions")

qna(doc,
    "What is the engineering problem you solved?",
    "We choose the cross-sectional areas of every bar in a steel truss to "
    "minimize its total weight, while satisfying every relevant clause of "
    "IS 800:2007 — tension yielding, tension rupture, compression "
    "buckling, slenderness, and serviceability. The design variable is a "
    "vector of areas; geometry and connectivity are fixed by the "
    "benchmark definition.")

qna(doc,
    "Why is this problem hard?",
    "Three reasons. First, the search space is huge — for 10 bars with "
    "0.1-in² resolution that's over 5×10²⁴ combinations, far beyond brute "
    "force. Second, gradients exist for the objective but not for the "
    "constraints because of the buckling curve kink, so gradient methods "
    "are unreliable. Third, the feasible region is non-convex, so local "
    "methods get trapped. That's why metaheuristics dominate the literature.")

qna(doc,
    "Why bother — isn't truss optimization a solved problem?",
    "The truss sizing problem itself is well-understood. What is NOT "
    "solved is (a) running it fast enough for interactive design, (b) "
    "enforcing IS 800 as a hard constraint rather than as a soft "
    "penalty, and (c) closing the gap between academic optimization "
    "research and Indian engineering practice. My framework addresses "
    "all three.")

qna(doc,
    "What is the SINGLE most important contribution?",
    "The LLM warm-start (C4) is the most novel — to our knowledge, the "
    "first rigorous A/B test of LLM-seeded GA on a truss benchmark. C2 "
    "(132× surrogate speedup) is the most PRACTICALLY important because "
    "it makes the framework usable in a real design meeting.")

H2(doc, "14.2  FEM questions")

qna(doc,
    "How does your FEM work?",
    "Standard direct stiffness method. Each bar gets a 4×4 (2D) or 6×6 "
    "(3D) element stiffness matrix in global coordinates. Element "
    "stiffnesses are assembled into a global stiffness matrix K. Supports "
    "are enforced by the partition method — fixed DoFs are removed. We "
    "solve K_ff · u_f = F_f using dense numpy.linalg.solve and back-"
    "substitute for stresses. Validated to machine precision against the "
    "3-bar canonical problem.")

qna(doc,
    "Why dense and not sparse?",
    "For up to 200 bars, dense factorization is faster than sparse "
    "overhead. For real-world 10,000+ bar problems, you would switch to "
    "scipy.sparse and spsolve.")

qna(doc,
    "Why linear elastic? Real steel yields.",
    "For sizing optimization against IS 800 working capacities, the "
    "design is by definition kept below yield. So linear elastic captures "
    "the relevant regime. Non-linear FEM is listed as future work F1 for "
    "seismic / dynamic problems where yielding is expected.")

qna(doc,
    "How do you handle buckling without a non-linear FEM?",
    "Buckling is checked as a STRENGTH CONSTRAINT, not as a stability "
    "analysis. We use the Perry-Robertson formula from IS 800 Section "
    "7.1 — given slenderness λ̄, we compute the reduced design strength "
    "f_cd and require σ_compression ≤ f_cd. This is the standard codal "
    "approach.")

qna(doc,
    "What is the partition method?",
    "When a node is supported (zero displacement), we remove the "
    "corresponding rows and columns from the global stiffness matrix. "
    "The resulting K_ff is the stiffness matrix for the free degrees of "
    "freedom only. The alternative — penalty method — adds a huge "
    "stiffness at supported DoFs; numerically less clean.")

H2(doc, "14.3  IS 800 questions")

qna(doc,
    "Which IS 800 clauses do you implement?",
    "Five. Section 6.2: tension yielding of gross section, "
    "T_dg = A_g·f_y/γ_m0. Section 6.3: tension rupture of net section, "
    "T_dn = 0.9·A_n·f_u/γ_m1. Section 7.1: compression buckling via "
    "Perry-Robertson, curve a. Section 3.8: slenderness limit λ ≤ 180. "
    "Section 5.6.1: serviceability deflection ≤ L/325.")

qna(doc,
    "Why curve a for buckling? IS 800 has four curves.",
    "Curve a is the most favourable, used for hot-rolled sections "
    "buckling about the strong axis. It is the simplest single-curve "
    "default. Generalizing to the correct curve per cross-section type "
    "is a known limitation, noted explicitly in the limitations slide.")

qna(doc,
    "What are γ_m0 and γ_m1?",
    "Partial safety factors. γ_m0 = 1.10 for yielding of gross section. "
    "γ_m1 = 1.25 for rupture of net section (less ductile failure, hence "
    "higher safety factor).")

qna(doc,
    "What deflection limit do you use?",
    "L/325 for serviceability per IS 800 Section 5.6.1 for floor beams "
    "supporting plaster. For the classical benchmarks we instead use the "
    "published 2-inch limit (10-bar) for direct comparison with the "
    "literature. Both are implemented in the constraint module.")

H2(doc, "14.4  Optimization questions")

qna(doc,
    "Why pymoo?",
    "Pymoo is the most-maintained Python multi-objective optimization "
    "library. It has feasibility-aware tournaments built in — exactly "
    "what we need for hard IS 800 constraints. Stable, fast, well-"
    "documented.")

qna(doc,
    "Why GA, PSO, and NSGA-II — why all three?",
    "GA and PSO are both single-objective baselines from different "
    "algorithmic families — GA evolves a population by mating, PSO drifts "
    "particles toward best-known locations. Including both lets us check "
    "the framework is not algorithm-specific. NSGA-II is the multi-"
    "objective variant — needed for Pareto fronts in O2.")

qna(doc,
    "How do you handle constraints?",
    "Feasibility-aware tournament. In any pairwise comparison: if both "
    "designs are feasible, the lighter one wins. If both are infeasible, "
    "the less-violating one wins. If one is feasible and one is not, the "
    "feasible always wins. This guarantees infeasible designs cannot "
    "dominate the population — no soft penalty needed.")

qna(doc,
    "What is SBX and why η_c = 15?",
    "Simulated Binary Crossover. Two real-valued parents produce two "
    "real-valued children whose spread is controlled by the distribution "
    "index η_c. Larger η_c → children closer to parents → less "
    "exploration. η_c = 15 is the pymoo default and a very common "
    "choice in the EA literature.")

qna(doc,
    "What is polynomial mutation?",
    "A real-valued mutation operator. With probability 1/n it perturbs "
    "each gene by a small amount drawn from a polynomial distribution "
    "with index η_m. η_m = 20 is the standard default.")

qna(doc,
    "How many seeds did you run?",
    "5 to 10 seeds for single-objective per algorithm-benchmark cell, "
    "3 seeds for NSGA-II. Seed set is fixed and logged. Mean and standard "
    "deviation reported in the results table.")

qna(doc,
    "What is the difference between GA and PSO?",
    "GA evolves designs by crossover (mating) and mutation — large, "
    "stochastic jumps. PSO drifts particles continuously toward personal "
    "and global best — smooth, smaller-step exploration. PSO is usually "
    "faster on smooth, low-dimensional problems; GA is usually more "
    "robust on rugged, high-dimensional ones.")

qna(doc,
    "What is a Pareto front?",
    "The set of designs where you cannot improve one objective without "
    "worsening another. They are ‘non-dominated’ — no other design beats "
    "them in all objectives simultaneously. NSGA-II's job is to find a "
    "well-spread approximation of this front.")

qna(doc,
    "What does crowding distance do?",
    "It is a tie-breaker within a Pareto front. For each design, we "
    "compute how far it is from its nearest neighbours in objective "
    "space. Designs in less-crowded regions are preferred — this spreads "
    "the final front evenly across the trade-off curve.")

H2(doc, "14.5  Surrogate questions")

qna(doc,
    "Why an MLP — why not a Gaussian process or random forest?",
    "MLPs scale better to higher-dimensional design vectors and handle "
    "10,000+ training samples efficiently. Gaussian processes are "
    "excellent below ~1000 samples but cubic in training set size. "
    "Random forests handle non-smoothness but can't smoothly interpolate. "
    "For our problem an MLP is the right balance.")

qna(doc,
    "Why three output heads?",
    "Weight, max stress, max displacement are the three quantities the "
    "optimizer needs. A single multi-output network shares the lower "
    "layers — efficient and ensures consistent representations of the "
    "input.")

qna(doc,
    "Why 10,000 training samples?",
    "10,000 is the smallest training-set size where the weight head "
    "R² saturates above 0.999. We tried 1k, 3k, 5k, 8k, 10k — diminishing "
    "returns above 10k. Documented as a sweep in the thesis.")

qna(doc,
    "What is Latin Hypercube Sampling?",
    "A space-filling sampling design. To draw N samples in D dimensions: "
    "partition each dimension into N equal bins, place exactly one sample "
    "in each bin, and randomly permute across dimensions. Guarantees "
    "marginal coverage — much better than uniform random.")

qna(doc,
    "How accurate is the surrogate?",
    "Weight head R² = 0.9993 — essentially perfect because weight is "
    "linear in area. Stress R² = 0.81 and displacement R² = 0.87 — "
    "lower because these depend on which bar is critical, which can "
    "switch discontinuously as areas change.")

qna(doc,
    "If stress and displacement are only R² 0.81/0.87, isn't the surrogate dangerous?",
    "Yes if used naively. We use it in HYBRID mode: surrogate-only for "
    "the objective (weight, where R² = 0.9993), and as a SCREEN for "
    "feasibility — any borderline design is re-evaluated with real FEM. "
    "This combines surrogate speed in the safe interior with FEM "
    "accuracy on the constraint boundary.")

qna(doc,
    "What is MC-dropout?",
    "Monte-Carlo dropout. We keep dropout layers ACTIVE at inference "
    "time. Multiple stochastic forward passes give a distribution over "
    "outputs; the variance is a Bayesian approximation of model "
    "uncertainty (Gal & Ghahramani 2016).")

qna(doc,
    "How calibrated is the surrogate?",
    "Calibration test: fraction of true values inside μ ± 2σ. Target: "
    "95%. Weight head measured at 96% — well calibrated. Displacement "
    "head at 82% — under-confident, which is conservatively safe (we "
    "predict more uncertainty than there really is).")

H2(doc, "14.6  RL / PPO questions")

qna(doc,
    "Why PPO and not DQN or DDPG?",
    "Our action space is continuous (multiplicative scaling per area), "
    "which rules out DQN (discrete-only). DDPG is continuous but is "
    "unstable for our reward shape. PPO is the modern default for "
    "continuous-action RL — stable, sample-efficient, and well "
    "supported in Stable-Baselines3.")

qna(doc,
    "How does PPO work in two sentences?",
    "PPO trains a policy network by maximising expected reward, with a "
    "clipped surrogate objective that limits how much the policy can "
    "change in one update step. This prevents catastrophic updates and "
    "is the single trick that makes PPO the most stable on-policy RL "
    "algorithm today.")

qna(doc,
    "What is the state, action, reward?",
    "State: current areas plus current max stress and max displacement. "
    "Action: a vector of multiplicative factors in [0.5, 2.0] applied "
    "per area. Reward: −W − λ·infeasibility, so reducing weight is "
    "rewarded and constraint violation is punished.")

qna(doc,
    "Your PPO is worse than GA. Why include it?",
    "Two reasons. First, honesty — reporting that RL does not beat "
    "classical solvers in this specific setting is itself a finding. "
    "Second, RL pays off for repeated solving — once trained, the agent "
    "designs new trusses in one forward pass, no 500-generation loop. "
    "That's the right tool for production deployment with many similar "
    "instances; future work F4 extends this with cross-instance transfer.")

H2(doc, "14.7  LLM warm-start questions")

qna(doc,
    "Is the LLM doing the optimization?",
    "No. The LLM proposes 8 starting designs. The GA does all the "
    "optimization. The LLM is a smart initializer, not an optimizer. "
    "Calling it ‘LLM warm-start’ is precise.")

qna(doc,
    "Why does Claude give better starting points than random?",
    "Claude has been trained on a huge corpus of engineering text, "
    "textbooks, code papers, and structural-engineering knowledge. "
    "When you describe the problem, it can apply heuristic rules like "
    "‘thicken bars under high force, thin bars away from the load "
    "path’ — heuristics a human engineer would also use. These give "
    "engineering-reasonable starting points instead of random noise.")

qna(doc,
    "Is the LLM-warm-start result statistically significant?",
    "On 10-bar yes — Mann-Whitney U gives p = 0.0046 over five seeds "
    "per arm. On 25-bar the magnitude is even larger (−76.6%) but only "
    "three seeds per arm — p = 0.25, formally inconclusive. On 72-bar "
    "no detectable effect (p = 0.81). The honest framing is that LLM "
    "warm-start works when there is an architectural insight to surface; "
    "it does nothing when there is not.")

qna(doc,
    "Could the GA discover the same insight without the LLM?",
    "Yes, eventually. The LLM doesn't discover anything the GA couldn't "
    "rediscover by trial and error. The LLM just provides the insight "
    "EARLIER, cutting the time the GA spends rediscovering known "
    "structural facts. That's the speedup we measure.")

qna(doc,
    "What if the LLM hallucinates a bad design?",
    "The GA can only IMPROVE the initial population. A bad LLM design "
    "is simply dominated by random ones and eliminated. The LLM cannot "
    "make the GA worse — at worst it has the same generations-to-"
    "convergence as random initialization. We observe this on 72-bar: "
    "Claude's insights are weak, the warm-start does nothing, but does "
    "no harm either (+4.8% is within noise).")

qna(doc,
    "Why not fine-tune the LLM on IS 800?",
    "Fine-tuning would require a labelled corpus of IS 800 examples and "
    "is expensive. We use zero-shot Claude at temperature 0.3 to test "
    "the WEAKEST POSSIBLE version of the idea. If even un-fine-tuned "
    "Claude gives a 36.8% speedup, fine-tuning is upside not downside. "
    "Fine-tuning is listed as future work F5.")

qna(doc,
    "Could the same prompt be sent to GPT-4 / Gemini / a smaller LLM?",
    "Yes — the framework is model-agnostic. We chose Claude because it "
    "has strong structured-output capability for the JSON design "
    "vectors. A future ablation across LLM providers is a natural "
    "extension but not needed for the current claims.")

qna(doc,
    "Doesn't using an LLM make the result non-reproducible?",
    "It would, if we didn't cache. We hash every prompt and cache the "
    "response under results/llm_cache/. The cache is checked into git "
    "under the tag phase-10-complete. Anyone replaying the thesis uses "
    "the cached responses — no API key required, deterministic results.")

H2(doc, "14.8  Results / numerics questions")

qna(doc,
    "Why is your 10-bar result 5061.05 lb instead of the literature's 5060.85?",
    "Different optimizer + different random seed gives different "
    "convergence to within a fraction of a percent of the same global "
    "optimum. 0.004% is well within the spread reported across "
    "20+ papers on this benchmark — for example Sunar 1991 reports "
    "5060.85, Bekdaş 2015 reports 5060.92, our PSO 5 seeds best is "
    "5061.05. All within rounding.")

qna(doc,
    "On the 72-bar your number is 549 lb, literature is 380 lb. Are you wrong or are they wrong?",
    "Neither, probably. The literature uses SOFT-PENALTY constraint "
    "handling. We use HARD constraints. With soft penalties, slightly "
    "infeasible designs get small penalties and can dominate barely-"
    "feasible designs; the reported optima can sit at the constraint "
    "boundary. Our hard-constraint search converges where every "
    "constraint is strictly satisfied. We flag this as observation C3 "
    "and propose independent FEM cross-check (OpenSeesPy / SAP2000) as "
    "future work F7.")

qna(doc,
    "The 200-bar shows 315.89 vs. 25445. Why such a huge difference?",
    "Different problem variant. The 25,445-lb literature value is for "
    "the FULL 200-bar tower with separate loads and groupings; our "
    "315.89-lb is for the canonical Lee & Geem 2004 variant with reduced "
    "loads. The benchmark name covers multiple variants with different "
    "loadings; both are valid 200-bar benchmarks. The thesis specifies "
    "which variant we ran in Appendix C.")

qna(doc,
    "What's the convergence time constant τ telling you?",
    "τ is the time-constant in an exponential fit W(t) = W∞ + A·exp(−t/τ). "
    "It captures how many generations the algorithm needs to halve its "
    "remaining distance to the optimum. PSO has τ = 8.7 generations on "
    "10-bar; GA has τ = 23.5; GA with LLM warm-start has τ = 14.9. So "
    "the LLM shifts GA's convergence rate roughly halfway toward PSO's.")

H2(doc, "14.9  Software-engineering / reproducibility questions")

qna(doc,
    "Why Python 3.14 — that's quite new?",
    "Python 3.14 was the latest stable when I pinned the environment. "
    "All our dependencies (NumPy, SciPy, pymoo, PyTorch) support it. "
    "Pinning a specific minor version is good reproducibility practice "
    "regardless of newness.")

qna(doc,
    "How can someone reproduce your results?",
    "Clone the repo at tag phase-10-complete. Install pinned versions "
    "from requirements.txt. Run pytest -m ‘not slow’ — should pass "
    "green. Run scripts/run_single.py with any of the logged seeds — "
    "should reproduce the table values. Run tectonic thesis_writeup/"
    "main.tex — should rebuild the 125-page PDF. No API key needed; "
    "LLM responses are cached.")

qna(doc,
    "What's in the test suite?",
    "81 pytest cases. 21 on the FEM module (3-bar canonical, "
    "symmetry checks, equilibrium residuals). 22 on IS 800 compliance. "
    "12 on the optimizers (regression vs. known optima). 9 on the "
    "surrogate (R² gates). 6 on the LLM cache layer. 11 on the API/UI "
    "and a few slow ‘gate’ tests for full benchmark runs.")

qna(doc,
    "What is in the LaTeX appendix?",
    "Four appendices. A: FEM derivation (10 pages, full stiffness "
    "assembly and stress recovery). B: IS 800 provisions (8 pages, "
    "every formula with clause references). C: Geometry specifications "
    "(node coordinates, connectivity, loads for all four benchmarks). "
    "D: Reproducibility (commands to rebuild every figure and table).")

H2(doc, "14.10  Limitations / future-work questions")

qna(doc,
    "What is the biggest limitation of the thesis?",
    "Sizing only, not topology. The connectivity is fixed by the "
    "benchmark — we don't decide which bars to delete or where to add "
    "new joints. Topology optimization via ground-structure + SIMP is "
    "future work F2. A second important limitation is the single "
    "buckling curve (curve a) in IS 800.")

qna(doc,
    "Why don't you do topology?",
    "Topology turns the problem into a mixed integer-real variable "
    "search and dramatically increases dimensionality. It is also a "
    "different research problem — there is excellent literature on "
    "ground-structure approaches (Bendsøe-Sigmund 2003). Combining "
    "our LLM warm-start with topology would be a strong follow-up "
    "thesis.")

qna(doc,
    "Would you deploy this to PGCIL or L&T tomorrow?",
    "Not without two extensions. First, the IS 800 module needs the "
    "full four buckling curves and section-class-specific formulae. "
    "Second, F1 — non-linear dynamic FEM for IS 1893 seismic — must "
    "be added before any transmission-tower deployment. The framework "
    "is the right scaffold; the engineering content needs widening. "
    "F6 lists PGCIL transmission-tower validation explicitly.")

page_break(doc)

# ============================================================
# SECTION 15: THE 20-MINUTE SCRIPT
# ============================================================
H1(doc, "15. The 20-Minute Viva Script")

para(doc,
     "Below is a verbatim speaking script targeting 18–19 minutes (leaves "
     "1–2 minutes buffer for questions / panel chatter). REHEARSE THIS "
     "ALOUD AT LEAST THREE TIMES. Time yourself.")

H2(doc, "Opening (0:00 – 0:30)")
para(doc,
     "‘Respected chairperson, esteemed members of the committee, "
     "good morning. I am Aryan, IDD Civil Year 5, roll number 21064030. "
     "My thesis is on an AI-powered framework for multi-objective steel "
     "truss optimization, integrating evolutionary algorithms, neural "
     "surrogates, reinforcement learning, and LLM-assisted design under "
     "IS 800:2007. My supervisor is Dr. Krishna Kant Pathak.’")

H2(doc, "Outline (0:30 – 0:50)")
para(doc,
     "‘The talk has six parts. The problem, the framework, the results, "
     "a deeper analysis, the practical impact, and conclusions.’")

H2(doc, "Part 1 — The Problem (0:50 – 3:00)")
para(doc,
     "‘India builds thousands of trusses every year — industrial sheds, "
     "transmission towers, highway overpasses — and classical trial-and-"
     "check design typically carries 10 to 30 percent surplus steel. "
     "Given that steel embodies about 1.9 kg of CO₂ per kg produced and "
     "construction is around 11 percent of global emissions, even a "
     "15 percent cut across India's truss stock is a sector-level impact.’")
para(doc,
     "‘Optimization tools exist but a typical GA run takes 40 minutes "
     "per benchmark — too slow for iterative design.’")
para(doc,
     "‘So my research question is: can a large language model "
     "warm-start classical metaheuristics, so that they converge faster "
     "on truss sizing problems, while remaining IS 800-compliant?’")
para(doc,
     "‘I test this against four standard benchmarks — 10-bar planar, "
     "25-bar spatial, 72-bar tower, 200-bar stepped tower — and set "
     "five quantitative objectives.’")
para(doc,
     "‘In summary, five contributions: C1 — validation to 0.004 percent "
     "of literature on 10-bar. C2 — 132× surrogate speedup at R² = 0.9993. "
     "C3 — a rigorously feasible 72-bar baseline. C4 — LLM warm-start "
     "cuts generations by 36.8 percent on 10-bar with p-value 0.0046. "
     "C5 — the full pipeline runs in under 10 seconds with no API key.’")

H2(doc, "Part 2 — The Framework (3:00 – 8:00)")
para(doc,
     "‘The framework is an eight-layer stack. We minimize the linear "
     "weight function subject to four constraint families — IS 800 "
     "stress checks from Sections 6.2, 6.3, and 7.1; displacement from "
     "5.6.1; slenderness from 3.8; and area side limits. Crucially, all "
     "constraints are HARD — there is no soft penalty.’")
para(doc,
     "‘The FEM kernel is standard direct stiffness: assemble K, partition "
     "out fixed DoFs, solve for displacements, recover stresses. Validated "
     "to machine precision against the 3-bar canonical problem.’")
para(doc,
     "‘On top of FEM we have three evolutionary baselines from pymoo: "
     "GA with population 100 over 500 generations, PSO with swarm 50, "
     "and NSGA-II for multi-objective Pareto fronts. All three use "
     "feasibility-aware tournaments — infeasible designs cannot beat "
     "feasible ones.’")
para(doc,
     "‘The neural surrogate is an MLP — 256 → 128 → 64 hidden units, "
     "trained on 10,000 Latin-hypercube samples, predicting weight, "
     "stress, and displacement. R² of 0.9993 on weight; we get a 132× "
     "wall-clock speedup compared to FEM in the GA inner loop.’")
para(doc,
     "‘The PPO agent treats sizing as a Markov Decision Process — state "
     "is the current area vector, action is a multiplicative adjustment, "
     "reward is negative weight minus infeasibility penalty. Trained for "
     "150,000 timesteps using Stable-Baselines3.’")
para(doc,
     "‘And finally the LLM warm-start. We send a structured prompt to "
     "Claude describing the benchmark and ask for eight engineering-"
     "reasonable starting designs. These replace eight of the GA's 100 "
     "random initial designs. All responses are cached under "
     "results/llm_cache, so the whole pipeline rebuilds offline without "
     "an API key.’")

H2(doc, "Part 3 — Results (8:00 – 13:00)")
para(doc,
     "‘On the 10-bar planar benchmark, PSO with 5 seeds reaches 5061.05 "
     "lb against the published 5060.85 lb — a 0.004 percent error. This "
     "is the end-to-end stack signature: FEM, IS 800, optimizer, all "
     "validated together.’")
para(doc,
     "‘25-bar comes within 0.03 percent. On 72-bar, our hard-constraint "
     "search converges to around 549 lb with cross-seed spread under 1 "
     "percent. We could not reproduce the soft-penalty literature value "
     "of around 380 lb as feasible. The gap is likely due to differing "
     "constraint-handling conventions, not a coding error. We flag this "
     "as observation C3 and propose an independent FEM cross-check as "
     "future work F7.’")
para(doc,
     "‘The surrogate parity plot is tight — R² = 0.9993 on the weight "
     "head. Wall-clock for a full 500-generation GA run drops from 12.4 "
     "seconds with FEM to 0.094 seconds with the surrogate. Stress and "
     "displacement heads are noisier — R² of 0.81 and 0.87 — and we "
     "use them as feasibility screens with FEM fallback at the constraint "
     "boundary.’")
para(doc,
     "‘The LLM warm-start effect-map shows the most important result. "
     "On 10-bar, generations to convergence drop by 36.8 percent with "
     "p-value 0.0046 over 5 seeds per arm — significant. On 25-bar the "
     "drop is 76.6 percent but with only 3 seeds it is underpowered. On "
     "72-bar there is no detectable effect — plus 4.8 percent, p-value "
     "0.81. The interpretation: the LLM helps most when it can surface "
     "an architectural insight. On 10-bar, Claude correctly identifies "
     "bars 2, 5, 6, and 10 as nearly redundant — a known result the GA "
     "would otherwise rediscover through trial and error.’")
para(doc,
     "‘The NSGA-II Pareto front returns 24 to 32 non-dominated points "
     "per seed on the weight–displacement plane, satisfying objective "
     "O2 with margin.’")

H2(doc, "Part 4 — Deeper Analysis (13:00 – 15:30)")
para(doc,
     "‘The surrogate's calibration was tested with MC-dropout — 40 "
     "stochastic forward passes per input, variance as uncertainty "
     "estimate. The weight head sits at 96 percent of true values inside "
     "the 2-sigma band against the 95 percent target — well calibrated. "
     "The displacement head is at 82 percent — under-confident, which "
     "is the safe direction.’")
para(doc,
     "‘Convergence rate analysis. Fitting W(t) = W∞ + A·exp(−t/τ) to "
     "each seed's history gives time constants. PSO at 8.7 generations. "
     "GA at 23.5. GA with LLM warm-start at 14.9 — about halfway "
     "between. So the LLM warm-start mechanism is making GA behave like "
     "PSO in terms of convergence speed.’")
para(doc,
     "‘The hard- versus soft-constraint Pareto front comparison is "
     "important for civil engineering practice. Under hard IS 800 the "
     "72-bar feasible mass sits at 549 lb. Under a representative soft "
     "penalty it drops to around 380 lb. The point is not that one is "
     "right and the other wrong — it is that reported optima in the "
     "truss-sizing literature can be very sensitive to constraint-"
     "handling conventions. Designers using commercial soft-penalty "
     "solvers should run an independent hard-feasibility check before "
     "sign-off.’")

H2(doc, "Part 5 — Impact (15:30 – 17:00)")
para(doc,
     "‘Three practical implications. First, the surrogate-accelerated "
     "stack brings GA runs from 30 minutes down to under 20 seconds — "
     "enabling parametric studies during design meetings, not overnight "
     "batches. Second, the constraint-handling sensitivity should be "
     "flagged to anyone using soft-penalty solvers in IS 800 work. "
     "Third, the Streamlit UI exposes the whole framework without "
     "requiring Python literacy.’")
para(doc,
     "‘On reproducibility: every dependency is pinned, every seed is "
     "logged, every LLM response is cached, all 89 references and 27 "
     "figures are checked in, and a full pytest -m ‘not slow’ suite "
     "passes on every push. Anyone can rebuild the 125-page PDF and "
     "the 35-slide deck with one Tectonic command.’")

H2(doc, "Part 6 — Conclusion (17:00 – 19:00)")
para(doc,
     "‘What I am not claiming: FEM is linear-elastic, sizing-only, "
     "single buckling curve, no PPO transfer learning, LLM is zero-shot "
     "not fine-tuned. These are limitations stated honestly — none of "
     "them are load-bearing for C1 through C5.’")
para(doc,
     "‘Future work spans seven directions: F1 non-linear dynamic FEM for "
     "IS 1893 seismic, F2 topology optimization, F3 GNN surrogate, F4 "
     "cross-instance PPO transfer, F5 fine-tuned domain LLM, F6 PGCIL "
     "transmission-tower validation, F7 independent FEM cross-check of "
     "the 72-bar discrepancy.’")
para(doc,
     "‘To summarise — the five contributions: 10-bar to literature at "
     "0.004 percent, 132× surrogate speedup at R² = 0.9993, a rigorously "
     "feasible 72-bar baseline at 549 lb, LLM warm-start saves 36.8 "
     "percent of generations on 10-bar at p = 0.0046, and the entire "
     "IS 800-compliant pipeline runs under 10 seconds on a laptop CPU "
     "with no API key required to rebuild.’")
para(doc,
     "‘The thesis is 125 pages, 27 figures, 89 references; the deck is "
     "35 slides; everything is tagged phase-10-complete.’")

H2(doc, "Closing (19:00 – 19:30)")
para(doc,
     "‘Thank you for your attention. I would be happy to take your "
     "questions.’")

callout(doc, "TIMING TIPS",
        "If you are running long, COMPRESS Part 4 (deeper analysis). "
        "If you are running short, EXPAND Part 3 with one extra sentence "
        "per result. Never compress Part 1 (problem) — examiners use it "
        "to anchor every subsequent question.")

page_break(doc)

# ============================================================
# SECTION 16: GLOSSARY
# ============================================================
H1(doc, "16. Glossary of Every Technical Term Used")

glossary = [
    ("Adam", "An adaptive learning-rate optimizer for neural networks. Combines momentum with per-parameter learning rates. The default optimizer for almost all deep learning today."),
    ("Buckling", "Sudden sideways failure of a long, thin compression member at a load below pure yield. The buckling load depends on slenderness; IS 800 uses Perry-Robertson curves to compute design strength."),
    ("Constraint handling", "How an optimizer treats designs that violate a constraint. Hard: feasibility flag — infeasible cannot beat feasible. Soft: add penalty to objective. We use hard throughout."),
    ("Crossover (SBX)", "Genetic operator that mixes two parent designs to produce children. Simulated Binary Crossover uses a polynomial distribution with index η_c to control offspring spread."),
    ("Crowding distance", "NSGA-II's secondary sort criterion — measures how isolated a design is in objective space. Higher distance is preferred, to spread the front evenly."),
    ("Dropout", "Regularization technique that randomly zeros activations during training. Prevents overfitting. Can also be used at inference to estimate uncertainty (MC-dropout)."),
    ("Element stiffness matrix", "A small matrix describing one bar's force-displacement relationship. For a truss bar: k_local = (EA/L)·[[1,−1],[−1,1]]. Transformed to global coordinates before assembly."),
    ("Eulerian instability / Euler load", "The classical analytical critical buckling load: P_Euler = π²EI/L². For very long columns f_cc → P_Euler/A. IS 800's curves smoothly transition from yield to Euler."),
    ("FEM", "Finite Element Method. Numerical procedure to solve continuum mechanics by discretizing into elements. For trusses each bar is one element with axial stiffness."),
    ("Feasibility-aware tournament", "Selection rule in EA: any feasible design beats any infeasible design; among feasible, lower fitness wins; among infeasible, lower violation wins."),
    ("Generations", "Iterations of an EA. One generation = evaluate, select, crossover, mutate, replace. We run 500 generations for GA, 300 for NSGA-II."),
    ("Hyperparameter", "A parameter of the algorithm itself, not the design. Examples: population size, crossover η_c, mutation η_m, learning rate, dropout p."),
    ("IS 800:2007", "Bureau of Indian Standards code for general construction in steel. Limit State Design philosophy. Five clauses are implemented: 6.2, 6.3, 7.1, 5.6.1, 3.8."),
    ("Latin Hypercube Sampling (LHS)", "Space-filling sample design. Partitions each dimension into N bins, places one sample per bin, permutes across dimensions. Better marginal coverage than uniform random."),
    ("Mann-Whitney U test", "Non-parametric statistical test for whether two distributions differ. Used because generations-to-convergence is not normally distributed. We report U-test p-values for LLM warm-start."),
    ("MC-dropout", "Monte-Carlo Dropout. Keep dropout active at inference; multiple stochastic forward passes give a distribution. Variance approximates epistemic uncertainty (Gal & Ghahramani 2016)."),
    ("MDP", "Markov Decision Process. The mathematical framework for RL: states, actions, transitions, rewards. PPO trains a policy that operates inside an MDP."),
    ("Metaheuristic", "A high-level, problem-independent search strategy. GA, PSO, simulated annealing, etc. They don't require gradients — only the ability to RANK designs."),
    ("MLP", "Multi-Layer Perceptron. A feedforward neural network with one or more hidden layers, fully connected, with non-linear activations (ReLU, tanh)."),
    ("Mutation", "Genetic operator that randomly perturbs a child. We use polynomial mutation with index η_m = 20 and probability 1/n per gene."),
    ("Non-dominated sort", "NSGA-II's primary sort. Partition population into Pareto fronts: Front 1 dominated by nothing, Front 2 dominated only by Front 1, etc."),
    ("NSGA-II", "Non-dominated Sorting GA II (Deb et al. 2002). Multi-objective EA combining non-dominated sort and crowding distance."),
    ("Pareto front", "The set of non-dominated solutions in multi-objective space. No design beats them on all objectives simultaneously."),
    ("Partition method", "Boundary-condition enforcement by removing rows/columns of fixed DoFs from the stiffness matrix. Cleaner than penalty methods."),
    ("Perry-Robertson", "Empirical buckling curve used in IS 800 §7.1 to compute design compression strength accounting for imperfections and slenderness."),
    ("PPO", "Proximal Policy Optimization (Schulman et al. 2017). On-policy RL algorithm with clipped objective for stable updates."),
    ("PSO", "Particle Swarm Optimization (Kennedy & Eberhart 1995). Population-based metaheuristic where particles drift toward personal-best and global-best positions."),
    ("Pymoo", "Python multi-objective optimization library used for GA, PSO, NSGA-II. Stable, well-maintained, has feasibility-aware tournament built in."),
    ("R²", "Coefficient of determination. Fraction of variance in the target captured by the model. 1.0 is perfect; 0 is no better than mean prediction."),
    ("Reward (RL)", "Scalar feedback to an RL agent per action. We use r = −W − λ·infeasibility, encouraging weight reduction and feasibility."),
    ("Seed", "Initial value for a random number generator. Fixing seeds makes stochastic algorithms reproducible. We use 5–10 seeds per algorithm-benchmark."),
    ("Serviceability", "Limit on deflection under service loads. IS 800 §5.6.1: δ ≤ L/325 for floor beams supporting plaster."),
    ("Slenderness ratio", "λ = KL/r where K is effective length factor, L is unsupported length, r is radius of gyration. IS 800 §3.8 limits λ ≤ 180 for compression."),
    ("Stable-Baselines3 (SB3)", "Production-quality RL library on PyTorch. Provides PPO, DDPG, SAC, etc., with consistent API."),
    ("Stiffness matrix", "Matrix K such that K·u = F for the structure. Assembled from element stiffnesses. Symmetric, sparse for large structures, well-conditioned after partitioning supports."),
    ("Surrogate model", "Approximate, cheap-to-evaluate model that mimics an expensive simulation. Our MLP surrogate replaces FEM in the optimization inner loop."),
    ("Tournament selection", "EA parent-selection rule: pick k random individuals from population, keep best. Tournament size k = 2 or 3 is standard."),
    ("Tractable", "Computationally feasible at a useful problem size. Brute-force is intractable for 10-bar; metaheuristics are tractable."),
    ("Truss", "A structure of straight bars connected at frictionless joints, loaded only at joints. All members carry pure axial force."),
    ("Wall-clock time", "Real elapsed time, as measured by a stopwatch. Distinct from CPU time (which counts core-seconds). We report wall-clock because it is what users experience."),
]

for term, defn in glossary:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(term + ".  ")
    r.font.name = "Calibri"
    r.font.size = Pt(11)
    r.bold = True
    r.font.color.rgb = ACCENT
    r2 = p.add_run(defn)
    r2.font.name = "Calibri"
    r2.font.size = Pt(11)
    r2.font.color.rgb = INK

page_break(doc)

# ============================================================
# SECTION 17: THE FIVE CONTRIBUTIONS
# ============================================================
H1(doc, "17. The Five Contributions (C1–C5) — Memorize These Numbers")

para(doc,
     "If the examiner asks ‘what are your contributions in one minute’, "
     "say these five sentences. Memorize them word-for-word.")

H2(doc, "C1.  End-to-end stack validation")
rich_para(doc,
          [("On the 10-bar planar truss, PSO with 5 seeds converges to ",
            False, False, INK),
           ("5061.05 lb",
            True, False, ACCENT),
           (" against the published optimum of 5060.85 lb (Sunar & Belegundu 1991) — an error of ",
            False, False, INK),
           ("0.004%",
            True, False, ACCENT),
           (". This validates the entire FEM + IS 800 + optimizer stack against the most-cited benchmark.",
            False, False, INK)],
          size=11)

H2(doc, "C2.  Surrogate speedup")
rich_para(doc,
          [("A 3-hidden-layer MLP trained on 10,000 LHS samples achieves ",
            False, False, INK),
           ("R² = 0.9993 on the weight head",
            True, False, ACCENT),
           (" and delivers a ",
            False, False, INK),
           ("132× wall-clock speedup",
            True, False, ACCENT),
           (" — a full 500-generation GA run drops from 12.4 s with FEM to 0.094 s with the surrogate.",
            False, False, INK)],
          size=11)

H2(doc, "C3.  Rigorously-feasible 72-bar baseline")
rich_para(doc,
          [("Under hard IS 800 constraints the 72-bar tower converges to ",
            False, False, INK),
           ("~549 lb with cross-seed spread under 1%",
            True, False, ACCENT),
           (". The gap versus the soft-penalty literature value of ~380 lb (Camp & Bichon 2004; Bekdaş et al. 2015) is flagged as an observation pending independent FEM cross-check (OpenSeesPy / SAP2000) — future work F7.",
            False, False, INK)],
          size=11)

H2(doc, "C4.  LLM warm-start reduces convergence generations")
rich_para(doc,
          [("Seeding the GA with 8 LLM-proposed designs cuts generations-to-convergence by ",
            False, False, INK),
           ("36.8% on 10-bar with p-value 0.0046",
            True, False, ACCENT),
           (". Effect-map: ‒76.6% on 25-bar (underpowered, n=3); no effect on 72-bar (p=0.81). LLM helps when an architectural insight exists.",
            False, False, INK)],
          size=11)

H2(doc, "C5.  Reproducible, deployment-ready pipeline")
rich_para(doc,
          [("The full IS 800-compliant pipeline runs ",
            False, False, INK),
           ("end-to-end in under 10 s on a laptop CPU",
            True, False, ACCENT),
           (" with no API key required — LLM responses are cached. Streamlit UI + FastAPI + thesis PDF all rebuild from the ",
            False, False, INK),
           ("phase-10-complete",
            True, False, ACCENT),
           (" tag with a single Tectonic command.",
            False, False, INK)],
          size=11)

callout(doc, "MEMORIZE THESE FIVE NUMBERS",
        "5061.05 lb (C1)  •  132× and R² = 0.9993 (C2)  •  549 lb hard "
        "vs 380 lb soft (C3)  •  −36.8% with p = 0.0046 (C4)  •  <10 s "
        "end-to-end (C5). If you forget everything else, remember these.")

page_break(doc)

# ============================================================
# SECTION 18: FINAL CHECKLIST
# ============================================================
H1(doc, "18. Final Checklist — The Day of the Viva")

H2(doc, "The night before")
bullet(doc, "Re-read Sections 8–13 of this guide (slide-by-slide walkthrough).")
bullet(doc, "Rehearse Section 15 (the 20-minute script) ALOUD. Time yourself. Aim for 18–19 minutes.")
bullet(doc, "Print Section 14 (Q&A bank). Read each Q, cover the A, and try answering aloud.")
bullet(doc, "Memorize the five C-numbers in Section 17.")
bullet(doc, "Sleep 7+ hours. Tired is worse than under-prepared.")

H2(doc, "60 minutes before")
bullet(doc, "Eat a normal breakfast/lunch. No caffeine experiments today.")
bullet(doc, "Open the deck and the Streamlit demo. Confirm both work.")
bullet(doc, "Open this guide on your phone — for last-minute number lookups.")
bullet(doc, "Test microphone / projector / pointer.")
bullet(doc, "Bring a printed copy of the title slide and the C1–C5 numbers — physical backup.")

H2(doc, "During the talk")
bullet(doc, "Speak slowly. Pause after each contribution number.")
bullet(doc, "Make eye contact with each committee member at least twice.")
bullet(doc, "If you forget a number, say ‘the exact figure is in Chapter 4’ — never guess.")
bullet(doc, "Don't apologize for limitations. State them confidently — limitations are not weaknesses.")
bullet(doc, "If a slide feels long, skip its details and move on; the appendix has them.")

H2(doc, "During Q&A")
bullet(doc, "Repeat the question back in your own words before answering. Gives you 5 seconds to think and confirms you understood.")
bullet(doc, "If you don't know — say so. ‘I don't have that result, but I would expect X because Y.’ Honest beats wrong.")
bullet(doc, "Defend the limitations as DELIBERATE choices, not omissions. Single buckling curve, sizing-only, linear elastic — all chosen so the rest of the framework could be rigorous.")
bullet(doc, "For the 72-bar question (it WILL come up): ‘The literature uses soft penalties; we use hard constraints. The 169 lb gap is a constraint-handling sensitivity, not a coding error. F7 proposes independent FEM cross-check.’")
bullet(doc, "Smile. The committee wants you to pass — they are looking for confidence, not omniscience.")

H2(doc, "If something goes wrong")
bullet(doc, "Projector dies: have the PDF on a USB stick. Have the deck on your phone.")
bullet(doc, "Streamlit demo fails: skip the live demo, move to the next slide. Don't debug live.")
bullet(doc, "You go blank on a number: ‘The exact value is in Chapter 4, Table 4.3 — let me move on and come back if there's time.’")
bullet(doc, "Hostile question: stay calm, restate it in your own words, give the best answer you have, then say ‘I'd be happy to explore this further with the committee after the viva.’")

H2(doc, "After the viva")
bullet(doc, "Thank every committee member by name.")
bullet(doc, "Thank Dr. Pathak (supervisor) by name. Wait for him to leave first.")
bullet(doc, "Don't post-mortem the talk in your head for at least 24 hours. You did fine.")

callout(doc, "ONE LAST THING",
        "You wrote 125 pages, ran four benchmarks, implemented an entire "
        "AI framework from scratch, and you can explain it in 20 minutes. "
        "That IS a thesis. Walk in confident. You've already done the hard "
        "part — now just describe what you built.")

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("— END OF GUIDE —")
r.font.name = "Calibri"
r.font.size = Pt(12)
r.italic = True
r.font.color.rgb = MUTED

# ============================================================
# SAVE
# ============================================================
out_path = "/home/user/thesis/viva_prep/Viva_Complete_Guide.docx"
doc.save(out_path)
print(f"Wrote {out_path}")
