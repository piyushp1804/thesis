"""
Prompt construction for the LLM truss designer.

The prompt gives Claude:
  * the benchmark's geometry summary (nodes + element connectivity)
  * material and constraint bounds
  * literature reference (so it can recall published patterns)
  * output-format spec (JSON only)

Claude is asked to return one design-vector guess plus reasoning, which
we parse and feed in as the first GA individual (warm-start).
"""

from __future__ import annotations

from src.benchmarks.base import BenchmarkProblem


SYSTEM_PROMPT = """You are a senior structural engineer with deep expertise
in steel truss optimization (Sunar-Belegundu, Schmit-Miura, Erbatur, Kaveh
benchmarks). Respond ONLY in the JSON format described, no prose, no
code fences. Numbers must be floats within the stated bounds."""


def build_user_prompt(benchmark: BenchmarkProblem) -> str:
    """Return a plain-text user prompt for `benchmark`."""
    lo, hi = benchmark.area_bounds
    units = benchmark.units
    nvars = benchmark.n_design_vars
    conn_summary = _summarize_connectivity(benchmark)

    return f"""TASK: propose a promising initial cross-sectional area design
for the {benchmark.name} truss (reference: {benchmark.reference_source}).

PROBLEM DATA:
- number of design variables: {nvars} (one area per symmetry group)
- area bounds                : [{lo}, {hi}] ({_area_units(units)})
- material E                 : {benchmark.E}
- density                    : {benchmark.density}
- stress limit (tens/comp)   : {benchmark.stress_limit_tension} / {benchmark.stress_limit_compression}
- displacement limit         : {benchmark.displacement_limit}
- units                      : {units}
- published optimum weight   : {benchmark.reference_optimum_weight}

TOPOLOGY SUMMARY:
{conn_summary}

OUTPUT — reply with a single JSON object with fields:
- "areas": list of {nvars} float numbers in [{lo}, {hi}]
- "reasoning": one sentence explaining which members carry the primary
  load path and why you sized them larger.
- "confidence": float in [0, 1].
"""


def _area_units(units: str) -> str:
    return "in^2" if units == "imperial" else "m^2"


def _summarize_connectivity(benchmark: BenchmarkProblem) -> str:
    """Compact English summary of truss topology + loads."""
    n_nodes = benchmark.nodes.shape[0]
    n_bars = benchmark.n_bars
    # Supports.
    sup_nodes = [n for n, _ in benchmark.supports]
    # Load cases.
    lc_lines = []
    for i, lc in enumerate(benchmark.load_cases):
        load_nodes = sorted(lc.nodal_forces.keys())
        lc_lines.append(f"  LC{i+1}: forces at nodes {load_nodes}")
    return (
        f"- {n_nodes} nodes, {n_bars} bars, {benchmark.ndim}-D.\n"
        f"- supports at nodes: {sup_nodes}\n"
        + "\n".join(lc_lines)
    )
