"""
High-level 'LLM designer' API.

    x0 = suggest_initial_design(benchmark)

Either gets a cached Claude reply, calls Claude fresh, or falls back to
the load-path heuristic — whichever the client layer decides. Always
returns a valid design vector inside `benchmark.area_bounds`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.benchmarks.base import BenchmarkProblem

from .client import call_claude, heuristic_design
from .prompts import SYSTEM_PROMPT, build_user_prompt


@dataclass
class DesignerSuggestion:
    x: np.ndarray
    source: str              # "anthropic" | "cache" | "heuristic"
    reasoning: str
    confidence: float


def suggest_initial_design(
    benchmark: BenchmarkProblem,
    use_cache: bool = True,
) -> DesignerSuggestion:
    """Return a warm-start design for `benchmark`."""
    system = SYSTEM_PROMPT
    user = build_user_prompt(benchmark)
    resp = call_claude(system=system, user=user, use_cache=use_cache)

    lo, hi = benchmark.area_bounds
    nvars = benchmark.n_design_vars

    if resp.source in ("anthropic", "cache"):
        areas = resp.parsed.get("areas")
        if isinstance(areas, list) and len(areas) == nvars:
            x = np.clip(np.asarray(areas, dtype=float), lo, hi)
            return DesignerSuggestion(
                x=x,
                source=resp.source,
                reasoning=str(resp.parsed.get("reasoning", "")),
                confidence=float(resp.parsed.get("confidence", 0.5)),
            )
        # Claude's reply was malformed -> fall through to heuristic.

    x = heuristic_design(benchmark)
    return DesignerSuggestion(
        x=x,
        source="heuristic",
        reasoning="Load-path weighted heuristic (offline fallback).",
        confidence=0.4,
    )
