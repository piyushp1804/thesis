"""
Tests for the LLM designer layer.

We do NOT require a live API key; every test passes offline by falling
through to the heuristic designer. When a real key is configured the
same tests exercise the cache path.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.registry import get_benchmark
from src.llm.client import heuristic_design
from src.llm.designer import suggest_initial_design
from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt


def test_heuristic_in_bounds():
    b = get_benchmark("10bar")
    x = heuristic_design(b)
    lo, hi = b.area_bounds
    assert x.shape == (b.n_design_vars,)
    assert (x >= lo).all()
    assert (x <= hi).all()


def test_prompt_contains_key_numbers():
    b = get_benchmark("10bar")
    prompt = build_user_prompt(b)
    assert "10-bar" in prompt
    assert str(b.E) in prompt
    assert str(b.reference_optimum_weight) in prompt
    assert "areas" in prompt
    assert "reasoning" in prompt


def test_suggest_returns_valid_vector_offline():
    b = get_benchmark("10bar")
    s = suggest_initial_design(b, use_cache=True)
    lo, hi = b.area_bounds
    assert s.x.shape == (b.n_design_vars,)
    assert (s.x >= lo).all()
    assert (s.x <= hi).all()
    # Offline: source must be one of cache/heuristic/anthropic depending on
    # environment; all three are acceptable contract-wise.
    assert s.source in {"cache", "heuristic", "anthropic"}


def test_heuristic_is_non_uniform():
    """The load-path heuristic should not return the uniform-mean design."""
    b = get_benchmark("10bar")
    x = heuristic_design(b)
    lo, hi = b.area_bounds
    uniform = 0.5 * (lo + hi)
    # heuristic must deviate meaningfully from the uniform vector
    assert np.max(np.abs(x - uniform)) > 0.05 * (hi - lo)
