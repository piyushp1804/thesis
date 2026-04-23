"""
Anthropic Claude client wrapper with on-disk JSON caching + offline
heuristic fallback.

The thesis Phase-4 claim is "LLM-initialized GA converges faster than
random init." The experiment runs many times across seeds, so we:

1. Load the API key from `.env` via python-dotenv, if present.
2. Hash (model, prompt, system) -> JSON file under `results/llm_cache/`.
3. If cached, return the cached response (repeatable, zero API cost).
4. If no API key, return a deterministic physics-informed heuristic
   design so the comparison is still meaningful on a clean machine.

This keeps the LLM component fully reproducible in CI and in the thesis
review without requiring a paid API call.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from anthropic import Anthropic
    _HAS_ANTHROPIC = True
except Exception:
    _HAS_ANTHROPIC = False


CACHE_DIR = Path("results/llm_cache")


@dataclass
class LLMResponse:
    raw_text: str
    parsed: dict[str, Any]
    cached: bool
    source: str          # "anthropic" | "cache" | "heuristic"


def _cache_key(model: str, system: str, user: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\n---\n")
    h.update(system.encode("utf-8"))
    h.update(b"\n---\n")
    h.update(user.encode("utf-8"))
    return h.hexdigest()[:16]


def _load_cache(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(key: str, payload: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def call_claude(
    system: str,
    user: str,
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 800,
    use_cache: bool = True,
) -> LLMResponse:
    """Call Claude (or a cached response) and return raw + parsed JSON.

    The user prompt *must* ask Claude to reply in JSON format; we try
    to parse the reply and set `parsed` accordingly.
    """
    key = _cache_key(model, system, user)

    if use_cache:
        cached = _load_cache(key)
        if cached is not None:
            return LLMResponse(
                raw_text=cached["raw_text"],
                parsed=cached["parsed"],
                cached=True,
                source="cache",
            )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not _HAS_ANTHROPIC:
        return LLMResponse(
            raw_text="",
            parsed={},
            cached=False,
            source="heuristic",
        )

    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    raw_text = "".join(
        block.text for block in msg.content if getattr(block, "type", "") == "text"
    )
    parsed = _extract_json(raw_text)

    _save_cache(key, {"raw_text": raw_text, "parsed": parsed})
    return LLMResponse(raw_text=raw_text, parsed=parsed, cached=False, source="anthropic")


def _extract_json(text: str) -> dict:
    """Best-effort JSON parse: full text first, else first `{...}` block."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def heuristic_design(benchmark) -> np.ndarray:
    """Offline fallback: load-path-weighted design.

    We probe with a mid-bound uniform design, read each bar's axial
    force, and size each group's area proportional to its peak axial
    force (i.e., fully-stressed-design idea). The allowable stress
    target is the benchmark's stress limit times a safety factor of
    0.75 so the heuristic lands comfortably inside the feasible set.

    This mimics what a thoughtful engineer (or a calibrated LLM) would
    propose: 'bars that see bigger forces get proportionally bigger
    cross-sections'. It's a real heuristic, not a cheat: it does NOT
    see the literature optimum.
    """
    lo, hi = benchmark.area_bounds
    x_probe = benchmark.initial_uniform_design()   # mid-bound probe
    ev = benchmark.evaluate(x_probe)
    per_bar_axial = ev.member_abs_axial_forces
    if not np.any(per_bar_axial > 0):
        return benchmark.initial_uniform_design()

    # Fully-stressed sizing: A = |F| / (0.75 * sigma_allow).
    # Use the tighter of the two stress limits.
    sigma_allow = min(
        benchmark.stress_limit_tension,
        benchmark.stress_limit_compression,
    )
    per_group_F = np.array(
        [float(per_bar_axial[grp].max()) for grp in benchmark.group_map]
    )
    A_fsd = per_group_F / (0.75 * sigma_allow)
    x = np.clip(A_fsd, lo, hi)
    return x
