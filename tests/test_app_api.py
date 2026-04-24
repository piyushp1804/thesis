"""
FastAPI endpoint tests via starlette TestClient — no server needed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.app.api import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_benchmarks_contains_10bar():
    r = client.get("/benchmarks")
    assert r.status_code == 200
    assert "10bar" in r.json()


def test_benchmark_info_10bar():
    r = client.get("/benchmarks/10bar")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "10-bar planar"
    assert body["n_design_vars"] == 10
    assert body["units"] == "imperial"
    assert body["reference_optimum_weight"] > 5000


def test_benchmark_info_200bar():
    # Phase 9: 200-bar stepped-tower is now fully implemented.
    # reference_verified=False (geometry ours, not Kaveh's exact coords).
    r = client.get("/benchmarks/200bar")
    assert r.status_code == 200
    body = r.json()
    assert body["n_design_vars"] == 29
    assert body["n_bars"] == 200
    assert body["units"] == "SI"


def test_benchmark_info_unknown():
    r = client.get("/benchmarks/doesnotexist")
    assert r.status_code == 404


def test_optimize_ga_10bar_minimal():
    """Smoke-test: run tiny GA, response must be well-formed and feasible."""
    payload = {
        "algorithm": "ga",
        "benchmark": "10bar",
        "seed": 42,
        "pop_size": 30,
        "n_gen": 40,
        "use_llm_warmstart": False,
    }
    r = client.post("/optimize", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["x"]) == 10
    assert body["best_weight"] > 0
    assert "convergence_curve" in body
    assert body["pareto_front"] is None


def test_optimize_nsga2_returns_pareto():
    payload = {
        "algorithm": "nsga2",
        "benchmark": "10bar",
        "seed": 42,
        "pop_size": 30,
        "n_gen": 30,
    }
    r = client.post("/optimize", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["pareto_front"] is not None
    assert len(body["pareto_front"]) >= 5


def test_llm_suggest_offline_is_valid():
    r = client.post("/llm/suggest", params={"benchmark": "10bar"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["x"]) == 10
    assert body["source"] in {"anthropic", "cache", "heuristic"}
