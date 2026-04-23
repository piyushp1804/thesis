"""
FastAPI wrapper around the optimization stack.

Endpoints:
  GET  /health                        -- liveness check
  GET  /benchmarks                    -- list available benchmarks
  GET  /benchmarks/{name}             -- metadata for one benchmark
  POST /optimize                      -- run optimization, return result
  POST /llm/suggest                   -- return LLM (or heuristic) warm-start

Kept deliberately small so the thesis demo runs both the API and the
Streamlit UI on one laptop without extra infra.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.algorithms.runner import run
from src.benchmarks.registry import available_benchmarks, get_benchmark
from src.llm.designer import suggest_initial_design


app = FastAPI(title="AI Truss Optimization API", version="1.0.0")


# ---------- request / response models ----------


class OptimizeRequest(BaseModel):
    algorithm: Literal["ga", "pso", "nsga2"]
    benchmark: str
    seed: int = 42
    pop_size: int = 60
    n_gen: int = 200
    use_llm_warmstart: bool = False


class OptimizeResponse(BaseModel):
    best_weight: float
    feasible: bool
    max_stress: float
    max_displacement: float
    x: list[float]
    wall_time_s: float
    n_evals: int
    convergence_curve: list[dict[str, Any]]
    pareto_front: list[dict[str, float]] | None = None


class BenchmarkInfo(BaseModel):
    name: str
    n_design_vars: int
    n_bars: int
    units: str
    reference_optimum_weight: float
    reference_source: str
    area_bounds: list[float]


class SuggestResponse(BaseModel):
    x: list[float]
    source: str
    reasoning: str
    confidence: float


# ---------- handlers ----------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/benchmarks")
def list_benchmarks() -> list[str]:
    return available_benchmarks()


@app.get("/benchmarks/{name}", response_model=BenchmarkInfo)
def benchmark_info(name: str) -> BenchmarkInfo:
    try:
        b = get_benchmark(name)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown benchmark: {name}")
    return BenchmarkInfo(
        name=b.name,
        n_design_vars=b.n_design_vars,
        n_bars=b.n_bars,
        units=b.units,
        reference_optimum_weight=b.reference_optimum_weight,
        reference_source=b.reference_source,
        area_bounds=list(b.area_bounds),
    )


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    try:
        bench = get_benchmark(req.benchmark)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown benchmark: {req.benchmark}")

    x0 = None
    if req.use_llm_warmstart:
        x0 = suggest_initial_design(bench).x

    kwargs: dict[str, Any] = {
        "seed": req.seed,
        "pop_size": req.pop_size,
        "n_gen": req.n_gen,
    }
    if x0 is not None and req.algorithm in {"ga", "pso"}:
        kwargs["x0"] = x0
    result = run(req.algorithm, bench, **kwargs)

    pareto = None
    if result.pareto_f is not None:
        pareto = [
            {"weight": float(f[0]), "max_disp": float(f[1])}
            for f in result.pareto_f
        ]

    return OptimizeResponse(
        best_weight=float(result.best_weight),
        feasible=bool(result.feasible),
        max_stress=float(result.max_stress),
        max_displacement=float(result.max_displacement),
        x=[float(v) for v in (result.best_x if result.best_x is not None else np.zeros(bench.n_design_vars))],
        wall_time_s=float(result.wall_time_s),
        n_evals=int(result.n_evals),
        convergence_curve=result.history,
        pareto_front=pareto,
    )


@app.post("/llm/suggest", response_model=SuggestResponse)
def llm_suggest(benchmark: str) -> SuggestResponse:
    try:
        bench = get_benchmark(benchmark)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    s = suggest_initial_design(bench)
    return SuggestResponse(
        x=[float(v) for v in s.x],
        source=s.source,
        reasoning=s.reasoning,
        confidence=s.confidence,
    )
