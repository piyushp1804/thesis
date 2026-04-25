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

import os
from pathlib import Path
from typing import Any, Literal

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.algorithms.runner import run
from src.benchmarks.registry import available_benchmarks, get_benchmark
from src.llm.designer import suggest_initial_design


app = FastAPI(
    title="AI Truss Optimization API",
    version="1.0.0",
    description=(
        "FastAPI wrapper around the M.Tech thesis optimization stack "
        "(FEM + GA/PSO/NSGA-II + Surrogate + RL + LLM warm-start, "
        "IS 800:2007 compliant). See / for the thesis landing page or "
        "/docs for the interactive Swagger UI."
    ),
)

# Repo root is two levels up from src/app/api.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_FIGURES_DIR = _REPO_ROOT / "thesis_writeup" / "figures"
_THESIS_PDF = _REPO_ROOT / "thesis_writeup" / "main.pdf"
_SLIDES_PDF = _REPO_ROOT / "thesis_writeup" / "slides" / "main.pdf"
_LLM_CACHE_DIR = _REPO_ROOT / "results" / "llm_cache"

if _FIGURES_DIR.is_dir():
    app.mount("/figures", StaticFiles(directory=str(_FIGURES_DIR)), name="figures")


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


# ---------- landing page + meta endpoints ----------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page() -> HTMLResponse:
    path = _TEMPLATES_DIR / "landing.html"
    if not path.is_file():
        return HTMLResponse(
            content="<h1>AI Truss Optimization API</h1><p>See <a href='/docs'>/docs</a>.</p>",
        )
    return HTMLResponse(content=path.read_text(encoding="utf-8"))


@app.get("/status")
def status() -> dict[str, Any]:
    """Single-shot metadata for the landing page dashboard."""
    cache_count = 0
    if _LLM_CACHE_DIR.is_dir():
        cache_count = sum(1 for _ in _LLM_CACHE_DIR.glob("*.json"))
    return {
        "service": app.title,
        "version": app.version,
        "thesis_tag": "phase-10-complete",
        "thesis_pages": 125,
        "slides_pages": 23,
        "citations": 89,
        "figures": 27,
        "benchmarks": available_benchmarks(),
        "anthropic_key_loaded": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "llm_cache_count": cache_count,
        "llm_cache_dir": str(_LLM_CACHE_DIR.relative_to(_REPO_ROOT)) if _LLM_CACHE_DIR.exists() else None,
    }


@app.get("/thesis.pdf", include_in_schema=False)
def thesis_pdf() -> FileResponse:
    if not _THESIS_PDF.is_file():
        raise HTTPException(status_code=404, detail="thesis PDF not built; run `tectonic thesis_writeup/main.tex`")
    return FileResponse(
        str(_THESIS_PDF),
        media_type="application/pdf",
        filename="aryan_thesis.pdf",
    )


@app.get("/slides.pdf", include_in_schema=False)
def slides_pdf() -> FileResponse:
    if not _SLIDES_PDF.is_file():
        raise HTTPException(status_code=404, detail="slides PDF not built; run `tectonic thesis_writeup/slides/main.tex`")
    return FileResponse(
        str(_SLIDES_PDF),
        media_type="application/pdf",
        filename="aryan_thesis_viva_slides.pdf",
    )


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
