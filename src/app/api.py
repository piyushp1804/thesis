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
from typing import Any, Literal

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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
        "IS 800:2007 compliant). See / for a landing page or /docs "
        "for the interactive Swagger UI."
    ),
)


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


_LANDING_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>AI Truss Optimization API</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  :root {{ color-scheme: dark light; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 860px; margin: 2.5rem auto; padding: 0 1.25rem;
    line-height: 1.55; color: #e7e9ee; background: #0f1115;
  }}
  a {{ color: #7aa7ff; }}
  a:hover {{ color: #a9c4ff; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 6px;
    background: #1d2330; color: #89f0b7; font-size: 0.82rem;
    margin-left: 0.3rem;
  }}
  .badge.warn {{ color: #f7c78c; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0 1.3rem; }}
  th, td {{
    border-bottom: 1px solid #2a3040; padding: 6px 10px; text-align: left;
    font-size: 0.93rem;
  }}
  th {{ color: #a8b4cc; font-weight: 600; }}
  code, pre {{
    background: #181c25; border: 1px solid #262c3a;
    border-radius: 6px; padding: 2px 6px;
  }}
  pre {{ padding: 10px 12px; overflow-x: auto; font-size: 0.86rem; }}
  .tiles {{
    display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr));
    gap: 10px; margin: 1rem 0 1.4rem;
  }}
  .tile {{
    background: #161b25; border: 1px solid #262c3a; padding: 12px 14px;
    border-radius: 10px; text-decoration: none; color: #dfe3ec;
  }}
  .tile:hover {{ border-color: #3a4562; }}
  .tile strong {{ display: block; color: #7aa7ff; margin-bottom: 4px; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .sub {{ color: #8893ac; margin-top: 0; }}
</style></head><body>

<h1>AI Truss Optimization API <span class="badge">v{version}</span></h1>
<p class="sub">
  M.Tech thesis stack: FEM + GA/PSO/NSGA-II + neural surrogate + PPO
  + LLM warm-start, IS&nbsp;800:2007 compliance.
  Anthropic key: <span class="badge {key_cls}">{key_state}</span>
</p>

<div class="tiles">
  <a class="tile" href="/docs"><strong>Swagger UI &rarr;</strong>Interactive playground</a>
  <a class="tile" href="/redoc"><strong>ReDoc &rarr;</strong>Readable API reference</a>
  <a class="tile" href="/health"><strong>/health</strong>Liveness probe</a>
  <a class="tile" href="/benchmarks"><strong>/benchmarks</strong>List of benchmarks</a>
</div>

<h3>Endpoints</h3>
<table>
  <tr><th>Method</th><th>Path</th><th>Purpose</th></tr>
  <tr><td>GET</td><td><a href="/health"><code>/health</code></a></td><td>Liveness check</td></tr>
  <tr><td>GET</td><td><a href="/benchmarks"><code>/benchmarks</code></a></td><td>List benchmarks</td></tr>
  <tr><td>GET</td><td><code>/benchmarks/{{name}}</code></td><td>Metadata for one benchmark</td></tr>
  <tr><td>POST</td><td><code>/optimize</code></td><td>Run GA/PSO/NSGA-II, return best design + history</td></tr>
  <tr><td>POST</td><td><code>/llm/suggest</code></td><td>Claude (or cached / heuristic) warm-start design</td></tr>
</table>

<h3>Quick test</h3>
<pre>curl -s http://127.0.0.1:8000/benchmarks/10bar | jq
curl -s -X POST http://127.0.0.1:8000/optimize \\
  -H "Content-Type: application/json" \\
  -d '{{"algorithm":"pso","benchmark":"10bar","seed":42,"pop_size":60,"n_gen":100}}'</pre>

<p style="color:#8893ac;font-size:0.88rem;margin-top:2rem">
  Prefer a graphical interface? Run the Streamlit companion UI:
  <code>python scripts/run_ui.py</code> &rarr;
  <a href="http://localhost:8501">http://localhost:8501</a>
</p>

</body></html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page() -> HTMLResponse:
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    html = _LANDING_HTML.format(
        version=app.version,
        key_state="loaded from .env" if has_key else "not set (heuristic fallback)",
        key_cls="" if has_key else "warn",
    )
    return HTMLResponse(content=html)


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
