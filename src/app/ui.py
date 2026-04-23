"""
Streamlit UI for the thesis demo. Run with:

    ./venv/bin/streamlit run src/app/ui.py

Talks directly to the optimization stack in-process (no FastAPI needed
for the demo — simpler setup, identical results). The FastAPI layer in
`src/app/api.py` is there for programmatic access and Chapter-5 text.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.algorithms.runner import run
from src.benchmarks.registry import available_benchmarks, get_benchmark
from src.llm.designer import suggest_initial_design


st.set_page_config(page_title="AI Truss Optimizer", layout="wide")
st.title("AI-Powered Steel Truss Optimization")
st.caption("M.Tech Thesis — 8-layer stack: FEM + GA/PSO/NSGA-II + Surrogate + RL + LLM")


# ---------- sidebar controls ----------

with st.sidebar:
    st.header("Configuration")
    available = available_benchmarks()
    bench_name = st.selectbox("Benchmark", available, index=available.index("10bar"))
    algo = st.selectbox("Algorithm", ["ga", "pso", "nsga2"], index=0)
    seed = st.number_input("Random seed", value=42, step=1)
    pop_size = st.slider("Population size", 20, 200, 60, step=10)
    n_gen = st.slider("Generations", 20, 500, 150, step=10)
    use_llm = st.checkbox("LLM warm-start (Phase 4)", value=False)
    run_btn = st.button("Run optimization", type="primary")


# ---------- benchmark info panel ----------

try:
    bench = get_benchmark(bench_name)
    col1, col2, col3 = st.columns(3)
    col1.metric("Design variables", bench.n_design_vars)
    col2.metric("Members", bench.n_bars)
    col3.metric("Literature optimum", f"{bench.reference_optimum_weight:.2f}")
    st.caption(f"Reference: {bench.reference_source} · units: {bench.units}")
except NotImplementedError as exc:
    st.warning(f"Benchmark '{bench_name}' is not yet encoded ({exc}).")
    st.stop()


# ---------- run optimization ----------

if run_btn:
    x0 = None
    if use_llm:
        with st.spinner("Getting LLM warm-start..."):
            s = suggest_initial_design(bench)
            x0 = s.x
        st.info(f"**LLM source:** {s.source} — {s.reasoning}")

    with st.spinner(f"Running {algo.upper()} (pop={pop_size}, gens={n_gen})..."):
        t0 = time.perf_counter()
        result = run(algo, bench, seed=int(seed), pop_size=pop_size, n_gen=n_gen, x0=x0)
        wall = time.perf_counter() - t0

    lit = bench.reference_optimum_weight
    err_pct = 100.0 * (result.best_weight - lit) / lit

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best weight", f"{result.best_weight:.2f}", f"{err_pct:+.2f}% vs lit")
    c2.metric("Feasible?", "yes" if result.feasible else "NO")
    c3.metric("Max stress", f"{result.max_stress:.1f}")
    c4.metric("Max disp", f"{result.max_displacement:.4f}")

    c5, c6 = st.columns(2)
    c5.metric("Wall time", f"{wall:.2f} s")
    c6.metric("FEM evals", f"{result.n_evals:,}")

    # Convergence plot
    if result.history:
        df_hist = pd.DataFrame(result.history)
        fig = px.line(
            df_hist,
            x="gen",
            y=["best_weight", "mean_weight"],
            title="Convergence curve",
            labels={"value": "weight", "gen": "generation", "variable": ""},
        )
        fig.add_hline(y=lit, line_dash="dash", line_color="green",
                      annotation_text=f"literature optimum {lit:.1f}")
        st.plotly_chart(fig, use_container_width=True)

    # Pareto front (NSGA-II)
    if result.pareto_f is not None:
        df_p = pd.DataFrame(result.pareto_f, columns=["weight", "max_disp"])
        fig = px.scatter(
            df_p,
            x="weight",
            y="max_disp",
            title="Pareto front: weight vs max displacement",
            labels={"weight": "weight", "max_disp": "max |displacement|"},
        )
        fig.add_hline(
            y=bench.displacement_limit,
            line_dash="dash",
            line_color="red",
            annotation_text=f"disp limit {bench.displacement_limit}",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Final design table
    if result.best_x is not None:
        df_x = pd.DataFrame(
            {
                "group": np.arange(bench.n_design_vars),
                "area": result.best_x,
            }
        )
        st.subheader("Best design (per group)")
        st.dataframe(df_x, use_container_width=True)

    # Truss geometry preview (2-D benchmarks only)
    if bench.ndim == 2:
        st.subheader("Truss geometry")
        xs = bench.nodes[:, 0]
        ys = bench.nodes[:, 1]
        fig = go.Figure()
        x_full = bench.expand_design(result.best_x) if result.best_x is not None else np.ones(bench.n_bars)
        max_area = float(x_full.max()) if x_full.max() > 0 else 1.0
        for i, (n1, n2) in enumerate(bench.connectivity):
            fig.add_trace(
                go.Scatter(
                    x=[xs[n1], xs[n2]],
                    y=[ys[n1], ys[n2]],
                    mode="lines",
                    line={"width": 2 + 10 * x_full[i] / max_area, "color": "steelblue"},
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=f"bar {i}: A={x_full[i]:.3f}",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers+text",
                marker={"size": 10, "color": "black"},
                text=[str(i) for i in range(len(xs))],
                textposition="top center",
                showlegend=False,
            )
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(title="Member thickness proportional to area")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Set parameters and click **Run optimization**.")
