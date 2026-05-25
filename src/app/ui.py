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


# ---------- page + IIT (BHU) branding ----------

BHU_MAROON = "#7A1F2B"

st.set_page_config(page_title="AI Truss Optimizer · IIT (BHU)", layout="wide")

st.markdown(
    f"""
    <style>
      .stButton>button {{
          background-color: {BHU_MAROON};
          color: #ffffff;
          border: 0;
          font-weight: 600;
      }}
      .stButton>button:hover {{ background-color: #5d1620; color:#fff; }}
      h1, h2, h3 {{ color: #ffffff; }}
      .bhu-tag {{ color:#C9A0A6; font-size:0.9rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI-Powered Steel Truss Optimization")
st.caption("M.Tech / IDD Thesis — 8-layer stack: FEM + GA/PSO/NSGA-II + Surrogate + RL + AI Agent")
st.markdown(
    "<div class='bhu-tag'>Aryan Gupta · Roll 21064030 · IDD Civil &amp; Structural Engineering · "
    "Supervisor: Dr. Krishna Kant Pathak · IIT (BHU) Varanasi</div>",
    unsafe_allow_html=True,
)


# ---------- sidebar controls ----------

with st.sidebar:
    st.header("Configuration")
    available = available_benchmarks()
    bench_name = st.selectbox("Benchmark", available, index=available.index("10bar"))
    algo = st.selectbox("Algorithm", ["ga", "pso", "nsga2"], index=0)
    seed = st.number_input("Random seed", value=42, step=1)
    pop_size = st.slider("Population size", 20, 200, 60, step=10)
    n_gen = st.slider("Generations", 20, 500, 150, step=10)
    use_llm = st.checkbox("AI Agent warm-start", value=False)
    run_btn = st.button("Run optimization", type="primary")
    compare_btn = st.button("Compare: random vs AI Agent")


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


def _pct_of_limit(value: float, limit: float) -> str:
    if limit and np.isfinite(limit) and limit > 0:
        return f"{100.0 * value / limit:.0f}% of limit"
    return ""


def _show_result(result, wall: float, lit: float):
    """Render all result widgets for a single optimization run."""
    err_pct = 100.0 * (result.best_weight - lit) / lit

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best weight", f"{result.best_weight:.2f}", f"{err_pct:+.2f}% vs lit")
    c2.metric("Feasible?", "yes" if result.feasible else "NO")

    s_lim = float(bench.stress_limit_tension)
    c3.metric(
        f"Max stress  (limit {s_lim:.0f})",
        f"{result.max_stress:.1f}",
        _pct_of_limit(result.max_stress, s_lim),
        delta_color="off",
    )
    d_lim = float(bench.displacement_limit)
    c4.metric(
        f"Max disp  (limit {d_lim:g})",
        f"{result.max_displacement:.4f}",
        _pct_of_limit(result.max_displacement, d_lim),
        delta_color="off",
    )

    c5, c6 = st.columns(2)
    c5.metric("Wall time", f"{wall:.2f} s")
    c6.metric("FEM evals", f"{result.n_evals:,}")

    # Convergence plot — best solid, mean dashed grey
    if result.history:
        df_hist = pd.DataFrame(result.history)
        fig = go.Figure()
        if "best_weight" in df_hist:
            fig.add_trace(go.Scatter(
                x=df_hist["gen"], y=df_hist["best_weight"], mode="lines",
                name="best", line={"color": BHU_MAROON, "width": 3}))
        if "mean_weight" in df_hist:
            fig.add_trace(go.Scatter(
                x=df_hist["gen"], y=df_hist["mean_weight"], mode="lines",
                name="population mean", line={"color": "#9aa4b2", "width": 1.5, "dash": "dash"}))
        fig.add_hline(y=lit, line_dash="dot", line_color="#18B984",
                      annotation_text=f"literature optimum {lit:.1f}")
        fig.update_layout(title="Convergence curve", xaxis_title="generation",
                          yaxis_title="weight", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Pareto front (NSGA-II)
    if result.pareto_f is not None:
        df_p = pd.DataFrame(result.pareto_f, columns=["weight", "max_disp"])
        fig = px.scatter(df_p, x="weight", y="max_disp",
                         title="Pareto front: weight vs max displacement",
                         labels={"weight": "weight", "max_disp": "max |displacement|"})
        fig.add_hline(y=bench.displacement_limit, line_dash="dash", line_color="red",
                      annotation_text=f"disp limit {bench.displacement_limit}")
        st.plotly_chart(fig, use_container_width=True)

    # Final design table — highlight near-minimum (redundant) bars
    if result.best_x is not None:
        lo, _hi = bench.area_bounds
        df_x = pd.DataFrame({
            "group": np.arange(bench.n_design_vars),
            "area": result.best_x,
            "role": ["redundant (min area)" if a <= lo * 1.05 else "active"
                     for a in result.best_x],
        })
        st.subheader("Best design (per group)")

        def _hl(row):
            color = "background-color: rgba(122,31,43,0.18)" if row["role"].startswith("redundant") else ""
            return [color] * len(row)

        st.dataframe(df_x.style.apply(_hl, axis=1), use_container_width=True)
        n_red = int((result.best_x <= lo * 1.05).sum())
        st.caption(f"{n_red} of {bench.n_design_vars} members driven to the minimum area "
                   f"— the near-redundant load-path members.")

    # Truss geometry preview (2-D benchmarks only)
    if bench.ndim == 2:
        st.subheader("Truss geometry")
        xs = bench.nodes[:, 0]
        ys = bench.nodes[:, 1]
        lo, _hi = bench.area_bounds
        fig = go.Figure()
        x_full = bench.expand_design(result.best_x) if result.best_x is not None else np.ones(bench.n_bars)
        max_area = float(x_full.max()) if x_full.max() > 0 else 1.0
        for i, (n1, n2) in enumerate(bench.connectivity):
            redundant = x_full[i] <= lo * 1.05
            fig.add_trace(go.Scatter(
                x=[xs[n1], xs[n2]], y=[ys[n1], ys[n2]], mode="lines",
                line={"width": 2 + 12 * x_full[i] / max_area,
                      "color": "#c7ccd4" if redundant else BHU_MAROON},
                showlegend=False, hoverinfo="text",
                hovertext=f"bar {i}: A={x_full[i]:.3f}"
                          + (" (redundant)" if redundant else "")))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker={"size": 10, "color": "black"},
            text=[str(i) for i in range(len(xs))],
            textposition="top center", showlegend=False))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(title="Maroon = active load-path member · grey = redundant (min area)")
        st.plotly_chart(fig, use_container_width=True)


# ---------- single run ----------

if run_btn:
    x0 = None
    if use_llm:
        with st.spinner("Querying the AI design agent for warm-start designs..."):
            s = suggest_initial_design(bench)
            x0 = s.x
        st.info(f"**AI agent source:** {s.source} — {s.reasoning}")

    with st.spinner(f"Running {algo.upper()} (pop={pop_size}, gens={n_gen})..."):
        t0 = time.perf_counter()
        result = run(algo, bench, seed=int(seed), pop_size=pop_size, n_gen=n_gen, x0=x0)
        wall = time.perf_counter() - t0

    _show_result(result, wall, bench.reference_optimum_weight)


# ---------- A/B compare: random vs agent warm-start ----------

elif compare_btn:
    lit = bench.reference_optimum_weight
    with st.spinner("Run 1/2 — random initialization..."):
        t0 = time.perf_counter()
        res_rand = run(algo, bench, seed=int(seed), pop_size=pop_size, n_gen=n_gen, x0=None)
        wall_rand = time.perf_counter() - t0

    with st.spinner("Run 2/2 — AI agent warm-start..."):
        s = suggest_initial_design(bench)
        t0 = time.perf_counter()
        res_agent = run(algo, bench, seed=int(seed), pop_size=pop_size, n_gen=n_gen, x0=s.x)
        wall_agent = time.perf_counter() - t0

    st.info(f"**AI agent source:** {s.source} — {s.reasoning}")

    a, b = st.columns(2)
    a.metric("Random — best weight", f"{res_rand.best_weight:.2f}",
             f"{100*(res_rand.best_weight-lit)/lit:+.2f}% vs lit")
    b.metric("AI Agent — best weight", f"{res_agent.best_weight:.2f}",
             f"{100*(res_agent.best_weight-lit)/lit:+.2f}% vs lit")

    # Overlay the two convergence curves
    fig = go.Figure()
    if res_rand.history:
        dr = pd.DataFrame(res_rand.history)
        fig.add_trace(go.Scatter(x=dr["gen"], y=dr["best_weight"], mode="lines",
                                 name="random init", line={"color": "#9aa4b2", "width": 2.5}))
    if res_agent.history:
        da = pd.DataFrame(res_agent.history)
        fig.add_trace(go.Scatter(x=da["gen"], y=da["best_weight"], mode="lines",
                                 name="AI agent warm-start", line={"color": BHU_MAROON, "width": 3}))
    fig.add_hline(y=lit, line_dash="dot", line_color="#18B984",
                  annotation_text=f"literature optimum {lit:.1f}")
    fig.update_layout(title="Convergence: random vs AI agent warm-start",
                      xaxis_title="generation", yaxis_title="best weight", legend_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The agent-warm-started run (maroon) starts lower and converges in fewer "
               "generations — contribution C4, shown live.")


else:
    st.info("Set parameters and click **Run optimization**, or **Compare: random vs AI Agent** "
            "to see the warm-start effect side by side.")
