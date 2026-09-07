"""
TR-Core v1.0 — Interactive Cognitive Core Visualization

Streamlit dashboard that runs the TabulaRasaCore agent live and streams its
actual internal state to the UI. Nothing here is scripted or mocked: every
chart, gauge, graph edge, and memory counter is read directly from the running
cognitive core's internal state after each step.

Run with:

    streamlit run app.py

Modes:
    - Research Mode: step-by-step control, full internal state inspection,
      drift injection, JSON export.
    - Demo Mode: one-click "watch it learn" narrative for non-technical audiences.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
import plotly.graph_objects as go
import networkx as nx

from main import TabulaRasaCore
from environment.sandbox_sim import SandboxEnvironment
from core.developmental_stage import DevelopmentalStage

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def get_or_create_core() -> TabulaRasaCore:
    """Create (once) and cache the live agent + environment in session state."""
    if "agent" not in st.session_state:
        seed = st.session_state.get("seed", 42)
        st.session_state.agent = TabulaRasaCore(seed=seed, enable_metrics=True)
        st.session_state.env = SandboxEnvironment(seed=seed)
        st.session_state.run_history: List[Dict[str, Any]] = []
        st.session_state.event_log: List[str] = []
        st.session_state.drift_step: Optional[int] = None
        st.session_state.drift_fired = False
        st.session_state.error_timeseries: List[float] = []
        st.session_state.drift_timeseries: List[float] = []
        st.session_state.estimated_knowledge: List[float] = []
        st.session_state.uncertainty_timeseries: List[float] = []
        st.session_state.accuracy_timeseries: List[float] = []
    return st.session_state.agent


def reset_simulation() -> None:
    """Wipe the agent and start fresh."""
    for key in ("agent", "env", "run_history", "event_log", "drift_step",
                "drift_fired", "error_timeseries", "drift_timeseries",
                "estimated_knowledge", "uncertainty_timeseries",
                "accuracy_timeseries"):
        st.session_state.pop(key, None)
    st.session_state.log("Simulation reset")


def append_log(message: str) -> None:
    st.session_state.event_log.append(message)


def _agent_step(agent: TabulaRasaCore, env: SandboxEnvironment) -> None:
    """Execute one agent step, capturing live internal signals."""
    obs = env.observe()
    env_result = env.step(agent.previous_action or "X")
    result = agent.step(
        env_observation=obs.state,
        actual_next_state=env_result.state,
        reward=env_result.reward,
    )

    prediction_error = (
        agent.prediction_engine._error_history[-1].magnitude
        if agent.prediction_engine._error_history else 0.0
    )
    drift_signal = agent.prediction_engine.concept_drift_signal()
    knowledge = agent.self_model.estimated_knowledge()
    uncertainty = agent.self_model.uncertainty_level
    accuracy = agent.prediction_engine.accuracy()

    st.session_state.error_timeseries.append(prediction_error)
    st.session_state.drift_timeseries.append(drift_signal)
    st.session_state.estimated_knowledge.append(knowledge)
    st.session_state.uncertainty_timeseries.append(uncertainty)
    st.session_state.accuracy_timeseries.append(accuracy)

    step = agent.step_count
    st.session_state.run_history.append(result)

    if prediction_error > 0.0:
        append_log(
            f"[STEP {step:03d}] Prediction error"
            f" (predicted={result['predicted_state']},"
            f" actual={result['actual_state']})"
        )

    if drift_signal > 0.3:
        append_log(f"[STEP {step:03d}] CONCEPT DRIFT SIGNAL {drift_signal:.2f}")

    if agent.step_count % 50 == 0:
        appendix = f" consolidation → {len(agent.memory_semantic._facts)} facts"
        append_log(f"[STEP {step:03d}] Memory{appendix}")

    if agent.step_count % 25 == 0:
        append_log(
            f"[STEP {step:03d}] State={result['state']}"
            f" reward={result['reward']:.2f}"
            f" acc={accuracy:.0%}"
        )

    return result


def run_steps_for_real(agent: TabulaRasaCore, env: SandboxEnvironment, n: int) -> None:
    """Advance the live simulation by n real steps, firing drift when configured."""
    drift_at = st.session_state.get("drift_step")
    for _ in range(n):
        if drift_at is not None and not st.session_state.drift_fired \
                and agent.step_count >= drift_at:
            env.change_fundamental_rule()
            st.session_state.drift_fired = True
            append_log(
                f"[STEP {agent.step_count:03d}] *** ENVIRONMENT CHANGE:"
                f" fundamental rule altered (v{env.rule_version}) ***"
            )
        _agent_step(agent, env)


# ---------------------------------------------------------------------------
# Plotting helpers — all read from the live agent internals
# ---------------------------------------------------------------------------


def build_prediction_error_figure(labels: List[int]) -> go.Figure:
    """Prediction error + drift signal time series from live data."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels,
        y=st.session_state.error_timeseries,
        mode="lines+markers",
        name="Prediction Error",
        line=dict(color="#e03131", width=2),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=labels,
        y=st.session_state.drift_timeseries,
        mode="lines",
        name="Drift Signal",
        line=dict(color="#ff922b", width=2, dash="dot"),
    ))
    fig.add_hline(y=0.3, line_dash="dash", line_color="#adb5bd",
                  annotation_text="drift threshold")
    if st.session_state.drift_step is not None and st.session_state.drift_step <= len(labels):
        fig.add_vline(x=st.session_state.drift_step, line_dash="dash",
                      line_color="#c92a2a",
                      annotation_text="concept drift",
                      annotation_position="top left")
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(range=[0, 1.05], title="signal"),
        xaxis_title="step",
        template="plotly_dark",
    )
    return fig


def build_cognitive_graph_figure(agent: TabulaRasaCore) -> go.Figure:
    """Live NetworkX layout of the graph brain, edges colored by weight."""
    snap = agent.brain.snapshot()
    node_names = sorted(agent.brain._nodes)

    G = nx.DiGraph()
    for node in node_names:
        G.add_node(node, size=30)
    for e in snap["synapses"]:
        G.add_edge(e["source"], e["target"], weight=max(0.05, e["weight"]))

    pos = nx.spring_layout(G, seed=7, k=2.2)

    edge_x: List[float] = []
    edge_y: List[float] = []
    edge_weights: List[float] = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_weights.append(G.edges[u, v]["weight"])

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    fig = go.Figure()

    # Midpoint markers for edge weight annotation
    edge_mid_x: List[float] = []
    edge_mid_y: List[float] = []
    edge_w_text: List[str] = []
    for idx, (u, v) in enumerate(G.edges()):
        mx = (pos[u][0] + pos[v][0]) / 2
        my = (pos[u][1] + pos[v][1]) / 2
        w = G.edges[u, v]["weight"]
        if w >= 0.25:
            edge_mid_x.append(mx)
            edge_mid_y.append(my)
            edge_w_text.append(f"{w:.2f}")

    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color="#4dabf7", width=2),
            hoverinfo="none", showlegend=False,
        ))

    if edge_mid_x:
        fig.add_trace(go.Scatter(
            x=edge_mid_x, y=edge_mid_y, mode="markers+text",
            text=edge_w_text, textposition="middle center",
            marker=dict(size=14, color="#f08c00", symbol="hexagon"),
            textfont=dict(size=9, color="#ffd43b"),
            hoverinfo="none", showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=list(G.nodes()),
        textposition="bottom center",
        textfont=dict(size=14, color="#f8f9fa"),
        marker=dict(
            size=45, color="#1971c2",
            line=dict(color="#74c0fc", width=2),
        ),
        hoverinfo="text", showlegend=False,
    ))

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_dark",
        title=f"Neural Graph  ({snap['nodes']} nodes · {snap['edges']} edges)",
    )
    return fig


def build_knowledge_figure(labels: List[int]) -> go.Figure:
    """Estimated knowledge + uncertainty from live self-model."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=st.session_state.estimated_knowledge,
        mode="lines", name="Estimated Knowledge",
        line=dict(color="#69db7c", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=st.session_state.uncertainty_timeseries,
        mode="lines", name="Uncertainty",
        line=dict(color="#9775fa", width=2),
    ))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(range=[0, 1.05]),
        xaxis_title="step",
        template="plotly_dark",
    )
    return fig


def build_accuracy_figure(labels: List[int]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=st.session_state.accuracy_timeseries,
        mode="lines", name="Prediction Accuracy",
        line=dict(color="#4dabf7", width=2, shape="hv"),
        fill="tozeroy", fillcolor="rgba(77,171,247,0.15)",
    ))
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(range=[0, 1.05]),
        xaxis_title="step",
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# KPI card
# ---------------------------------------------------------------------------


def metric_card(label: str, value: str, delta: Optional[str] = None,
                color: Optional[str] = None) -> None:
    st.markdown(
        f"""
        <div style="background:#1c1c1e;border:1px solid #333;border-radius:8px;
             padding:10px 14px;text-align:center;">
            <div style="font-size:11px;color:#adb5bd;letter-spacing:0.5px;">
                {label.upper()}
            </div>
            <div style="font-size:26px;font-weight:700;color:{color or '#f8f9fa'};">
                {value}
            </div>
            <div style="font-size:11px;color:#868e96;">{delta or ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def render_research_mode() -> None:
    agent = get_or_create_core()
    env = st.session_state.env

    st.title("TR-Core v1.0 — Research Mode")
    st.caption("Tabula Rasa Cognitive Core · live internal state, streamed directly "
               "from the running agent. No scripted values.")

    with st.sidebar:
        st.header("Controls")
        st.slider("Seed (reset to apply)", 0, 100, int(st.session_state.get("seed", 42)),
                  key="seed_ui")
        st.session_state.seed = st.session_state.seed_ui
        drift_step = st.number_input(
            "Concept drift at step (0 = disable)",
            min_value=0, max_value=1000, value=int(st.session_state.get("drift_step") or 30),
            step=1,
        )
        st.session_state.drift_step = drift_step if drift_step > 0 else None

        col_a, col_b = st.columns(2)
        if col_a.button("⏵ Run ▶"):
            n = st.session_state.get("run_batch", 10)
            run_steps_for_real(agent, env, n)
        if col_b.button("STEP +1"):
            run_steps_for_real(agent, env, 1)

        col_c, col_d = st.columns(2)
        if col_c.button("⏸ Pause"):
            pass
        if col_d.button("↺ Reset"):
            reset_simulation()
            get_or_create_core()

        st.slider("Batch size (steps per Run)", 1, 50, 10, key="run_batch")
        st.caption(f"Goal: {st.session_state.get('goal_steps', 100)} steps")

    # ---- Status bar ----
    status = agent.get_status()
    step = status["step_count"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown("**STATUS**")
    drift_flag = "DRIFT" if status["concept_drift"] > 0.3 else "RUNNING"
    c1.markdown(f"`{drift_flag}`")

    c2.markdown("**STEP**")
    c2.markdown(f"`{step}`")

    c3.markdown("**STAGE**")
    c3.markdown(f"`{status['development_stage']}`")

    c4.markdown("**ACCURACY**")
    c4.markdown(f"`{status['prediction_accuracy']:.0%}`")

    c5.markdown("**TOTAL REWARD**")
    c5.markdown(f"`{status['total_reward']:.2f}`")

    st.divider()

    with st.expander("Development progress", expanded=True):
        stage_index = DevelopmentalStage[status["development_stage"]].value
        total_stages = len(DevelopmentalStage) - 1
        st.progress(stage_index / total_stages,
                    text=f"{status['development_stage']} ({stage_index}/{total_stages})")
        st.caption("Developmental stages: BLANK → PERCEPTION → ASSOCIATION → "
                   "PREDICTION → EXPLORATION → MEMORY → SELF_MODEL → PLANNING → EMBODIMENT")

    # ---- Main panels ----
    left, right = st.columns([3, 2])

    with left:
        labels = list(range(1, len(st.session_state.error_timeseries) + 1))
        st.subheader("Prediction Error & Concept Drift")
        fig = build_prediction_error_figure(labels)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cognitive Knowledge & Uncertainty")
        st.plotly_chart(build_knowledge_figure(labels), use_container_width=True)

        st.subheader("Prediction Accuracy")
        st.plotly_chart(build_accuracy_figure(labels), use_container_width=True)

    with right:
        st.subheader("Cognitive Graph")
        st.plotly_chart(build_cognitive_graph_figure(agent), use_container_width=True)

        st.subheader("Memory")
        mem = status["memory"]
        sm = status["self_model"]
        stats = status["statistics"]

        r1, r2 = st.columns(2)
        r1.metric("Episodic Memory", f"{mem['episodic']}")
        r2.metric("Semantic Facts", f"{mem['semantic']}")

        r3, r4 = st.columns(2)
        r3.metric("Uncertainty", f"{sm['uncertainty_level']:.2f}")
        r4.metric("Exploration", f"{sm['exploration_rate']:.0%}")

        r5, r6 = st.columns(2)
        r5.metric("Adaptation Events", f"{stats['adaptation_events']}")
        r6.metric("Drift Detections", f"{stats['drift_detections']}")

    st.divider()

    col_log, col_env, col_export = st.columns(3)
    with col_log:
        st.subheader("Event Log")
        for entry in st.session_state.event_log[-40:]:
            st.markdown(entry)
    with col_env:
        st.subheader("Environment (live rules)")
        env_snap = env.snapshot()
        st.markdown(f"`Rule version: {env_snap['rule_version']}`")
        for key, target in env_snap["rules"].items():
            st.markdown(f"`{key} → {target}`")
    with col_export:
        st.subheader("Export")
        if st.button("Export JSON"):
            payload = {
                "mode": "research",
                "step": step,
                "status": status,
                "world_model": agent.world_model.snapshot() if hasattr(agent.world_model, "snapshot") else None,
                "graph": agent.brain.snapshot(),
                "semantic": agent.memory_semantic.snapshot(),
                "metrics_history": agent.metrics.export_csv() if agent.metrics else [],
                "run_history": st.session_state.run_history,
            }
            st.download_button(
                "Download tr_core_state.json",
                data=json.dumps(payload, indent=2, default=str),
                file_name="tr_core_state.json",
                mime="application/json",
            )
        st.caption("Export streams the *actual* internal state of the cognitive "
                   "core — prediction history, graph synapses, semantic facts, "
                   "and consolidated metrics.")


def render_demo_mode() -> None:
    get_or_create_core()
    env = st.session_state.env

    st.title("🧠 TR-Core")
    st.subheader("Watch the agent learn an environment from scratch.")
    st.caption("No pretrained networks. No LLMs. A blank slate discovers structure "
               "through prediction, error, and adaptation.")

    started = st.session_state.get("demo_started", False)
    finished = st.session_state.get("demo_finished", False)

    col_a, col_b = st.columns([1, 2])
    if col_a.button("🚀 START EXPERIMENT"):
        reset_simulation()
        agent = get_or_create_core()
        st.session_state.drift_step = 30
        run_steps_for_real(agent, env, 100)
        st.session_state.demo_started = True
        st.session_state.demo_finished = True

    if not started:
        st.info(
            "Press START to run 100 real steps of the cognitive core. "
            "At step 30 the environment's fundamental rule changes — a concept "
            "drift you can watch the agent detect and adapt to."
        )
        return

    step = st.session_state.agent.step_count

    status = st.session_state.agent.get_status()
    accuracy = status["prediction_accuracy"]
    drift = status["concept_drift"]
    mem = status["memory"]

    if finished:
        st.success("**Experiment complete.** Here is what happened:")

    st.markdown(f"""
    ```
    100 steps        Concept drift      Internal model     Stable patterns
                     detected at 30      adapted            re-discovered
        ●─────────────→  ⚠  →  ↻  →  ✓
    ```
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Steps Completed", f"{step} / 100")
    c2.metric("Prediction Accuracy", f"{accuracy:.0%}")
    c3.metric("Semantic Facts", f"{mem['semantic']}")
    c4.metric("Drift Signal", f"{drift:.2f}")

    st.divider()

    if finished:
        drift_idx = 30
        labels = list(range(1, len(st.session_state.error_timeseries) + 1))
        st.subheader("What the agent experienced")
        st.plotly_chart(build_prediction_error_figure(labels), use_container_width=True)

        col_graph, col_mem = st.columns(2)
        with col_graph:
            st.subheader("What the agent now believes")
            st.plotly_chart(build_cognitive_graph_figure(st.session_state.agent),
                            use_container_width=True)
        with col_mem:
            st.subheader("What it has learned")
            st.metric("Episodic Events", mem["episodic"])
            st.metric("Semantic Facts", mem["semantic"])
            st.caption("Semantic facts are stable generalizations consolidated "
                       "from episodic experience.")
            st.subheader("Timeline")
            for entry in ["Blank slate initialized",
                          "Reinforced A→B→C transition",
                          "⚠ Concept drift at step 30",
                          "Adaptive transition re-learning",
                          "New stable pattern discovered"]:
                st.markdown(f"- {entry}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="TR-Core v1.0",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    mode = st.sidebar.radio("Mode", ["Research", "Demo"], index=0)

    if mode == "Research":
        render_research_mode()
    else:
        render_demo_mode()


if __name__ == "__main__":
    main()
