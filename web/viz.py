from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def prediction_error_plot(history: list[Any]) -> go.Figure:
    steps = [s.step for s in history]
    errors = [max(s.drift_signal, 0.0) for s in history]
    raw = [1.0 if s.error_history and s.error_history[-1] > 0.0 else 0.0 for s in history]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=raw,
            mode="lines",
            name="Prediction Error",
            line=dict(color="#ff6b6b", width=1.5),
            opacity=0.6,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=errors,
            mode="lines",
            name="Drift Signal",
            line=dict(color="#4ecdc4", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(78, 205, 196, 0.15)",
        )
    )

    drift_steps = [s.step for s in history if s.env_changed]
    if drift_steps:
        for ds in drift_steps:
            fig.add_vline(
                x=ds,
                line_dash="dash",
                line_color="#feca57",
                annotation_text=f"CONCEPT DRIFT @ {ds}",
                annotation_position="top right",
                annotation_font=dict(color="#feca57"),
            )

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(title="Step", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Error", range=[0, 1.1], gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def accuracy_plot(history: list[Any]) -> go.Figure:
    steps = [s.step for s in history]
    acc = [s.prediction_accuracy for s in history]
    recent = [s.accuracy_history[-1] if s.accuracy_history else 0.0 for s in history]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=acc,
            mode="lines",
            name="Cumulative Accuracy",
            line=dict(color="#48dbfb", width=1.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=recent,
            mode="lines",
            name="Recent Accuracy (win-5)",
            line=dict(color="#feca57", width=2.2),
        )
    )

    drift_steps = [s.step for s in history if s.env_changed]
    for ds in drift_steps:
        fig.add_vline(x=ds, line_dash="dash", line_color="#feca57")

    fig.update_layout(
        height=220,
        margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(title="Step", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Accuracy", range=[0, 1.1], gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def cognitive_graph_plot(snapshot: Any) -> go.Figure:
    nodes = snapshot.graph_nodes
    edges = snapshot.graph_edges

    pos = _layout_nodes(nodes)
    fig = go.Figure()

    for e in edges:
        sx, sy = pos.get(e["source"], (0, 0))
        tx, ty = pos.get(e["target"], (0, 0))
        fig.add_trace(
            go.Scatter(
                x=[sx, tx],
                y=[sy, ty],
                mode="lines",
                line=dict(
                    color="#ff9ff3",
                    width=1 + 8 * e["weight"],
                ),
                opacity=0.4 + 0.6 * e["weight"],
                hoverinfo="text",
                hovertext=f"{e['source']} → {e['target']}<br>weight: {e['weight']:.3f}<br>activations: {e['activations']}",
                showlegend=False,
            )
        )

    for n in nodes:
        nx_p, ny_p = pos.get(n, (0, 0))
        is_current = n == snapshot.state
        size = 30 if is_current else 22
        color = "#feca57" if is_current else "#5f27cd"
        fig.add_trace(
            go.Scatter(
                x=[nx_p],
                y=[ny_p],
                mode="markers+text",
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(color="#ffffff", width=0 if is_current else 1),
                    symbol="circle",
                ),
                text=[n],
                textposition="middle center",
                textfont=dict(color="#ffffff", size=13, family="Arial Black"),
                hoverinfo="text",
                hovertext=f"State: {n}" + ("<br><b>CURRENT</b>" if is_current else ""),
                showlegend=False,
            )
        )

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-1.3, 1.3]),
        yaxis=dict(visible=False, range=[-1.3, 1.3]),
        showlegend=False,
    )
    return fig


def _layout_nodes(nodes: list[str]) -> dict[str, tuple[float, float]]:
    if not nodes:
        return {}

    from math import cos, sin, pi

    positions: dict[str, tuple[float, float]] = {}
    n = len(nodes)
    if n == 1:
        positions[nodes[0]] = (0.0, 0.0)
    else:
        for i, node in enumerate(nodes):
            angle = 2 * pi * i / n - pi / 2
            positions[node] = (cos(angle), sin(angle))
    return positions


def memory_plot(history: list[Any]) -> go.Figure:
    steps = [s.step for s in history]
    episodic = [s.episodic_count for s in history]
    semantic = [s.semantic_count for s in history]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=episodic,
            mode="lines",
            name="Episodic Memory",
            line=dict(color="#48dbfb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=semantic,
            mode="lines",
            name="Semantic Memory (facts)",
            line=dict(color="#1dd1a1", width=2),
        )
    )

    fig.update_layout(
        height=200,
        margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(title="Step", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def uncertainty_plot(history: list[Any]) -> go.Figure:
    steps = [s.step for s in history]
    epi = [s.uncertainty.get("epistemic", 0.0) for s in history]
    model = [s.uncertainty.get("model", 0.0) for s in history]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=epi,
            mode="lines",
            name="Epistemic",
            line=dict(color="#ff9ff3", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=model,
            mode="lines",
            name="Model",
            line=dict(color="#feca57", width=2),
        )
    )

    drift_steps = [s.step for s in history if s.env_changed]
    for ds in drift_steps:
        fig.add_vline(x=ds, line_dash="dash", line_color="#feca57")

    fig.update_layout(
        height=200,
        margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(title="Step", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Uncertainty", range=[0, 1.1], gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig
