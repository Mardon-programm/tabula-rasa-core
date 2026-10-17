from __future__ import annotations

import json
import sys
import os
import random
from typing import Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="TR-Core — Tabula Rasa Cognitive Core",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from web.state import CognitiveStateTracker
from web import viz as V

# Behavioral Program V3 imports
try:
    from behavioral.program_v3 import BehavioralProgramV3
    from behavioral.program import Params
    from behavioral.rules import Action, Literal, Condition
    from environment.attribute_tasks_v4 import (
        AttributeTaskGeneratorV4,
        ATTRIBUTE_DOMAINS_V4,
        V4_ATTRIBUTES,
        ACTIONS_V4,
        oracle_action_v4,
        is_success_v4,
        ContextV4,
    )
    BP_V3_AVAILABLE = True
except ImportError:
    BP_V3_AVAILABLE = False


st.markdown(
    """
    <style>
    .main, .block-container { background-color: #0b0d17; }
    html, body, [class*="css"] { color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #14172b; }
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #48dbfb, #ff9ff3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .app-subtitle { color: #7f8fa6; margin-top: -8px; font-size: 1rem; }
    .metric-box {
        background: #14172b;
        border: 1px solid #2a2e45;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 4px 0;
    }
    .metric-label { color: #7f8fa6; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #ecf0f1; line-height: 1.1; }
    .drift-banner {
        background: #2c1b1b;
        border-left: 6px solid #ff6b6b;
        color: #ffb3b3;
        padding: 10px 16px;
        border-radius: 6px;
        font-weight: 700;
        animation: pulse 1s infinite;
    }
    .log-line { font-family: 'Cascadia Code', monospace; font-size: 0.85rem; }
    .env-rule { font-family: 'Cascadia Code', monospace; font-size: 0.9rem; padding: 3px 0; }
    .env-rule.new { color: #ff6b6b; font-weight: 700; }
    .stage-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 14px;
        font-size: 0.8rem;
        font-weight: 700;
        background: #5f27cd;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="app-title">TR-Core</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Tabula Rasa Cognitive Core — learn an environment from scratch (no pretrained networks, no LLMs)</div>',
    unsafe_allow_html=True,
)
st.divider()


@st.cache_resource
def get_tracker() -> CognitiveStateTracker:
    return CognitiveStateTracker(seed=42)


tracker = get_tracker()

if "mode" not in st.session_state:
    st.session_state.mode = "Research"
if "running" not in st.session_state:
    st.session_state.running = False

with st.sidebar:
    st.subheader("Mode")
    mode_options = ["Research Mode", "Demo Mode"]
    if BP_V3_AVAILABLE:
        mode_options.insert(1, "Phase 4 Demo")
    mode = st.radio(
        "Select interface",
        mode_options,
        index=0,
        key="mode",
    )

    st.divider()
    st.markdown("**Total Steps**")
    total_target = st.slider("Steps", 50, 300, 100, step=10)

    st.markdown("**Drift**")
    drift_at = st.slider("Inject drift at step", 10, 200, 28, step=1)

    if st.button("↻ Reset", width='stretch'):
        tracker.reset()
        st.session_state.running = False
        st.rerun()

    if st.button("⚡ Inject Concept Drift", width='stretch'):
        tracker.inject_drift()
        st.rerun()


done = len(tracker.history)

# ---------- DEMO MODE ----------
if mode == "Demo Mode":
    col_status, col_gauge = st.columns([1, 1])

    with col_status:
        current = tracker.history[-1] if tracker.history else None
        stage = current.developmental_stage if current else "BLANK"
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Status</div>
                <div class="metric-value" style="color:#1dd1a1;">{'RUNNING' if st.session_state.running and done < total_target else 'READY'}</div>
                <div class="metric-label" style="margin-top:8px;">Development Stage</div>
                <span class="stage-badge">#{stage}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_gauge:
        if current:
            pe = current.prediction_accuracy if current.error_history else 0.0
        else:
            pe = 0.0
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Prediction Accuracy</div>
                <div class="metric-value">{pe * 100:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("▶ START EXPERIMENT", width='stretch', type="primary"):
        st.session_state.running = True

    if st.session_state.running and done < total_target:
        # Auto-run in demo mode
        placeholder = st.empty()
        progress = st.progress(0.0)
        for _ in range(total_target - done):
            tracker.step()
            if tracker.core.step_count == drift_at:
                tracker.inject_drift()
            progress.progress(min(1.0, len(tracker.history) / total_target))
        st.session_state.running = False
        st.rerun()

    st.markdown("---")
    if tracker.history:
        current = tracker.history[-1]
        st.subheader(f"Step {current.step} / {len(tracker.history)}")

        if current.env_changed:
            st.markdown(
                '<div class="drift-banner">⚡ CONCEPT DRIFT — the environment rule changed. The agent was not told in advance.</div>',
                unsafe_allow_html=True,
            )

        st.plotly_chart(V.prediction_error_plot(tracker.history), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cognitive Graph**")
            st.plotly_chart(V.cognitive_graph_plot(current), width="stretch")
        with c2:
            st.markdown("**Environment (learned rules)**")
            for k, v in sorted(current.env_rules.items()):
                st.markdown(f'<div class="env-rule">{k} → {v}</div>', unsafe_allow_html=True)

        st.markdown("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Episodic Memory", current.episodic_count)
        m2.metric("Semantic Facts", current.semantic_count)
        m3.metric("Uncertainty", f"{current.self_model['uncertainty_level']:.2f}")
        m4.metric("Knowledge", f"{current.self_model['estimated_knowledge'] * 100:.0f}%")
        m5.metric("Drift Detections", current.drift_detections)
    else:
        st.info("Press START to begin the live experiment.")

    st.stop()

# ---------- PHASE 4 DEMO (BehavioralProgramV3) ----------
if mode == "Phase 4 Demo":
    if not BP_V3_AVAILABLE:
        st.error("BehavioralProgramV3 not available. Check imports.")
        st.stop()

    st.markdown("### Phase 4: Structural-OOD Compositional Generalization")
    st.caption("Agent learns `calm ∧ guard → approach` (pair), generalizes to never-seen `calm ∧ guard ∧ night → approach` (triple)")

    # Session state for BP demo
    if "bp_demo" not in st.session_state:
        st.session_state.bp_demo = {
            "bp": None,
            "gen": None,
            "sets": None,
            "experience": [],
            "step_count": 0,
            "phase": "init",  # init, train, develop, test
            "heldout_results": None,
        }

    bp_demo = st.session_state.bp_demo

    # Sidebar controls for Phase 4
    with st.sidebar:
        st.divider()
        st.markdown("**Phase 4 Controls**")
        
        if bp_demo["phase"] == "init":
            if st.button("🔧 Initialize Environment", width='stretch'):
                bp_demo["gen"] = AttributeTaskGeneratorV4(seed=42)
                bp_demo["sets"] = bp_demo["gen"].generate()
                
                bp_demo["bp"] = BehavioralProgramV3(
                    attributes=list(V4_ATTRIBUTES),
                    domains=ATTRIBUTE_DOMAINS_V4,
                    actions=ACTIONS_V4,
                    params=Params(
                        develop_interval=50,
                        min_coverage=2,
                        validate_accuracy=0.4,
                        validate_margin=0.05,
                        explore_rate=0.5,
                    ),
                    seed=42,
                    max_arity=4,
                )
                bp_demo["phase"] = "train"
                st.rerun()

        elif bp_demo["phase"] == "train":
            st.markdown("**Training Steps**")
            train_steps = st.slider("Experience steps", 10, 500, 100, step=10)
            if st.button("▶ Collect Experience", width='stretch'):
                rng = random.Random(42 + bp_demo["step_count"])
                for _ in range(train_steps):
                    ctx = rng.choice(bp_demo["sets"].train)
                    action_name = rng.choice(ACTIONS_V4)
                    action = Action("primitive", action_name)
                    success = is_success_v4(ctx, action_name)
                    bp_demo["experience"].append((ctx, action, success, True))
                    bp_demo["bp"].learn(ctx, action, success, True)
                    bp_demo["step_count"] += 1
                bp_demo["phase"] = "develop"
                st.rerun()

        elif bp_demo["phase"] == "develop":
            st.markdown("**Induction**")
            if st.button("🧠 Run Apriori Induction (develop)", width='stretch', type="primary"):
                n_new = bp_demo["bp"].develop()
                st.success(f"Materialized {n_new} new rules")
                bp_demo["phase"] = "test"
                st.rerun()

        elif bp_demo["phase"] == "test":
            st.markdown("**Evaluation**")
            if st.button("🧪 Test Held-Out Triple", width='stretch', type="primary"):
                # Test the held-out triple: calm ∧ guard ∧ night → approach
                heldout_ctx: ContextV4 = {
                    "emotion": "calm",
                    "task": "guard",
                    "hour": "night",
                    "agent": "dog",
                    "terrain": "forest",
                    "weather": "sunny",
                }
                pred = bp_demo["bp"].choose_action(heldout_ctx)
                oracle = oracle_action_v4(heldout_ctx)
                success = pred == oracle if pred else False
                
                bp_demo["heldout_results"] = {
                    "context": heldout_ctx,
                    "predicted": pred.name if pred else "None",
                    "oracle": oracle,
                    "success": success,
                }
                st.rerun()

    # Main display
    if bp_demo["phase"] == "init":
        st.info("Click **Initialize Environment** in sidebar to start.")
        
    elif bp_demo["phase"] == "train":
        st.markdown(f"**Experience collected:** {len(bp_demo['experience'])} steps")
        st.markdown(f"**Rules in program:** {len(bp_demo['bp'].rules)}")
        
        # Show experience summary
        if bp_demo["experience"]:
            actions = [str(e[1]) for e in bp_demo["experience"]]
            from collections import Counter
            action_counts = Counter(actions)
            st.bar_chart(action_counts)
            
    elif bp_demo["phase"] == "develop":
        st.markdown(f"**Total rules:** {len(bp_demo['bp'].rules)}")
        
        # Rule visualization
        st.markdown("### Rule Base")
        rule_data = []
        for r in bp_demo["bp"].rules:
            status_color = {
                "LATENT": "#ffa500",
                "ACTIVE": "#00ff00",
                "RETIRED": "#ff0000",
                "DELETED": "#888888",
            }.get(r.status, "#ffffff")
            
            prov = ", ".join(r.provenance) if r.provenance else "—"
            cond_str = " ∧ ".join([f"{l.attribute}={l.value}" for l in r.condition.literals]) if r.condition.literals else "TRUE"
            
            rule_data.append({
                "ID": r.id,
                "Status": r.status,
                "Condition": cond_str,
                "Action": r.action.name,
                "Provenance": prov,
                "Evidence": f"{r.evidence.n_success}/{r.evidence.n_failure}",
            })
        
        if rule_data:
            import pandas as pd
            df = pd.DataFrame(rule_data)
            st.dataframe(df, width='stretch', hide_index=True)
            
    elif bp_demo["phase"] == "test":
        st.markdown(f"**Total rules:** {len(bp_demo['bp'].rules)}")
        
        # Show all rules
        st.markdown("### Final Rule Base")
        rule_data = []
        for r in bp_demo["bp"].rules:
            if r.status in ("ACTIVE", "LATENT"):
                prov = ", ".join(r.provenance) if r.provenance else "—"
                cond_str = " ∧ ".join([f"{l.attribute}={l.value}" for l in r.condition.literals]) if r.condition.literals else "TRUE"
                rule_data.append({
                    "ID": r.id,
                    "Status": r.status,
                    "Condition": cond_str,
                    "Action": r.action.name,
                    "Provenance": prov,
                    "Evidence": f"{r.evidence.n_success}/{r.evidence.n_failure}",
                })
        
        if rule_data:
            import pandas as pd
            df = pd.DataFrame(rule_data)
            st.dataframe(df, width='stretch', hide_index=True)
        
        # Held-out test result
        if bp_demo["heldout_results"]:
            hr = bp_demo["heldout_results"]
            st.markdown("---")
            st.markdown("### 🧪 Held-Out Triple Test")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Context", "calm ∧ guard ∧ night")
            col2.metric("Predicted", hr["predicted"])
            col3.metric("Oracle", hr["oracle"])
            
            if hr["success"]:
                st.success("✅ **SUCCESS**: Agent correctly generalized from pair to triple!")
            else:
                st.error(f"❌ **FAILED**: Predicted {hr['predicted']}, oracle was {hr['oracle']}")
            
            # Test pair rule recovery
            pair_found = False
            for r in bp_demo["bp"].active_rules:
                if r.action.name == "approach":
                    lits = {f"{l.attribute}={l.value}" for l in r.condition.literals}
                    if "emotion=calm" in lits and "task=guard" in lits and len(lits) == 2:
                        pair_found = True
                        break
            
            if pair_found:
                st.success("✅ **Pair Rule Recovered**: `calm ∧ guard → approach` found as ACTIVE rule")
            else:
                st.warning("⚠️ Pair rule not found in ACTIVE rules")

    st.stop()

# ---------- RESEARCH MODE ----------
st.markdown("### Control Console")

controls = {"run": False, "pause": False, "step": False, "reset": False, "drift": False}
c1, c2, c3, c4, c5, c6 = st.columns(6)
if c1.button("▶ RUN", width='stretch'):
    controls["run"] = True
if c2.button("⏸ PAUSE", width='stretch'):
    controls["pause"] = True
if c3.button("⏭ STEP", width='stretch'):
    controls["step"] = True
if c4.button("↻ RESET", width='stretch'):
    controls["reset"] = True
if c5.button("⚡ DRIFT", width='stretch'):
    controls["drift"] = True
if c6.button("⬇ Export JSON", width='stretch'):
    controls["export"] = True

if controls["reset"]:
    tracker.reset()
    st.rerun()

if controls["drift"]:
    tracker.inject_drift()
    st.rerun()

if controls["step"]:
    if tracker.core.step_count + 1 == drift_at:
        tracker.inject_drift()
    tracker.step()

if controls["run"]:
    st.session_state.running = True

if controls["pause"]:
    st.session_state.running = False

if st.session_state.running and done < total_target:
    for _ in range(10):
        if done >= total_target:
            break
        if tracker.core.step_count + 1 == drift_at:
            tracker.inject_drift()
        tracker.step()
    st.rerun()

if controls.get("export"):
    payload = {
        "mode": "research",
        "steps": len(tracker.history),
        "history": [vars(s) for s in tracker.history],
    }
    st.download_button(
        "Download JSON",
        data=json.dumps(payload, indent=2, default=str),
        file_name=f"tr_core_export_{len(tracker.history)}.json",
        mime="application/json",
        type="primary",
    )

current = tracker.history[-1] if tracker.history else None
s1, s2, s3, s4 = st.columns(4)
s1.metric("Status", "RUNNING" if st.session_state.running else "IDLE")
s2.metric("Step", f"{done} / {total_target}")
s3.metric("Stage", current.developmental_stage if current else "BLANK")
s4.metric("Drift Signal", f"{current.drift_signal:.2f}" if current else "—")

st.divider()

if current is None:
    st.info("Press **STEP** or **RUN** to start observing the core live. Every panel streams actual internal state from `core.step()`.")

    st.markdown(
        """
        ### What you'll see
        1. **Prediction Error** — error spikes and drift detection over time
        2. **Cognitive Graph** — real nodes/synapses of the graph brain, weights update live
        3. **Memory** — episodic (raw events) and semantic (stable facts) growth
        4. **Environment** — the transition model the agent actually learned
        5. **Event Log** — adaptation, memory consolidation, drift — as they happen
        """
    )
    st.stop()

left, right = st.columns([1.4, 1])

with left:
    st.markdown("#### Prediction Error & Drift")
    st.plotly_chart(V.prediction_error_plot(tracker.history), width="stretch")

    st.markdown("#### Cognitive Graph — Streamed from Graph Brain")
    st.plotly_chart(V.cognitive_graph_plot(current), width="stretch")

with right:
    st.markdown("#### Accuracy")
    st.plotly_chart(V.accuracy_plot(tracker.history), width="stretch")

    st.markdown("#### Memory Build-up")
    st.plotly_chart(V.memory_plot(tracker.history), width="stretch")

    st.markdown("#### Model Uncertainty")
    st.plotly_chart(V.uncertainty_plot(tracker.history), width="stretch")

st.divider()

col_env, col_mem, col_self = st.columns([1, 1, 1])

with col_env:
    st.markdown("#### Environment (internal transition model)")
    if current.env_changed:
        st.markdown(
            '<div class="drift-banner">⚡ CONCEPT DRIFT DETECTED</div>',
            unsafe_allow_html=True,
        )
    for k, v in sorted(current.env_rules.items()):
        st.markdown(f'<div class="env-rule">{k} → {v}</div>', unsafe_allow_html=True)

with col_mem:
    st.markdown("#### Semantic Memory")
    if current.semantic_facts:
        for f in current.semantic_facts:
            st.markdown(
                f'<div class="log-line">{f["subject"]} --{f["relation"]}-> {f["object"]} '
                f'(conf {f["confidence"]:.2f}, obs {f["observations"]})</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No stable facts yet")

    st.markdown("**Episodic stats**")
    ep = current.episodic_stats
    st.markdown(
        f'<div class="log-line">size: {ep.get("size", 0)} / {ep.get("capacity", 0)}</div>'
        f'<div class="log-line">avg error: {ep.get("avg_error", 0):.3f}</div>'
        f'<div class="log-line">avg surprise: {ep.get("avg_surprise", 0):.3f}</div>',
        unsafe_allow_html=True,
    )

with col_self:
    st.markdown("#### Self-Model")
    sm = current.self_model
    st.markdown(
        f'<div class="log-line">exploration rate: {sm["exploration_rate"]:.3f}</div>'
        f'<div class="log-line">knowledge: {sm["estimated_knowledge"]:.3f}</div>'
        f'<div class="log-line">uncertainty: {sm["uncertainty_level"]:.3f}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Plasticity / Stability**")
    st.markdown(
        f'<div class="log-line">plasticity: {current.plasticity:.3f}</div>'
        f'<div class="log-line">stability: {current.stability:.3f}</div>',
        unsafe_allow_html=True,
    )

st.divider()

st.markdown("#### Live Event Log")
for s in tracker.history[-12:]:
    events = []
    if s.env_changed:
        events.append(f'<span style="color:#ff6b6b">CONCEPT DRIFT</span> @ step {s.step}')
    if s.error_history and s.error_history[-1] > 0.0:
        events.append(f"prediction error spike ({s.error_history[-1]:.2f})")
    if s.adaptation_events > 0 and (len(events) == 0 or True):
        pass
    if s.step % 50 == 0 and s.episodic_count > 0:
        events.append("memory consolidation")
    if s.graph_edges and len(events) == 0:
        events.append("prediction updated")

    label = f'<div class="log-line">[{s.step:04d}]</div>'
    if not events:
        events.append("stable")

    line = f'<div class="log-line">[{s.step:04d}] ' + " · ".join(events) + "</div>"
    st.markdown(line, unsafe_allow_html=True)

st.markdown(
    '<div style="color:#7f8fa6;font-size:0.75rem;margin-top:16px;">All charts stream actual internal state from the running cognitive core — no simulated data.</div>',
    unsafe_allow_html=True,
)
