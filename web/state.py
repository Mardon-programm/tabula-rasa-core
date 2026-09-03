from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import TabulaRasaCore
from environment.sandbox_sim import SandboxEnvironment


@dataclass
class StepSnapshot:
    step: int
    state: str
    action: str
    predicted_state: str
    actual_state: str
    prediction_accuracy: float
    drift_signal: float
    reward: float
    total_reward: float

    error_history: list[float] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)

    graph_nodes: list[str] = field(default_factory=list)
    graph_edges: list[dict[str, Any]] = field(default_factory=list)

    episodic_count: int = 0
    semantic_count: int = 0
    episodic_stats: dict[str, Any] = field(default_factory=dict)
    semantic_facts: list[dict[str, Any]] = field(default_factory=list)

    adaptation_events: int = 0
    drift_detections: int = 0
    recent_adaptations: list[dict[str, Any]] = field(default_factory=list)

    uncertainty: dict[str, float] = field(default_factory=dict)
    self_model: dict[str, Any] = field(default_factory=dict)
    developmental_stage: str = "BLANK"

    env_rules: dict[str, str] = field(default_factory=dict)
    env_changed: bool = False
    rule_version: int = 1

    curiosity_novelty: float = 0.0
    curiosity_reward: float = 0.0

    plasticity: float = 0.5
    stability: float = 0.5


class CognitiveStateTracker:
    def __init__(self, seed: int = 42) -> None:
        self.core = TabulaRasaCore(seed=seed, enable_metrics=True, verbose=False)
        self.env = SandboxEnvironment(seed=seed)
        self.history: list[StepSnapshot] = []
        self._drift_step: Optional[int] = None
        self._last_rule_version = self.env.rule_version

    def reset(self) -> None:
        self.core = TabulaRasaCore(seed=42, enable_metrics=True, verbose=False)
        self.env = SandboxEnvironment(seed=42)
        self.history.clear()
        self._drift_step = None
        self._last_rule_version = self.env.rule_version

    def inject_drift(self) -> None:
        self.env.change_fundamental_rule()
        self._drift_step = self.core.step_count

    def step(self) -> StepSnapshot:
        action = self.core.previous_action or "X"
        env_result = self.env.step(action)

        drift_just_happened = self.env.rule_version != self._last_rule_version
        self._last_rule_version = self.env.rule_version

        result = self.core.step(
            env_observation=self.core.current_state,
            actual_next_state=env_result.state,
            reward=env_result.reward,
        )

        return self._extract_snapshot(result, drift_just_happened)

    def _extract_snapshot(self, result: dict, env_changed: bool) -> StepSnapshot:
        pe = self.core.prediction_engine
        brain = self.core.brain
        wm = self.core.world_model
        ae = self.core.adaptation_engine
        mem_ep = self.core.memory_episodic
        mem_sem = self.core.memory_semantic
        sm = self.core.self_model
        dev = self.core.development
        cur = self.core.curiosity_engine

        error_hist = [e.magnitude for e in pe._error_history]
        accuracy_hist = []
        window = 5
        for i in range(len(error_hist)):
            start = max(0, i - window + 1)
            chunk = error_hist[start : i + 1]
            accuracy_hist.append(1.0 - sum(chunk) / len(chunk))

        graph_edges = []
        for syn in brain.edges():
            graph_edges.append({
                "source": syn.source,
                "target": syn.target,
                "weight": round(syn.weight, 4),
                "activations": syn.activations,
            })

        semantic_facts = []
        for (subj, rel, obj), fact in mem_sem._facts.items():
            semantic_facts.append({
                "subject": subj,
                "relation": rel,
                "object": obj,
                "confidence": round(fact.confidence, 4),
                "observations": fact.observations,
            })

        recent_adapt = []
        for ev in ae._events[-10:]:
            recent_adapt.append({
                "error_magnitude": round(ev.error_magnitude, 4),
                "belief_updated": ev.belief_updated,
                "confidence_delta": round(ev.confidence_delta, 4),
                "drift_detected": ev.concept_drift_detected,
            })

        unc = wm.get_uncertainty(self.core.current_state)
        pv_s = ae.plasticity_vs_stability()

        env_rules = {}
        for (src, act, tgt), tr in wm._transitions.items():
            env_rules[f"{src}+{act}"] = f"{tgt} (p={tr.probability:.2f})"

        snap = StepSnapshot(
            step=result["step"],
            state=result["state"],
            action=result["action"],
            predicted_state=result["predicted_state"],
            actual_state=result["actual_state"],
            prediction_accuracy=round(result["prediction_accuracy"], 4),
            drift_signal=round(result["drift_signal"], 4),
            reward=result["reward"],
            total_reward=round(result["total_reward"], 4),
            error_history=error_hist,
            accuracy_history=accuracy_hist,
            graph_nodes=sorted(brain._nodes),
            graph_edges=graph_edges,
            episodic_count=len(mem_ep.get_all()),
            semantic_count=len(mem_sem._facts),
            episodic_stats=mem_ep.statistics(),
            semantic_facts=semantic_facts,
            adaptation_events=ae._total_updates,
            drift_detections=ae._drift_detections,
            recent_adaptations=recent_adapt,
            uncertainty={
                "epistemic": round(unc.epistemic, 4),
                "aleatoric": round(unc.aleatoric, 4),
                "model": round(unc.model_uncertainty, 4),
                "prediction": round(unc.prediction_uncertainty, 4),
            },
            self_model={
                "exploration_rate": round(sm.exploration_rate, 4),
                "estimated_knowledge": round(sm.estimated_knowledge(), 4),
                "uncertainty_level": round(sm.uncertainty_level, 4),
                "should_explore": sm.should_explore(),
            },
            developmental_stage=dev.current_stage,
            env_rules=env_rules,
            env_changed=env_changed,
            rule_version=self.env.rule_version,
            curiosity_novelty=round(cur.novelty(self.core.current_state), 4),
            curiosity_reward=round(
                cur.calculate_reward(
                    state=self.core.current_state,
                    prediction_confidence=result["prediction_accuracy"],
                    prediction_error=result["drift_signal"],
                ),
                4,
            ),
            plasticity=round(pv_s["plasticity"], 4),
            stability=round(pv_s["stability"], 4),
        )

        self.history.append(snap)
        return snap
