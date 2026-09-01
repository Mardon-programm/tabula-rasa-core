from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Metrics:
    step: int
    timestamp: float
    
    # Learning
    prediction_accuracy: float = 0.0
    calibration_error: float = 0.0
    
    # Adaptation
    adaptation_rate: float = 0.0
    plasticity: float = 0.0
    stability: float = 0.0
    drift_detection_rate: float = 0.0
    recovery_score: float = 0.0
    
    # Memory
    episodic_size: int = 0
    semantic_size: int = 0
    retention_rate: float = 0.0
    
    # Exploration
    curiosity_score: float = 0.0
    exploration_rate: float = 0.0
    
    # Planning
    planning_success_rate: float = 0.0
    
    # Self-model
    estimated_knowledge: float = 0.0
    uncertainty_level: float = 0.0
    
    # Overall
    total_reward: float = 0.0


class MetricsCollector:
    def __init__(self) -> None:
        self._history: list[Metrics] = []
        self._start_time = 0.0
        self._step_count = 0

    def record(
        self,
        timestamp: float,
        prediction_accuracy: float = 0.0,
        calibration_error: float = 0.0,
        adaptation_rate: float = 0.0,
        plasticity: float = 0.0,
        stability: float = 0.0,
        drift_detection_rate: float = 0.0,
        recovery_score: float = 0.0,
        episodic_size: int = 0,
        semantic_size: int = 0,
        retention_rate: float = 0.0,
        curiosity_score: float = 0.0,
        exploration_rate: float = 0.0,
        planning_success_rate: float = 0.0,
        estimated_knowledge: float = 0.0,
        uncertainty_level: float = 0.0,
        total_reward: float = 0.0,
    ) -> None:

        metrics = Metrics(
            step=self._step_count,
            timestamp=timestamp,
            prediction_accuracy=prediction_accuracy,
            calibration_error=calibration_error,
            adaptation_rate=adaptation_rate,
            plasticity=plasticity,
            stability=stability,
            drift_detection_rate=drift_detection_rate,
            recovery_score=recovery_score,
            episodic_size=episodic_size,
            semantic_size=semantic_size,
            retention_rate=retention_rate,
            curiosity_score=curiosity_score,
            exploration_rate=exploration_rate,
            planning_success_rate=planning_success_rate,
            estimated_knowledge=estimated_knowledge,
            uncertainty_level=uncertainty_level,
            total_reward=total_reward,
        )
        
        self._history.append(metrics)
        self._step_count += 1

    def get_history(self) -> list[Metrics]:
        return self._history.copy()

    def get_recent(self, n: int = 100) -> list[Metrics]:
        return self._history[-n:]

    def average_metric(
        self,
        metric_name: str,
        recent_n: int = 100,
    ) -> float:
        recent = self.get_recent(recent_n)
        
        if not recent:
            return 0.0
        
        values = [
            getattr(m, metric_name, 0.0)
            for m in recent
        ]
        
        return sum(values) / len(values)

    def trend(
        self,
        metric_name: str,
        window: int = 20,
    ) -> str:
        recent = self.get_recent(window * 2)
        
        if len(recent) < window * 2:
            return "→"
        
        older = recent[:window]
        newer = recent[window:]
        
        older_avg = sum(
            getattr(m, metric_name, 0.0) for m in older
        ) / len(older)
        
        newer_avg = sum(
            getattr(m, metric_name, 0.0) for m in newer
        ) / len(newer)
        
        diff = newer_avg - older_avg
        
        if diff > 0.01:
            return "↑"
        elif diff < -0.01:
            return "↓"
        else:
            return "→"

    def summary(self) -> dict[str, Any]:
        if not self._history:
            return {
                "steps": 0,
                "metrics_recorded": 0,
            }
        
        return {
            "steps": self._step_count,
            "metrics_recorded": len(self._history),
            "prediction_accuracy": self.average_metric("prediction_accuracy"),
            "adaptation_rate": self.average_metric("adaptation_rate"),
            "exploration_rate": self.average_metric("exploration_rate"),
            "estimated_knowledge": self.average_metric("estimated_knowledge"),
            "total_reward": self.average_metric("total_reward"),
        }

    def export_csv(self) -> list[dict[str, Any]]:
        rows = []
        for m in self._history:
            rows.append({
                "step": m.step,
                "timestamp": m.timestamp,
                "prediction_accuracy": m.prediction_accuracy,
                "calibration_error": m.calibration_error,
                "adaptation_rate": m.adaptation_rate,
                "plasticity": m.plasticity,
                "stability": m.stability,
                "drift_detection_rate": m.drift_detection_rate,
                "recovery_score": m.recovery_score,
                "episodic_size": m.episodic_size,
                "semantic_size": m.semantic_size,
                "retention_rate": m.retention_rate,
                "curiosity_score": m.curiosity_score,
                "exploration_rate": m.exploration_rate,
                "planning_success_rate": m.planning_success_rate,
                "estimated_knowledge": m.estimated_knowledge,
                "uncertainty_level": m.uncertainty_level,
                "total_reward": m.total_reward,
            })
        return rows

    def clear(self) -> None:
        self._history.clear()
        self._step_count = 0
