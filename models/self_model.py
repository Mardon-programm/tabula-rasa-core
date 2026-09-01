from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Capability:
    name: str
    confidence: float = 0.5
    recent_successes: int = 0
    recent_failures: int = 0
    last_tested: int = 0


class SelfModel:

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}
        
        # Resource tracking
        self.energy_level = 1.0
        self.memory_load = 0.0
        self.processing_load = 0.0
        
        # Performance tracking
        self.prediction_accuracy = 0.0
        self.resource_level = 1.0
        self.graph_size = 0
        self.uncertainty_level = 0.5
        
        self._recent_errors: list[float] = []
        self.exploration_rate = 0.5
        self.planning_success_rate = 0.0
        
        self._step_count = 0

    def record_capability(
        self,
        name: str,
        success: bool,
    ) -> None:
        if name not in self._capabilities:
            self._capabilities[name] = Capability(
                name=name,
                confidence=0.5,
            )
        
        cap = self._capabilities[name]
        
        if success:
            cap.recent_successes += 1
            cap.confidence = min(1.0, cap.confidence + 0.05)
        else:
            cap.recent_failures += 1
            cap.confidence = max(0.0, cap.confidence - 0.05)
        
        cap.last_tested = self._step_count

    def get_capability_confidence(
        self,
        name: str,
    ) -> float:
        if name not in self._capabilities:
            return 0.5 
        
        return self._capabilities[name].confidence

    def update_metrics(
        self,
        prediction_accuracy: float,
        graph_size: int,
        uncertainty: float,
        exploration_rate: float,
        planning_success: float,
    ) -> None:
        self.prediction_accuracy = prediction_accuracy
        self.graph_size = graph_size
        self.uncertainty_level = uncertainty
        self.exploration_rate = exploration_rate
        self.planning_success_rate = planning_success
        
        self.memory_load = min(1.0, graph_size / 100.0)

    def should_explore(self) -> bool:
        explore_signal = (
            self.uncertainty_level * 0.5 +
            (1.0 - self.prediction_accuracy) * 0.5
        )
        
        return explore_signal > 0.4

    def should_exploit(self) -> bool:
        return not self.should_explore()

    def estimated_knowledge(self) -> float:
        known = 1.0 - self.uncertainty_level
        graph_knowledge = min(1.0, self.graph_size / 50.0)

        return (known * 0.6 + graph_knowledge * 0.4)

    def estimated_unknown(self) -> float:
        return 1.0 - self.estimated_knowledge()

    def learning_rate_recommendation(self) -> float:
        if self.prediction_accuracy > 0.8:
            return 0.05  
        elif self.prediction_accuracy < 0.3:
            return 0.3  
        else:
            return 0.15  

    def step(self) -> None:
        self._step_count += 1

    def statistics(self) -> dict[str, Any]:
        return {
            "prediction_accuracy": self.prediction_accuracy,
            "memory_load": self.memory_load,
            "processing_load": self.processing_load,
            "graph_size": self.graph_size,
            "uncertainty_level": self.uncertainty_level,
            "exploration_rate": self.exploration_rate,
            "planning_success_rate": self.planning_success_rate,
            "estimated_knowledge": self.estimated_knowledge(),
            "estimated_unknown": self.estimated_unknown(),
            "capabilities": len(self._capabilities),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "prediction_accuracy": self.prediction_accuracy,
            "memory_load": self.memory_load,
            "graph_size": self.graph_size,
            "uncertainty_level": self.uncertainty_level,
            "statistics": self.statistics(),
        }
