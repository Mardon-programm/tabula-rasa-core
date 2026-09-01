from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, Callable

class DevelopmentalStage(Enum):
    BLANK = 0
    PERCEPTION = 1
    ASSOCIATION = 2
    PREDICTION = 3
    EXPLORATION = 4
    MEMORY = 5
    SELF_MODEL = 6
    PLANNING = 7
    EMBODIMENT = 8

@dataclass(slots=True)
class StageMetrics:
    observations: int = 0
    associations: int = 0
    predictions: int = 0

    predictions_accuracy: float = 0.0
    uncertainty_reduction: float = 0.0

    curiosity_score: float = 0.0
    memory_consolidation: float = 0.0

    self_model_accuracy: float = 0.0
    planning_success: float = 0.0

    resource_efficiency: float = 0.0

    custom: dict[str, float] = field(default_factory=dict)

@dataclass(slots=True)
class StageTransition:
    previous_stage: DevelopmentalStage
    next_stage: DevelopmentalStage
    timestamp: float
    metrics: StageMetrics

@dataclass(slots=True)
class StageRequirement:
    name: str
    evaluator: Callable[[StageMetrics], bool]

    def satisfied(self, metrics: StageMetrics) -> bool:
        return self.evaluator(metrics)

class DevelopmentalController:
    def __init__(self, *, initial_stage: DevelopmentalStage = DevelopmentalStage.BLANK) -> None:
        self._stage = initial_stage 
        self._started_at = monotonic()
        
        self._metrics = StageMetrics()
        self._history: list[StageTransition] = []

        self._requirements = self._build_requirements()

    @property
    def stage(self) -> DevelopmentalStage:
        return self._stage

    @property
    def stage_name(self) -> str:
        return self._stage.name

    @property
    def metrics(self) -> StageMetrics:
        return self._metrics

    @property
    def history(self) -> tuple[StageTransition, ...]:
        return tuple(self._history)

    @property
    def uptime(self) -> float:
        return monotonic() - self._started_at

    def update_metrics(self, **kwargs: Any) -> None:
        for name, value in kwargs.items():
            if hasattr(self._metrics, name):
                setattr(self._metrics, name, value)
            else:
                self._metrics.custom[name] = value

    def reset_metrics(self) -> None:
        self._metrics = StageMetrics()

    def can_advance(self) -> bool:
        if self._stage == DevelopmentalStage.EMBODIMENT:
            return False

        next_stage = DevelopmentalStage(self._stage.value + 1)

        requirement = self._requirements.get(next_stage)

        return all(requirement.satisfied(self._metrics) for requirement in requirement)


    def missing_requirements(self) -> list[StageRequirement]:
        if self._stage == DevelopmentalStage.EMBODIMENT:
            return []

        next_stage = DevelopmentalStage(self._stage.value + 1)

        return [requirement.name for requirement in self._requirements.get(next_stage, []) if not requirement.satisfied(self._metrics)]


    def advance(self) -> bool:
        if not self.can_advance():
            return False

        previous = self._stage
        next_stage = DevelopmentalStage(self._stage.value + 1)

        transition = StageTransition(
            previous_stage=previous,
            next_stage=next_stage,
            timestamp=monotonic(),
            metrics=self._metrics,
        )

        self._history.append(transition)
        self._stage = next_stage

        self.reset_metrics()


        return True


    @staticmethod
    def _build_requirements() -> dict[DevelopmentalStage, list[StageRequirement]]:
        return {
            DevelopmentalStage.PERCEPTION: [
                StageRequirement(
                    "minimum observations",
                    lambda m: m.observations >= 100,
                ),
            ],

            DevelopmentalStage.ASSOCIATION: [
                StageRequirement(
                    "minimum observations",
                    lambda m: m.observations >= 1_000,
                ),
                StageRequirement(
                    "formed associations",
                    lambda m: m.associations >= 100,
                ),
            ],

            DevelopmentalStage.PREDICTION: [
                StageRequirement(
                    "formed predictions",
                    lambda m: m.predictions >= 500,
                ),
                StageRequirement(
                    "prediction accuracy >= 60%",
                    lambda m: m.prediction_accuracy >= 0.60,
                ),
            ],

            DevelopmentalStage.EXPLORATION: [
                StageRequirement(
                    "prediction accuracy >= 70%",
                    lambda m: m.prediction_accuracy >= 0.70,
                ),
                StageRequirement(
                    "uncertainty reduction >= 20%",
                    lambda m: m.uncertainty_reduction >= 0.20,
                ),
            ],

            DevelopmentalStage.MEMORY: [
                StageRequirement(
                    "memory consolidation >= 50%",
                    lambda m: m.memory_consolidation >= 0.50,
                ),
            ],

            DevelopmentalStage.SELF_MODEL: [
                StageRequirement(
                    "self-model accuracy >= 60%",
                    lambda m: m.self_model_accuracy >= 0.60,
                ),
            ],

            DevelopmentalStage.PLANNING: [
                StageRequirement(
                    "planning success >= 60%",
                    lambda m: m.planning_success >= 0.60,
                ),
            ],

            DevelopmentalStage.EMBODIMENT: [
                StageRequirement(
                    "planning success >= 80%",
                    lambda m: m.planning_success >= 0.80,
                ),
                StageRequirement(
                    "resource efficiency >= 50%",
                    lambda m: m.resource_efficiency >= 0.50,
                ),
            ],
        }


    def snapshot(self) -> dict[str, Any]:
        return {
            "stage": self._stage.value,
            "stage_name": self._stage.name,
            "uptime": self.uptime,
            "metrics": {
                "observations": self._metrics.observations,
                "associations": self._metrics.associations,
                "predictions": self._metrics.predictions,
                "prediction_accuracy": self._metrics.prediction_accuracy,
                "uncertainty_reduction": self._metrics.uncertainty_reduction,
                "curiosity_score": self._metrics.curiosity_score,
                "memory_consolidation": self._metrics.memory_consolidation,
                "self_model_accuracy": self._metrics.self_model_accuracy,
                "planning_success": self._metrics.planning_success,
                "resource_efficiency": self._metrics.resource_efficiency,
                "custom": dict(self._metrics.custom),
            },
            "can_advance": self.can_advance(),
            "missing_requirements": self.missing_requirements(),
            "transitions": self._history,

        }


    def __repr__(self):
        return (
            f"<DevelopmentalController"
            f"stage={self._stage.name}"
            f"({self._stage.value}/8)>"
            )

    def record_stage_transition(self, current_state: str, knowledge_level: float) -> None:
        """Record current development progress."""
        self.update_metrics(
            observations=self._metrics.observations + 1,
            uncertainty_reduction=max(0.0, knowledge_level - 0.3),
        )

    @property
    def current_stage(self) -> str:
        """Get current stage name."""
        return self._stage.name