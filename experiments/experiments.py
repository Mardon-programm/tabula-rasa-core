from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum


class ExperimentType(Enum):
    LEARNING = "learning"
    PREDICTION = "prediction"
    ADAPTATION = "adaptation"
    MEMORY = "memory"
    PLANNING = "planning"
    DRIFT = "drift"


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    experiment_type: ExperimentType
    max_steps: int = 1000
    seed: int = 42
    description: str = ""
    parameters: dict[str, Any] = None


@dataclass(slots=True)
class ExperimentResult:
    name: str
    experiment_type: ExperimentType
    steps_completed: int
    total_steps: int
    
    # Aggregate metrics
    final_accuracy: float = 0.0
    learning_curve: list[float] = None
    
    # Stage-specific results
    adaptation_time: float = 0.0
    memory_retention: float = 0.0
    recovery_after_drift: float = 0.0
    
    total_reward: float = 0.0
    success: bool = False
    
    def __post_init__(self) -> None:
        if self.learning_curve is None:
            self.learning_curve = []


class StandardExperiments:
    """Standard benchmark experiments."""

    @staticmethod
    def learning_curve_experiment() -> ExperimentConfig:
        """
        Experiment 1: Learning Curve
        
        Agent learns basic state transitions.
        Measures how quickly accuracy improves.
        """
        return ExperimentConfig(
            name="learning_curve",
            experiment_type=ExperimentType.LEARNING,
            max_steps=500,
            description=(
                "Agent learns A → B → C → A cycle. "
                "Measures learning speed and accuracy curve."
            ),
            parameters={
                "environment": "simple_cycle",
                "noise": 0.0,
            },
        )

    @staticmethod
    def prediction_accuracy_experiment() -> ExperimentConfig:
        """
        Experiment 2: Prediction Accuracy
        
        After learning, how accurate are predictions?
        Measures calibration.
        """
        return ExperimentConfig(
            name="prediction_accuracy",
            experiment_type=ExperimentType.PREDICTION,
            max_steps=1000,
            description=(
                "After learning, agent makes predictions. "
                "Measures accuracy and confidence calibration."
            ),
            parameters={
                "learning_phase": 500,
                "test_phase": 500,
            },
        )

    @staticmethod
    def adaptation_experiment() -> ExperimentConfig:
        """
        Experiment 3: Adaptation to Change
        
        Environment rule changes.
        How quickly does agent adapt?
        """
        return ExperimentConfig(
            name="adaptation",
            experiment_type=ExperimentType.ADAPTATION,
            max_steps=1000,
            description=(
                "Environment rule changes at t=500. "
                "Measures adaptation time and error recovery."
            ),
            parameters={
                "learning_phase": 500,
                "drift_at_step": 500,
                "adaptation_phase": 500,
            },
        )

    @staticmethod
    def memory_retention_experiment() -> ExperimentConfig:
        """
        Experiment 4: Memory Retention
        
        Sequential learning: A → B → C → back to A.
        Measures catastrophic forgetting.
        """
        return ExperimentConfig(
            name="memory_retention",
            experiment_type=ExperimentType.MEMORY,
            max_steps=1500,
            description=(
                "Learn A, then B, then C, then test A again. "
                "Measures knowledge retention and catastrophic forgetting."
            ),
            parameters={
                "task_a_steps": 400,
                "task_b_steps": 400,
                "task_c_steps": 400,
                "retest_a_steps": 300,
            },
        )

    @staticmethod
    def planning_experiment() -> ExperimentConfig:
        """
        Experiment 5: Planning Depth
        
        Can agent plan multi-step sequences?
        """
        return ExperimentConfig(
            name="planning",
            experiment_type=ExperimentType.PLANNING,
            max_steps=500,
            description=(
                "Agent must find 3-step sequence to goal. "
                "Measures planning depth and success rate."
            ),
            parameters={
                "planning_depth": 3,
                "goal_state": "TARGET",
            },
        )

    @staticmethod
    def concept_drift_experiment() -> ExperimentConfig:
        """
        Experiment 6: Concept Drift Detection
        
        Can agent detect when environment rules change?
        """
        return ExperimentConfig(
            name="concept_drift",
            experiment_type=ExperimentType.DRIFT,
            max_steps=1500,
            description=(
                "Multiple environment shifts. "
                "Measures drift detection time and recovery quality."
            ),
            parameters={
                "drift_1_at": 300,
                "drift_2_at": 900,
            },
        )

    @staticmethod
    def all_experiments() -> list[ExperimentConfig]:
        """Get all standard experiments."""
        return [
            StandardExperiments.learning_curve_experiment(),
            StandardExperiments.prediction_accuracy_experiment(),
            StandardExperiments.adaptation_experiment(),
            StandardExperiments.memory_retention_experiment(),
            StandardExperiments.planning_experiment(),
            StandardExperiments.concept_drift_experiment(),
        ]

    @staticmethod
    def experiment_by_type(
        exp_type: ExperimentType,
    ) -> ExperimentConfig | None:
        """Get experiment by type."""
        for exp in StandardExperiments.all_experiments():
            if exp.experiment_type == exp_type:
                return exp
        return None


def make_experiment_task(
    config: ExperimentConfig,
) -> Callable[[Any], ExperimentResult]:
    """
    Create a task function for an experiment.
    
    Returns a callable that can be executed with an agent.
    """
    
    async def experiment_task(agent: Any) -> ExperimentResult:
        """Execute experiment with given agent."""
        result = ExperimentResult(
            name=config.name,
            experiment_type=config.experiment_type,
            steps_completed=0,
            total_steps=config.max_steps,
        )
        
        # Placeholder: experiments are run by the runner
        # This is just the config wrapper
        
        return result
    
    return experiment_task
