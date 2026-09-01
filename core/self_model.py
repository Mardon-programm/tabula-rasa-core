from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class InternalState:
    energy: float = 1.0
    processing_load: float = 0.0
    uncertainty: float = 1.0

    successful_actions: int = 0
    failed_actions: int = 0


class SelfModel:
    def __init__(self) -> None:
        self.state = InternalState()

        self.capabilities: dict[str, float] = {}

        self._predictions: list[float] = []
        self._observations: list[float] = []


    # Internal state

    def update_resources(
        self,
        *,
        energy_delta: float = 0.0,
        processing_load: float | None = None,
    ) -> None:
        self.state.energy = max(
            0.0,
            min(
                1.0,
                self.state.energy + energy_delta,
            ),
        )

        if processing_load is not None:
            self.state.processing_load = max(
                0.0,
                min(1.0, processing_load),
            )

    def update_uncertainty(
        self,
        uncertainty: float,
    ) -> None:
        self.state.uncertainty = max(
            0.0,
            min(1.0, uncertainty),
        )


    # Capabilities

    def set_capability(
        self,
        name: str,
        confidence: float,
    ) -> None:
        self.capabilities[name] = max(
            0.0,
            min(1.0, confidence),
        )

    def capability(
        self,
        name: str,
    ) -> float:
        return self.capabilities.get(name, 0.0)


    # Self prediction

    def record_prediction(
        self,
        expected_success: float,
    ) -> None:
        self._predictions.append(
            max(0.0, min(1.0, expected_success))
        )

    def record_outcome(
        self,
        actual_success: bool,
    ) -> None:
        value = 1.0 if actual_success else 0.0

        self._observations.append(value)

        if actual_success:
            self.state.successful_actions += 1
        else:
            self.state.failed_actions += 1

    @property
    def self_model_accuracy(self) -> float:
        if not self._predictions:
            return 0.0

        count = min(
            len(self._predictions),
            len(self._observations),
        )

        if count == 0:
            return 0.0

        errors = [
            abs(
                self._predictions[i]
                - self._observations[i]
            )
            for i in range(count)
        ]

        return max(
            0.0,
            1.0 - sum(errors) / count,
        )

    @property
    def success_rate(self) -> float:
        total = (
            self.state.successful_actions
            + self.state.failed_actions
        )

        if total == 0:
            return 0.0

        return (
            self.state.successful_actions / total
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "internal_state": {
                "energy": round(
                    self.state.energy,
                    4,
                ),
                "processing_load": round(
                    self.state.processing_load,
                    4,
                ),
                "uncertainty": round(
                    self.state.uncertainty,
                    4,
                ),
                "successful_actions": (
                    self.state.successful_actions
                ),
                "failed_actions": (
                    self.state.failed_actions
                ),
            },
            "capabilities": {
                name: round(value, 4)
                for name, value
                in self.capabilities.items()
            },
            "self_model_accuracy": round(
                self.self_model_accuracy,
                4,
            ),
            "success_rate": round(
                self.success_rate,
                4,
            ),
        }