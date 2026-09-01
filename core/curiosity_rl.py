from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any


@dataclass(slots=True)
class CuriosityEvent:
    state: str
    novelty: float
    uncertainty: float
    prediction_error: float
    intrinsic_reward: float


class CuriosityEngine:
    def __init__(
        self,
        *,
        novelty_weight: float = 0.4,
        uncertainty_weight: float = 0.3,
        prediction_error_weight: float = 0.3,
        novelty_decay: float = 0.97,
    ) -> None:
        self.novelty_weight = novelty_weight
        self.uncertainty_weight = uncertainty_weight
        self.prediction_error_weight = prediction_error_weight
        self.novelty_decay = novelty_decay

        self._visits: dict[str, float] = {}
        self._events: list[CuriosityEvent] = []

    def novelty(self, state: str) -> float:
        visits = self._visits.get(state, 0.0)

        return 1.0 / (1.0 + visits)

    def uncertainty(
        self,
        prediction_confidence: float,
    ) -> float:
        return max(
            0.0,
            min(1.0, 1.0 - prediction_confidence),
        )

    def calculate_reward(
        self,
        *,
        state: str,
        prediction_confidence: float = 0.0,
        prediction_error: float = 0.0,
    ) -> float:
        novelty = self.novelty(state)

        uncertainty = self.uncertainty(
            prediction_confidence
        )

        error = max(
            0.0,
            min(1.0, prediction_error),
        )

        reward = (
            self.novelty_weight * novelty
            + self.uncertainty_weight * uncertainty
            + self.prediction_error_weight * error
        )

        self._visits[state] = (
            self._visits.get(state, 0.0) + 1.0
        )

        event = CuriosityEvent(
            state=state,
            novelty=novelty,
            uncertainty=uncertainty,
            prediction_error=error,
            intrinsic_reward=reward,
        )

        self._events.append(event)

        return reward

    def decay_memory(self) -> None:
        for state in list(self._visits):
            self._visits[state] *= self.novelty_decay

            if self._visits[state] < 0.001:
                del self._visits[state]

    def most_curious_states(
        self,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        scores = []

        for state, visits in self._visits.items():
            score = 1.0 / (1.0 + visits)
            scores.append((state, score))

        scores.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scores[:limit]

    @property
    def average_reward(self) -> float:
        if not self._events:
            return 0.0

        return sum(
            event.intrinsic_reward
            for event in self._events
        ) / len(self._events)

    def snapshot(self) -> dict[str, Any]:
        return {
            "known_states": len(self._visits),
            "events": len(self._events),
            "average_intrinsic_reward": self.average_reward,
            "most_curious": self.most_curious_states(),
        }