from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AgentState:
    current_world_state: str
    prediction: str | None = None
    prediction_confidence: float = 0.0
    last_reward: float = 0.0
    last_error: float = 0.0


class StateManager:

    def __init__(self) -> None:
        self._state = AgentState(
            current_world_state="START",
        )
        self._state_history: list[AgentState] = []

    def update(
        self,
        world_state: str,
        prediction: str | None = None,
        prediction_confidence: float = 0.0,
        reward: float = 0.0,
        error: float = 0.0,
    ) -> None:
        self._state = AgentState(
            current_world_state=world_state,
            prediction=prediction,
            prediction_confidence=prediction_confidence,
            last_reward=reward,
            last_error=error,
        )
        self._state_history.append(self._state)

    def current(self) -> AgentState:
        return self._state

    def history(self, n: int | None = None) -> list[AgentState]:
        if n is None:
            return self._state_history.copy()
        return self._state_history[-n:]

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_state": self._state.current_world_state,
            "history_length": len(self._state_history),
        }
