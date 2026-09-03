from __future__ import annotations

import sys
import os
from abc import ABC, abstractmethod
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.sandbox_sim import SandboxEnvironment


class BasePredictor(ABC):
    """Shared interface for all baselines.

    Each baseline runs the same closed-loop cycle as TR-Core:
    perceive current state -> predict next -> act -> observe actual -> compute error.
    The environment (including drift) is identical across all models, so results are
    directly comparable.
    """

    name: str = "Base"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.current_state: str = "START"
        self.previous_action: str = "X"
        self.step_count: int = 0

    def reset(self) -> None:
        self.current_state = "START"
        self.previous_action = "X"
        self.step_count = 0

    @abstractmethod
    def predict(self, state: str) -> str:
        """Return the predicted next state for the given observation."""

    @abstractmethod
    def learn(self, state: str, action: str, actual_next: str) -> None:
        """Update internal model from experience (no-op for non-adaptive)."""

    @abstractmethod
    def act(self, state: str) -> str:
        """Select an action."""

    def run_step(
        self,
        env: SandboxEnvironment,
    ) -> dict[str, Any]:
        action = self.act(self.current_state)
        self.previous_action = action

        before = env.current_state
        env_result = env.step(action)
        actual_next = env_result.state

        predicted = self.predict(before)
        self.learn(before, action, actual_next)

        error = 0.0 if predicted == actual_next else 1.0

        self.current_state = actual_next
        self.step_count += 1

        return {
            "step": self.step_count,
            "state": before,
            "actual_state": actual_next,
            "predicted_state": predicted,
            "action": action,
            "error": error,
        }


class RandomPredictor(BasePredictor):
    """No prediction, no learning, random action selection."""

    name = "Random"

    def predict(self, state: str) -> str:
        return state

    def learn(self, state: str, action: str, actual_next: str) -> None:
        pass

    def act(self, state: str) -> str:
        from random import Random

        rng = Random(self.seed * 1000 + self.step_count)
        return rng.choice(["X", "Y", "Z"])


class StaticPredictor(BasePredictor):
    """Learns a fixed transition model ONLY from the pre-drift phase.

    It stops updating after the drift point, so it cannot adapt to the new
    environment dynamics. This isolates the value of post-drift adaptation.
    """

    name = "Static"

    def __init__(self, seed: int = 42, freeze_after: int | None = None) -> None:
        super().__init__(seed)
        self._counts: dict[tuple[str, str], dict[str, int]] = {}
        self._freeze_after = freeze_after

    def predict(self, state: str) -> str:
        trans = self._counts.get((state, self.previous_action), {})
        if not trans:
            return state
        return max(trans.items(), key=lambda kv: kv[1])[0]

    def learn(self, state: str, action: str, actual_next: str) -> None:
        if self._freeze_after is not None and self.step_count > self._freeze_after:
            return
        key = (state, action)
        self._counts.setdefault(key, {})
        self._counts[key][actual_next] = self._counts[key].get(actual_next, 0) + 1

    def act(self, state: str) -> str:
        return "X"


class AdaptivePredictor(BasePredictor):
    """Continuous recency-weighted learner.

    Updates its transition model from every observation, applying a decay so
    that old, superseded transitions lose weight over time. This is a strong,
    honest baseline: it genuinely CAN adapt to drift, but has no memory,
    curiosity, or self-model on top of plain transition learning.
    """

    name = "Adaptive"

    def __init__(self, seed: int = 42, recency: float = 0.9) -> None:
        super().__init__(seed)
        self._counts: dict[tuple[str, str], dict[str, int]] = {}
        self._recency = recency

    def predict(self, state: str) -> str:
        trans = self._counts.get((state, self.previous_action), {})
        if not trans:
            return state
        return max(trans.items(), key=lambda kv: kv[1])[0]

    def learn(self, state: str, action: str, actual_next: str) -> None:
        key = (state, action)
        bucket = self._counts.setdefault(key, {})
        # Decay all existing counts so recent observations dominate.
        for k in bucket:
            bucket[k] = bucket[k] * self._recency
        bucket[actual_next] = bucket.get(actual_next, 0.0) + 1.0

    def act(self, state: str) -> str:
        return "X"


BASELINES: dict[str, type[BasePredictor]] = {
    "Random": RandomPredictor,
    "Static": StaticPredictor,
    "Adaptive": AdaptivePredictor,
}
