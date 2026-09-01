from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any


@dataclass(slots=True)
class EnvironmentObservation:
    state: str
    available_actions: list[str]
    reward: float
    changed: bool = False


class SandboxEnvironment:
    def __init__(
        self,
        *,
        seed: int = 42,
    ) -> None:
        self.random = Random(seed)

        self.current_state = "A"

        self.actions = [
            "X",
            "Y",
            "Z",
        ]

        self._rules: dict[
            tuple[str, str],
            tuple[str, float],
        ] = {
            ("A", "X"): ("B", 1.0),
            ("B", "X"): ("C", 1.0),
            ("C", "X"): ("A", 1.0),

            ("A", "Y"): ("C", 0.2),
            ("B", "Y"): ("A", 0.2),
            ("C", "Y"): ("B", 0.2),

            ("A", "Z"): ("A", 0.0),
            ("B", "Z"): ("B", 0.0),
            ("C", "Z"): ("C", 0.0),
        }

        self.step_count = 0
        self.rule_version = 1

    def observe(self) -> EnvironmentObservation:
        return EnvironmentObservation(
            state=self.current_state,
            available_actions=list(self.actions),
            reward=0.0,
        )

    def step(
        self,
        action: str,
    ) -> EnvironmentObservation:
        if action not in self.actions:
            raise ValueError(
                f"Unknown action: {action}"
            )

        key = (
            self.current_state,
            action,
        )

        next_state, reward = self._rules[key]

        self.current_state = next_state
        self.step_count += 1

        return EnvironmentObservation(
            state=self.current_state,
            available_actions=list(self.actions),
            reward=reward,
        )

    def change_fundamental_rule(self) -> None:
        """
        Stress test.

        X no longer follows the old A→B→C→A cycle.

        New rule:

            A + X → C
            C + X → B
            B + X → A
        """

        self._rules[("A", "X")] = ("C", 1.0)
        self._rules[("C", "X")] = ("B", 1.0)
        self._rules[("B", "X")] = ("A", 1.0)

        self.rule_version += 1

    def reset(self) -> None:
        self.current_state = "A"
        self.step_count = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_state": self.current_state,
            "actions": list(self.actions),
            "step_count": self.step_count,
            "rule_version": self.rule_version,
            "rules": {
                f"{source}+{action}": target
                for (
                    source,
                    action,
                ), (target, _) in self._rules.items()
            },
        }