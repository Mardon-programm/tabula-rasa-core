from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class ActionCandidate:
    name: str
    expected_reward: float
    uncertainty: float
    cost: float = 0.0


@dataclass(slots=True)
class Plan:
    goal: str
    actions: list[str]
    expected_value: float


class Planner:
    def __init__(
        self,
        *,
        uncertainty_penalty: float = 0.3,
        cost_penalty: float = 0.2,
    ) -> None:
        self.uncertainty_penalty = uncertainty_penalty
        self.cost_penalty = cost_penalty

    def score(
        self,
        action: ActionCandidate,
    ) -> float:
        return (
            action.expected_reward
            - self.uncertainty_penalty
            * action.uncertainty
            - self.cost_penalty
            * action.cost
        )

    def choose(
        self,
        candidates: list[ActionCandidate],
    ) -> ActionCandidate | None:
        if not candidates:
            return None

        return max(
            candidates,
            key=self.score,
        )

    def create_plan(
        self,
        *,
        goal: str,
        candidates: list[ActionCandidate],
    ) -> Plan | None:
        selected = self.choose(candidates)

        if selected is None:
            return None

        return Plan(
            goal=goal,
            actions=[selected.name],
            expected_value=self.score(selected),
        )

    def snapshot(
        self,
        plan: Plan | None,
    ) -> dict[str, Any]:
        if plan is None:
            return {
                "plan": None,
            }

        return {
            "plan": {
                "goal": plan.goal,
                "actions": plan.actions,
                "expected_value": round(
                    plan.expected_value,
                    4,
                ),
            }
        }