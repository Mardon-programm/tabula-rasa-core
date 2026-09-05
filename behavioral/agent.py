"""
BehavioralTaskAgent — thin task-agent over a fixed substrate + behavioral program.

Parallels TR-Core's substrate/evaluator, but specialized for symbolic
attribute tasks. The behavioral program is the only learned, changeable
component; the substrate (perception, evaluation, interpreter) is fixed.
"""

from __future__ import annotations

from typing import Iterable

from behavioral import Action, BehavioralProgram, Params
from behavioral.rules import Action as A
from environment.attribute_tasks import ACTIONS


class BehavioralTaskAgent:
    def __init__(
        self,
        *,
        seed: int = 0,
        params: Params | None = None,
        allowed_operations: Iterable[str] | None = None,
    ) -> None:
        self.program = BehavioralProgram(
            params=params,
            seed=seed,
            allowed_operations=allowed_operations,
        )
        self.step_count = 0
        self.last_action: str | None = None

    def perceive(self, context: dict) -> None:
        """Substrate perception: accepted verbatim; cardinality checked."""

        known = {"role", "friendliness", "status", "mood", "context"}
        assert set(context) == known, f"unexpected context keys: {set(context) ^ known}"

    def select_action(self, context: dict, *, explore: bool = True) -> str:
        """Interpreter: specificity-first rule resolution via the program.

        When explore=True, the substrate sometimes takes a uniform-random
        action (explore_rate). That random probe creates unbiased evidence,
        which the induction pass (develop) uses to validate generalizations.
        """

        self.perceive(context)
        self.last_explored = False
        if explore and self.program._rng.random() < self.program.params.explore_rate:
            self.last_explored = True
            return ACTIONS[self.program._rng.randrange(len(ACTIONS))]
        action = self.program.choose_action(context, allow_explore=False)
        if action is None:
            # No covering rule: neutral behaviour (chance-equivalent).
            return ACTIONS[self.program._rng.randrange(len(ACTIONS))]
        assert action.kind == "primitive"
        return action.name

    def evaluate_and_learn(
        self,
        context: dict,
        chosen_action: str,
        success: bool,
    ) -> str:
        """Evaluation outcome feeds the behavioral program (revision trigger)."""

        self.step_count += 1
        self.last_action = chosen_action
        action = A("primitive", chosen_action)
        if getattr(self, "last_explored", False):
            trigger = self.program.observe_outcome(
                context, action, success, explored=True
            )
        else:
            trigger = self.program.observe_outcome(context, action, success)
        interval = self.program.params.develop_interval
        if interval and self.step_count % interval == 0:
            self.program.develop()
        return trigger

    def train(
        self,
        train_experience: list[dict],
        assess_fn=None,
    ) -> dict:
        """
        Run the agent over a train sequence. assess_fn(context)->(action, success)
        supplies action selection + oracle outcome for each context.
        """

        results = {"revision_triggers": 0, "successes": 0, "steps": 0}
        for ctx in train_experience:
            action, success = assess_fn(ctx)
            self.step_count += 1
            self.last_action = action
            trigger = self.evaluate_and_learn(ctx, action, success)
            results["steps"] += 1
            if success:
                results["successes"] += 1
            if trigger == "revised":
                results["revision_triggers"] += 1
        return results