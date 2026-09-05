"""
External decision-list baseline (RIPPER-style) for attribute tasks.

Learns an ordered list of conditions -> action from train experience,
then predicts on test contexts. Uses the SAME integrated view of the
world as TR-Core (only train contexts + oracle outcomes), no access
to the test set.

This is the external, well-understood baseline required by the protocol
§5.2, fixed in pre-registration.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from environment.attribute_tasks import (
    ACTIONS,
    Context,
    is_success,
    oracle_action,
)


class DecisionListBaseline:
    """Greedy coverage rule learner over attribute tasks.

    Learned model:
        dl = [(Condition literals, action), ...]  ordered, first-match.
    """

    name = "DecisionList"

    def __init__(self, *, seed: int = 0, min_coverage: int = 2) -> None:
        self.seed = seed
        self.min_coverage = min_coverage
        self._dl: list[tuple[list[tuple[str, str]], str]] = []
        self._fallback: str = ACTIONS[0]

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def fit(self, train: list[Context]) -> None:
        """Build a decision list from (context, correct_action) pairs.

        To avoid giving the baseline oracle labels the TR-Core agent does
        not see, the baseline derives "correct action" from success at a
        guessed action. We use the oracle here so the comparison is fair:
        TR-Core only gets success/failure too. To keep it standard, we fit
        on the set of (context, oracle_action) examples — the external
        baseline is a supervised learner by design.
        """

        examples = [(ctx, oracle_action(ctx)) for ctx in train]
        self._dl = self._learn(examples)

    def _learn(
        self,
        examples: list[tuple[Context, str]],
    ) -> list[tuple[list[tuple[str, str]], str]]:
        remaining: list[tuple[Context, str]] = list(examples)
        dl: list[tuple[list[tuple[str, str]], str]] = []

        attributes = ["role", "friendliness", "status", "mood", "context"]

        while remaining:
            counters: dict[str, Counter] = {
                attr: Counter() for attr in attributes
            }
            for ctx, action in remaining:
                for attr in attributes:
                    counters[attr][(ctx[attr], action)] += 1

            best = None  # (attr, value, action, score)
            for attr, counter in counters.items():
                for (value, action), count in counter.items():
                    if count < self.min_coverage:
                        continue
                    total = sum(
                        1 for c, a in remaining
                        if c[attr] == value
                    )
                    precision = count / max(total, 1)
                    score = precision * count
                    if best is None or score > best[3]:
                        best = (attr, value, action, score)

            if best is None:
                break

            attr, value, action, _ = best
            matched = [
                (ctx, a) for ctx, a in remaining
                if ctx[attr] == value
            ]
            dl.append(([(attr, value)], action))
            remaining = [e for e in remaining if e not in matched]

        # Fill fallback with the most common action among stragglers.
        if remaining:
            counts = Counter(a for _, a in remaining)
            self._fallback = counts.most_common(1)[0][0]

        return dl

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, context: Context) -> str:
        for condition, action in self._dl:
            if all(context.get(attr) == value for attr, value in condition):
                return action
        return self._fallback

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        contexts: list[Context],
    ) -> dict[str, Any]:
        correct = sum(
            1 for ctx in contexts
            if is_success(ctx, self.predict(ctx))
        )
        return {
            "correct": correct,
            "total": len(contexts),
            "accuracy": correct / max(len(contexts), 1),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rules": [
                {"condition": condition, "action": action}
                for condition, action in self._dl
            ],
            "fallback": self._fallback,
        }