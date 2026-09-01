from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Axiom:
    name: str
    description: str
    forbidden_actions: frozenset[str] = frozenset()
    required_actions: frozenset[str] = frozenset()


@dataclass(slots=True)
class AxiomDecision:
    allowed: bool
    action: str
    violated_axioms: list[str] = field(default_factory=list)


class AxiomMatrix:
    def __init__(
        self,
        axioms: list[Axiom] | None = None,
    ) -> None:
        self._axioms = tuple(axioms or [])

    @property
    def axioms(self) -> tuple[Axiom, ...]:
        return self._axioms

    def evaluate(self, action: str) -> AxiomDecision:
        violations: list[str] = []

        for axiom in self._axioms:
            if action in axiom.forbidden_actions:
                violations.append(axiom.name)

        return AxiomDecision(
            allowed=not violations,
            action=action,
            violated_axioms=violations,
        )

    def add_axiom(self, axiom: Axiom) -> None:
        self._axioms = (*self._axioms, axiom)

    def snapshot(self) -> dict[str, Any]:
        return {
            "axioms": [
                {
                    "name": axiom.name,
                    "description": axiom.description,
                    "forbidden_actions": sorted(
                        axiom.forbidden_actions
                    ),
                    "required_actions": sorted(
                        axiom.required_actions
                    ),
                }
                for axiom in self._axioms
            ]
        }