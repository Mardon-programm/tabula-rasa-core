"""
Behavioral program data structures (TR-Core Spec v0.2).

Rule = condition (conjunction of literals) → action.
Literal = (attribute, operator, value); operator ∈ {=, ≠, ∈}.
Action  = primitive 'do(...)' or subprogram call 'run(...)'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OPERATORS = ("=", "≠", "∈")


@dataclass(frozen=True, slots=True)
class Literal:
    attribute: str
    operator: str
    value: Any

    def matches(self, context: dict[str, Any]) -> bool:
        observed = context.get(self.attribute)
        if self.operator == "=":
            return observed == self.value
        if self.operator == "≠":
            return observed != self.value
        if self.operator == "∈":
            return observed in self.value
        return False

    def __str__(self) -> str:
        if self.operator == "∈":
            return f"{self.attribute} ∈ {{{self.value}}}"
        return f"{self.attribute} {self.operator} {self.value}"


@dataclass(slots=True)
class Action:
    kind: str  # "primitive" | "subprogram"
    name: str
    program_id: str | None = None

    def __str__(self) -> str:
        if self.kind == "primitive":
            return f"do({self.name})"
        return f"run({self.program_id})"


@dataclass(slots=True)
class Condition:
    literals: list[Literal] = field(default_factory=list)

    def matches(self, context: dict[str, Any]) -> bool:
        return all(lit.matches(context) for lit in self.literals)

    def specificity(self) -> int:
        constants = sum(1 for lit in self.literals if lit.operator != "∈")
        return constants

    def __str__(self) -> str:
        if not self.literals:
            return "always"
        return " AND ".join(str(lit) for lit in self.literals)


@dataclass(slots=True)
class Evidence:
    n_success: int = 0
    n_failure: int = 0

    @property
    def total(self) -> int:
        return self.n_success + self.n_failure

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.n_success / self.total

    def record(self, success: bool) -> None:
        if success:
            self.n_success += 1
        else:
            self.n_failure += 1

    def snapshot(self) -> dict[str, int]:
        return {"success": self.n_success, "failure": self.n_failure}


@dataclass(slots=True)
class Rule:
    condition: Condition
    action: Action
    status: str = "LATENT"  # LATENT | ACTIVE | RETIRED | DELETED
    evidence: Evidence = field(default_factory=Evidence)
    provenance: list[str] = field(default_factory=list)
    created_step: int = 0
    id: str = ""

    def add_operation(self, op: str) -> None:
        self.provenance.append(op)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "condition": str(self.condition),
            "action": str(self.action),
            "status": self.status,
            "evidence": self.evidence.snapshot(),
            "provenance": list(self.provenance),
        }


@dataclass(slots=True)
class Subprogram:
    program_id: str
    rules: list[Rule] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "rules": [r.snapshot() for r in self.rules],
        }