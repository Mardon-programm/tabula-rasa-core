"""
BehavioralProgram (TR-Core Spec v0.2).

A changeable layer of rules over a fixed substrate. All changes happen via
the seven development operations: CREATE, MODIFY, GENERALIZE, SPECIALIZE,
COMPOSE, SPLIT, DELETE.

Revision policy (spec v0.2 §4):
  - one failure ≠ wrong rule; evidence is accumulated first;
  - when failures exceed the threshold, context analysis finds a
    discriminating feature (separation score) → SPECIALIZE or SPLIT;
  - otherwise GENERALIZE on a train-validated hypothesis;
  - DELETE only as a last resort.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable

from behavioral.rules import (
    Action,
    Condition,
    Evidence,
    Literal,
    Rule,
    Subprogram,
)

RuleStatus = str  # LATENT | ACTIVE | RETIRED | DELETED
OperationName = str


@dataclass(slots=True)
class Params:
    """All knobs fixed in pre-registration (spec v0.2 §8)."""

    n_min: int = 3
    p_min: float = 0.7
    explore_rate: float = 0.2
    f_min: int = 3
    d_min: float = 0.3
    sep_min: float = 0.2
    validate_accuracy: float = 0.8
    validate_margin: float = 0.05

    # Induction (develop) knobs.
    develop_interval: int = 50
    max_condition_literals: int = 2
    min_coverage: int = 3

    # Complexity weights (spec v0.2 §6.1).
    base: float = 1.0
    w_lit: float = 1.0
    w_var: float = 2.0
    w_prog: float = 3.0
    w_depth: float = 2.0


class BehavioralProgram:
    def __init__(
        self,
        *,
        params: Params | None = None,
        seed: int = 0,
        allowed_operations: Iterable[str] | None = None,
        rng: int = 0,
    ) -> None:
        self.params = params or Params()
        self.allowed_operations = set(
            allowed_operations
            or [
                "CREATE",
                "MODIFY",
                "GENERALIZE",
                "SPECIALIZE",
                "COMPOSE",
                "SPLIT",
                "DELETE",
            ]
        )
        self.rules: list[Rule] = []
        self.subprograms: dict[str, Subprogram] = {}
        self.step = 0

        # Trace / history for process evidence (protocol §6.7).
        self.history: list[dict] = []

        # Every trial: (context, action, success). Feed for induction.
        self.trials: list[tuple[dict, Action, bool]] = []

        self._rng = __import__("random").Random(seed)

    # ------------------------------------------------------------------
    # Property helpers.
    # ------------------------------------------------------------------

    @property
    def active_rules(self) -> list[Rule]:
        return [r for r in self.rules if r.status == "ACTIVE"]

    @property
    def latent_rules(self) -> list[Rule]:
        return [r for r in self.rules if r.status == "LATENT"]

    def _next_id(self) -> str:
        return f"r{len(self.rules) + 1}"

    # ------------------------------------------------------------------
    # Matching / firing (interpreter, spec §2.5).
    # ------------------------------------------------------------------

    def matching_active(self, context: dict) -> list[Rule]:
        return [r for r in self.active_rules if r.condition.matches(context)]

    def choose_action(
        self,
        context: dict,
        allow_explore: bool = True,
    ) -> Action | None:
        """Deterministic specificity-first resolution over ACTIVE rules."""

        candidates = self.matching_active(context)
        if not candidates:
            if allow_explore:
                explores = [
                    r
                    for r in self.latent_rules
                    if r.condition.matches(context)
                    and self._rng.random() < self.params.explore_rate
                ]
                if explores:
                    return explores[0].action
            return None

        def key(r: Rule) -> tuple:
            return (
                -r.condition.specificity(),
                -r.evidence.total,
            )

        best = max(candidates, key=key)
        return best.action

    def complexity(self) -> float:
        """Adder measure C(program), spec v0.2 §6.1."""

        total = 0.0
        for rule in self.rules:
            total += self.params.base
            total += self.params.w_lit * len(rule.condition.literals)
            total += self.params.w_var * sum(
                1 for lit in rule.condition.literals if lit.operator == "∈"
            )
        total += self.params.w_prog * len(self.subprograms)
        return total

    # ------------------------------------------------------------------
    # Evidence update.
    # ------------------------------------------------------------------

    def observe_outcome(
        self,
        context: dict,
        chosen_action: Action,
        success: bool,
        *,
        explored: bool = False,
    ) -> str:
        """Record outcome against the fired rule; returns revision trigger."""

        self.step += 1
        self.trials.append((dict(context), chosen_action, success, explored))

        # If the action succeeded with no covering rule, CREATE one.
        if success and not any(
            r.condition.matches(context) and r.action == chosen_action
            for r in self.rules
        ):
            self._ensure_rule_for(context, chosen_action)

        # Find the rule that determined the chosen action (same action + fired).
        fired = self.matching_active(context)
        target = next(
            (r for r in fired if r.action == chosen_action),
            None,
        )

        if target is None:
            latent = next(
                (r for r in self.latent_rules if r.action == chosen_action),
                None,
            )
            if latent is not None:
                latent.evidence.record(success)
                self._maybe_activate(latent)
            return "no-rule"

        target.evidence.record(success)
        self._maybe_activate(target)

        # Record observation trace for discriminating-feature search.
        self._log_observation(target, context, success)

        if not success:
            if self._should_revise(target):
                return "revised" if self._revise(target, context) else "failure"
            return "failure"
        return "success"

    def _maybe_activate(self, rule: Rule) -> None:
        if rule.status == "LATENT":
            if (
                rule.evidence.n_success >= self.params.n_min
                and rule.evidence.success_rate >= self.params.p_min
            ):
                rule.status = "ACTIVE"
                self._log("ACTIVATE", rule, {})

    def _should_revise(self, rule: Rule) -> bool:
        ev = rule.evidence
        fail_ratio = ev.n_failure / ev.total if ev.total else 0.0
        return ev.n_failure >= self.params.f_min and fail_ratio >= self.params.d_min

    # ------------------------------------------------------------------
    # Operations (only via substrate mechanism).
    # ------------------------------------------------------------------

    def create(
        self,
        condition: Condition,
        action: Action,
        *,
        allow: bool = True,
    ) -> Rule:
        if "CREATE" not in self.allowed_operations and not allow:
            return Rule(condition, action)
        rule = Rule(
            condition=condition,
            action=action,
            status="LATENT",
            id=self._next_id(),
            created_step=self.step,
        )
        rule.add_operation("CREATE")
        self.rules.append(rule)
        self._log("CREATE", rule, {})
        return rule

    def specialize(
        self,
        rule: Rule,
        attribute: str,
        value: object,
        *,
        operator: str = "=",
    ) -> Rule:
        """Split rule along an attribute; keep the parent active (≠ value).

        The existing rule keeps firing minus the culprit region
        (attribute != value); a new LATENT rule covers attribute == value.
        So previously-working knowledge is never retired by specialization.
        """
        if "SPECIALIZE" not in self.allowed_operations:
            return rule

        if any(lit.attribute == attribute for lit in rule.condition.literals):
            return rule  # already differentiated along this attribute

        base = list(rule.condition.literals)
        # Keep parent active, but no longer covering attr == value.
        rule.condition = Condition(base + [Literal(attribute, "≠", value)])
        rule.add_operation("SPECIALIZE")

        new_rule = Rule(
            condition=Condition(base + [Literal(attribute, operator, value)]),
            action=rule.action,
            status="LATENT",
            id=self._next_id(),
        )
        new_rule.add_operation("SPECIALIZE")
        self.rules.append(new_rule)
        self._log("SPECIALIZE", rule, {"attribute": attribute, "value": value})
        return new_rule

    def split(
        self,
        rule: Rule,
        attribute: str,
        groups: list[tuple[str, list[object]]],
    ) -> list[Rule]:
        """Split rule into several rules differentiated along an attribute."""

        if "SPLIT" not in self.allowed_operations:
            return [rule]
        created: list[Rule] = []
        for value, _ in groups:
            new_literals = list(rule.condition.literals)
            new_literals.append(Literal(attribute, "=", value))
            new_rule = Rule(
                condition=Condition(new_literals),
                action=rule.action,
                status="LATENT",
                id=self._next_id(),
                evidence=Evidence(
                    rule.evidence.n_success,
                    rule.evidence.n_failure,
                ),
            )
            new_rule.add_operation("SPLIT")
            self.rules.append(new_rule)
            created.append(new_rule)
        rule.status = "RETIRED"
        self._log("SPLIT", rule, {"attribute": attribute, "groups": len(groups)})
        return created

    def generalize(
        self,
        rule_ids: list[str],
        attribute: str,
        values: list[object],
    ) -> Rule | None:
        """Replace constants on a shared attribute with a variable (∈ set)."""

        if "GENERALIZE" not in self.allowed_operations:
            return None
        source_rules = [r for r in self.rules if r.id in rule_ids]
        if not source_rules:
            return None
        action = source_rules[0].action
        new_literals = [
            lit for lit in source_rules[0].condition.literals
            if lit.attribute != attribute
        ]
        new_literals.append(Literal(attribute, "∈", tuple(values)))
        new_rule = Rule(
            condition=Condition(new_literals),
            action=action,
            status="LATENT",
            id=self._next_id(),
            evidence=Evidence(
                sum(r.evidence.n_success for r in source_rules),
                sum(r.evidence.n_failure for r in source_rules),
            ),
        )
        new_rule.add_operation("GENERALIZE")
        self.rules.append(new_rule)
        for r in source_rules:
            r.status = "RETIRED"
        self._log("GENERALIZE", new_rule, {"attribute": attribute})
        return new_rule

    def compose(
        self,
        program_a_id: str,
        program_b_id: str,
    ) -> str | None:
        if "COMPOSE" not in self.allowed_operations:
            return None
        new_id = f"sub{len(self.subprograms) + 1}"
        self.subprograms[new_id] = Subprogram(program_id=new_id)
        self._log(
            "COMPOSE",
            None,
            {"a": program_a_id, "b": program_b_id, "id": new_id},
        )
        return new_id

    def delete(self, rule: Rule) -> None:
        if "DELETE" not in self.allowed_operations:
            return
        rule.status = "DELETED"
        self._log("DELETE", rule, {})

    def modify(self, rule: Rule, condition: Condition | None = None, action: Action | None = None) -> Rule:
        if "MODIFY" not in self.allowed_operations:
            return rule
        if condition is not None:
            rule.condition = condition
        if action is not None:
            rule.action = action
        rule.add_operation("MODIFY")
        self._log("MODIFY", rule, {})
        return rule

    # ------------------------------------------------------------------
    # Revision machinery (spec §4).
    # ------------------------------------------------------------------

    def _revise(self, rule: Rule, context: dict) -> bool:
        """Attach context analysis: find discriminating feature, then act."""

        feature = self._find_discriminating_feature(rule)
        if feature is None:
            # No discriminating feature — try validate-generalize or delete.
            self._handle_undifferentiated(rule)
            return True

        attr, value = feature
        before = list(self.rules)
        new_rule = self.specialize(rule, attr, value)
        return new_rule is not rule and new_rule is not None

    def _find_discriminating_feature(
        self,
        rule: Rule,
    ) -> tuple[str, object] | None:
        """
        Find attribute whose value best separates successes from failures
        observed on this rule (spec §4.2, separation score).
        """

        successes: list[dict] = []
        failures: list[dict] = []
        for entry in self.history:
            if entry.get("rule_id") != rule.id:
                continue
            if entry.get("success"):
                successes.append(entry.get("context", {}))
            else:
                failures.append(entry.get("context", {}))

        if not successes or not failures:
            return None

        best: tuple[str, object] | None = None
        best_score = self.params.sep_min

        domains = {
            "role": ["adult", "child", "elder"],
            "friendliness": ["warm", "cold"],
            "status": ["insider", "stranger"],
            "mood": ["calm", "upset"],
            "context": ["greeting", "request", "conflict"],
        }
        present = {lit.attribute for lit in rule.condition.literals}

        for attr, values in domains.items():
            if attr in present:
                continue
            for value in values:
                p_share = sum(
                    1 for c in successes if c.get(attr) == value
                ) / max(1, len(successes))
                n_share = sum(
                    1 for c in failures if c.get(attr) == value
                ) / max(1, len(failures))
                sep = abs(p_share - n_share)
                if sep > best_score:
                    best_score = sep
                    best = (attr, value)

        return best

    def _handle_undifferentiated(self, rule: Rule) -> None:
        """No discriminating feature: DELETE as last resort."""

        self.delete(rule)

    def _log(self, op: OperationName, rule: Rule | None, meta: dict) -> None:
        self.history.append(
            {
                "step": self.step,
                "operation": op,
                "rule_id": rule.id if rule else None,
                "meta": meta,
            }
        )

    def _log_observation(
        self,
        rule: Rule,
        context: dict,
        success: bool,
    ) -> None:
        """Log a firing outcome for the discriminating-feature search."""

        self.history.append(
            {
                "step": self.step,
                "operation": "OBSERVE",
                "rule_id": rule.id,
                "success": success,
                "context": dict(context),
            }
        )

    # ------------------------------------------------------------------
    # Learning from a train sample.
    # ------------------------------------------------------------------

    def learn(self, context: dict, action: Action, success: bool) -> None:
        """Feed one episode: match rule, update evidence, possibly revise."""

        if success:
            self._ensure_rule_for(context, action)
        self.observe_outcome(context, action, success)

    def _ensure_rule_for(self, context: dict, action: Action) -> Rule:
        """CREATE a rule when a successful interaction has no covering rule."""

        for rule in self.rules:
            if rule.condition.matches(context) and rule.action == action:
                return rule
        condition = Condition(
            [Literal(k, "=", v) for k, v in context.items()]
        )
        return self.create(condition, action)

    # ------------------------------------------------------------------
    # Induction (develop) — build general rules from own trials.
    # Spec v0.2 §5: generalization is a hypothesis, validated on train.
    # ------------------------------------------------------------------

    def develop(self) -> int:
        """Run an induction pass over the agent's own unbiased trials.

        Only uniform-random (explore) trials are used for validation: they
        are policy-independent, so candidate accuracy is an unbiased estimate
        of P(oracle == A | condition).

        For each action, search single-literal and paired-literal conditions
        validated on those trials, and materialise through the permitted
        operations:

          - single literal  -> GENERALIZE (drops context attributes)
          - paired literals -> COMPOSE   (conjunction of two dimensions)

        Returns the number of rules materialised.
        """

        explore = [t for t in self.trials if t[3]]
        if not explore:
            return 0

        attributes = {"role", "friendliness", "status", "mood", "context"}
        domains = {
            "role": ["adult", "child", "elder"],
            "friendliness": ["warm", "cold"],
            "status": ["insider", "stranger"],
            "mood": ["calm", "upset"],
            "context": ["greeting", "request", "conflict"],
        }

        def condition_matches(
            cond: list[Literal], ctx: dict
        ) -> bool:
            return all(
                ctx.get(lit.attribute) == lit.value for lit in cond
            )

        def candidate_stats(
            cond: list[Literal], action: Action
        ) -> tuple[int, float] | None:
            """Unbiased accuracy/coverage over explore trials.

            We measure outcome-conditional reliability of the candidate
            action: of the uniform-random explore trials matching `cond`, we
            count those where the explore-chosen action equals the candidate
            and the outcome was a success. Because explore actions are drawn
            uniformly, this estimates how often, when the agent acts as the
            candidate, it is correct within that condition — a policy-aligned
            (not policy-biased) estimate over the full context coverage.
            """

            cov = 0
            hits = 0
            for c, a, s, _ in explore:
                if a != action or not condition_matches(cond, c):
                    continue
                cov += 1
                hits += 1 if s else 0
            if cov < self.params.min_coverage:
                return None
            return cov, hits / cov

        materialised = 0

        # Demote earlier generalizations that no longer hold.
        materialised += self._demote_stale_generalizations(explore, domains)

        for action_name in {"approach", "avoid", "ask", "obey"}:
            action = Action("primitive", action_name)

            # Single-literal candidates.
            singles: list[tuple[list[Literal], float, int]] = []
            for attr, values in domains.items():
                for value in values:
                    stats = candidate_stats([Literal(attr, "=", value)], action)
                    if stats is None:
                        continue
                    cov, acc = stats
                    if acc >= self.params.validate_accuracy + self.params.validate_margin:
                        singles.append(([Literal(attr, "=", value)], acc, cov))

            # Paired-literal candidates (conjunction across two attributes).
            pairs: list[tuple[list[Literal], float, int]] = []
            attr_list = list(attributes)
            for ai in range(len(attr_list)):
                for bi in range(ai + 1, len(attr_list)):
                    a1, a2 = attr_list[ai], attr_list[bi]
                    for v1 in domains[a1]:
                        for v2 in domains[a2]:
                            cond = [Literal(a1, "=", v1), Literal(a2, "=", v2)]
                            stats = candidate_stats(cond, action)
                            if stats is None:
                                continue
                            cov, acc = stats
                            if acc >= self.params.validate_accuracy + self.params.validate_margin:
                                pairs.append((cond, acc, cov))

            # Materialise every validated single first (GENERALIZE). Pairs are
            # a SPECIALIZE refinement, only added when no single suffices.
            if singles:
                ordered = sorted(
                    singles, key=lambda x: (-x[2], -x[1])
                )
                for cond, _acc, _cov in ordered:
                    if self._materialise(
                        cond, action, use=("GENERALIZE",)
                    ):
                        materialised += 1
            else:
                for cond, _acc, _cov in sorted(
                    pairs, key=lambda x: (-x[2], -x[1])
                ):
                    if self._materialise(
                        cond, action,
                        use=("GENERALIZE", "SPECIALIZE"),
                    ):
                        materialised += 1

        return materialised

    def _demote_stale_generalizations(
        self,
        explore: list[tuple[dict, Action, bool, bool]],
        domains: dict[str, list[str]],
    ) -> int:
        """RETIRE ACTIVE GENERALIZE/COMPOSE rules that no longer validate.

        Updates are cheap here because every develop() run recomputes
        accuracy on the agent's full unbiased (explore) trial history.
        """
        demoted = 0
        for rule in list(self.rules):
            if rule.status != "ACTIVE":
                continue
            if rule.action.kind != "primitive":
                continue
            if not any(op in ("GENERALIZE", "SPECIALIZE") for op in rule.provenance):
                continue
            cov = hits = 0
            for c, a, s, ex in explore:
                if a != rule.action or not rule.condition.matches(c):
                    continue
                cov += 1
                hits += 1 if s else 0
            if cov < self.params.min_coverage:
                continue
            acc = hits / cov
            if acc < self.params.validate_accuracy - self.params.validate_margin:
                rule.status = "RETIRED"
                demoted += 1
                self._log(
                    "RETIRE",
                    rule,
                    {"reason": "validation_failed", "accuracy": round(acc, 3)},
                )
        return demoted

    def _materialise(
        self,
        literals: list[Literal],
        action: Action,
        *,
        use: tuple[str, ...],
    ) -> bool:
        """CREATE a general rule if not already covered by an active one."""

        condition = Condition(literals)

        # Skip if an ACTIVE/LATENT rule with this action is already at least
        # as general (its literals are a subset of the candidate's).
        for rule in self.rules:
            if rule.action != action:
                continue
            if rule.status not in ("LATENT", "ACTIVE"):
                continue
            if all(lit in literals for lit in rule.condition.literals):
                return False

        requires = {
            "COMPOSE": "COMPOSE",
            "GENERALIZE": "GENERALIZE",
            "SPECIALIZE": "SPECIALIZE",
        }
        for key in use:
            if requires.get(key) not in self.allowed_operations:
                return False

        rule = self.create(condition, action)
        # Tag the provenance so traces show it arose from induction
        # (generalize, or specialize-refinement for paired conditions).
        induced_op = "GENERALIZE" if len(condition.literals) <= 1 else "SPECIALIZE"
        rule.add_operation(induced_op)
        # Hypothesis was already validated on the agent's own train trials
        # (validate_accuracy), so it is immediately usable (spec v0.2 §5).
        rule.status = "ACTIVE"
        self._log("ACTIVATE", rule, {})
        return True

    # ------------------------------------------------------------------
    # Snapshot / traces.
    # ------------------------------------------------------------------

    def trace(self) -> list[dict]:
        return [r.snapshot() for r in self.rules]

    def snapshot(self) -> dict:
        return {
            "step": self.step,
            "rules": [r.snapshot() for r in self.rules],
            "complexity": self.complexity(),
            "history": list(self.history),
        }