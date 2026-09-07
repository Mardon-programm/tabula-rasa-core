"""
BehavioralProgramV3 — generalized induction for Phase 3 (rare compositional rules).

Extends the v0.2 BehavioralProgram with an Apriori-style induction over
conjunctions of up to ``max_arity`` literals, over an injected attribute /
domain schema. This is what makes COMPOSE a *necessary* operation: rare
compositional rules are ambiguous at every single literal, so only a
conjunction of several literals reaches high precision.

The base-class rule mechanics (CREATE, SPECIALIZE, matrix, evidence, retrial,
demote) are reused; only the `develop()` induction pass is generalised.

Apriori idea:
  - level k = all candidate literal-sets of size k (each attribute used once).
  - coverage C(cand) = # uniform-explore trials matching cand AND choosing the
    action (outcome has a meaning only for the chosen action).
  - Candidate is "supported" at level k if C(cand) >= min_coverage.
  - Anti-monotonicity (Apriori): if C(cand) < min_coverage for a cand, then
    every superset has coverage >= ... NO — coverage of a more specific
    condition can only be LOWER or equal, so Apriori pruning holds: a size-k
    candidate is only worth extending if all its size-(k-1) subsets were
    supported. We prune on coverage only.
  - Among supported candidates, a rule materialises if precision
    (hits/cov) >= validate_accuracy + validate_margin.

Precision threshold disambiguates rare conjunctions that no single literal
satisfies; coverage floor ensures statistical meaning.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

from behavioral.program import BehavioralProgram, Params
from behavioral.rules import Action, Literal


class BehavioralProgramV3(BehavioralProgram):

    def __init__(
        self,
        *,
        attributes: Iterable[str] | None = None,
        domains: dict[str, list[str]] | None = None,
        actions: list[str] | None = None,
        params: Params | None = None,
        seed: int = 0,
        allowed_operations: Iterable[str] | None = None,
        max_arity: int = 4,
    ) -> None:
        super().__init__(
            params=params,
            seed=seed,
            allowed_operations=allowed_operations,
        )
        self.attributes = list(attributes) if attributes is not None else [
            "role", "friendliness", "status", "mood", "context",
        ]
        self.domains = domains or {
            "role": ["adult", "child", "elder"],
            "friendliness": ["warm", "cold"],
            "status": ["insider", "stranger"],
            "mood": ["calm", "upset"],
            "context": ["greeting", "request", "conflict"],
        }
        self.actions = actions or ["approach", "avoid", "ask", "obey"]
        self.max_arity = max_arity

    # ------------------------------------------------------------------
    # Generalized induction (Phase 3).
    # ------------------------------------------------------------------

    def develop(self) -> int:
        explore = [t for t in self.trials if t[3]]
        if not explore:
            return 0

        # Demote stale previous generalizations.
        materialised = self._demote_stale_generalizations_v3(explore)

        # Per-action custom candidate search.
        for action_name in self.actions:
            action = Action("primitive", action_name)
            self._induct_action(explore, action)
            materialised += self.last_materialised

        return materialised

    def _induct_action(
        self,
        explore: list[tuple[dict, Action, bool, bool]],
        action: Action,
        *,
        min_hits: int = 5,
        theta: float = 0.95,
        alpha: float = 1e-3,
        max_rules_per_action: int = 40,
    ) -> None:
        """Apriori-style conjunction search for one action.

        A candidate becomes a rule only if, over the uniform explore trials
        that CHOSE this action:
          * coverage >= params.min_coverage  (statistical meaning)
          * successes >= min_hits            (not a coin-flip streak)
          * precision >= theta               (pure region)
          * binomial p-value vs the action's base success rate <= alpha
        This screens out small-sample spurious conjunctions that would
        otherwise fire wrongly at high specificity.
        """

        import numpy as np

        self.last_materialised = 0
        min_cov = self.params.min_coverage

        atoms: list[tuple[str, str]] = []
        seen = set()
        for attr in self.attributes:
            for value in self.domains.get(attr, []):
                key = (attr, value)
                if key not in seen:
                    seen.add(key)
                    atoms.append(key)
        atom_ix = {atom: i for i, atom in enumerate(atoms)}

        # Tokenise per-action trials into 24-bit masks; stats become vectorised.
        row_masks: list[int] = []
        row_hits: list[bool] = []
        for c, a, s, _e in explore:
            if a == action:
                mask = 0
                for attr, value in c.items():
                    pos = atom_ix.get((attr, value))
                    if pos is not None:
                        mask |= 1 << pos
                row_masks.append(mask)
                row_hits.append(bool(s))
        if not row_masks:
            return
        masks = np.asarray(row_masks, dtype=np.uint32)
        hits = np.asarray(row_hits, dtype=np.bool_)
        base_rate = int(hits.sum()) / len(hits)

        def stats(cond_mask: int) -> tuple[int, int]:
            sel = (masks & cond_mask) == cond_mask
            cov = int(sel.sum())
            return cov, int(hits[sel].sum()) if cov else 0

        def _sort_key(cs: frozenset[tuple[str, str]]) -> list:
            return sorted((a, str(v)) for a, v in cs)

        supported_prev: list[frozenset[tuple[str, str]]] = []
        for k in range(1, self.max_arity + 1):
            if k == 1:
                candidates = [
                    frozenset({atom}) for atom in atoms
                ]
            else:
                built: set[frozenset[tuple[str, str]]] = set()
                for base in supported_prev:
                    for atom in atoms:
                        if atom in base:
                            continue
                        built.add(frozenset(base | {atom}))
                candidates = list(built)
                # Deterministic order (set iteration depends on the string
                # hash seed, which differs between processes).
                candidates.sort(key=_sort_key)

            if not candidates:
                break

            supported: list[frozenset[tuple[str, str]]] = []
            for cond_set in candidates:
                cond_mask = 0
                for attr, value in cond_set:
                    cond_mask |= 1 << atom_ix[(attr, value)]
                cov, n_hits = stats(cond_mask)
                if cov < min_cov:
                    continue
                supported.append(cond_set)
                if n_hits < min_hits:
                    continue
                if (n_hits / cov) < theta:
                    continue
                if self._binom_sf(n_hits - 1, cov, base_rate) > alpha:
                    continue
                if self.last_materialised >= max_rules_per_action:
                    break
                literals = [
                    Literal(attr, "=", val)
                    for attr, val in sorted(cond_set, key=lambda kv: (kv[0], str(kv[1])))
                ]
                if self._materialise_v3(literals, action):
                    self.last_materialised += 1
            supported_prev = supported
            if not supported:
                break

    @staticmethod
    def _binom_sf(k: int, n: int, p: float) -> float:
        """P(X > k) = P(X >= k+1) for X ~ Binomial(n, p)."""

        import numpy as np

        if k >= n or p <= 0.0:
            return 0.0
        if p >= 1.0:
            return 1.0 if k < n else 0.0
        from math import exp, lgamma

        def pmf(j: int) -> float:
            return exp(
                lgamma(n + 1.0)
                - lgamma(j + 1.0)
                - lgamma(n - j + 1.0)
                + j * np.log(p)
                + (n - j) * np.log1p(-p)
            )

        total = 0.0
        for j in range(k + 1, n + 1):
            total += pmf(j)
        return min(float(total), 1.0)

    @staticmethod
    def _binom_sf(k: int, n: int, p: float) -> float:
        """P(X > k) = P(X >= k+1) for X ~ Binomial(n, p)."""

        from math import exp, lgamma

        if k >= n:
            return 0.0
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return 1.0 if k < n else 0.0

        def pmf(j: int) -> float:
            return exp(
                lgamma(n + 1.0)
                - lgamma(j + 1.0)
                - lgamma(n - j + 1.0)
                + j * __import__("math").log(p)
                + (n - j) * __import__("math").log1p(-p)
            )

        total = 0.0
        for j in range(k + 1, n + 1):
            total += pmf(j)
        return min(float(total), 1.0)

    def _demote_stale_generalizations_v3(
        self,
        explore: list[tuple[dict, Action, bool, bool]],
    ) -> int:
        demoted = 0
        for rule in list(self.rules):
            if rule.status != "ACTIVE":
                continue
            if rule.action.kind != "primitive":
                continue
            if not any(op in ("GENERALIZE", "SPECIALIZE", "COMPOSE") for op in rule.provenance):
                continue
            cov = hits = 0
            for c, a, s, _e in explore:
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

    def _materialise_v3(
        self,
        literals: list[Literal],
        action: Action,
    ) -> bool:
        """CREATE a general rule via COMPOSE (arity>=2) or GENERALIZE (arity 1)."""
        from behavioral.rules import Condition

        condition = Condition(literals)

        # skip if an ACTIVE/LATENT rule for this action already subsumes.
        for rule in self.rules:
            if rule.action != action:
                continue
            if rule.status not in ("LATENT", "ACTIVE"):
                continue
            if all(lit in literals for lit in rule.condition.literals):
                return False

        op = "GENERALIZE" if len(literals) == 1 else "COMPOSE"
        if op not in self.allowed_operations:
            return False

        rule = self.create(condition, action)
        rule.add_operation(op)
        rule.status = "ACTIVE"
        self._log("ACTIVATE", rule, {})
        return True
