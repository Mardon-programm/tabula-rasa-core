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

    @property
    def last_profile(self) -> dict:
        return getattr(self, '_last_profile', {})

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
        import time
        import tracemalloc

        tracemalloc.start()
        t0 = time.perf_counter()

        explore = [t for t in self.trials if t[3]]
        if not explore:
            tracemalloc.stop()
            return 0

        # Demote stale previous generalizations.
        materialised = self._demote_stale_generalizations_v3(explore)

        # Per-action custom candidate search.
        for action_name in self.actions:
            action = Action("primitive", action_name)
            self._induct_action(explore, action)
            materialised += self.last_materialised

        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Store profiling info
        self._last_profile = {
            "latency_ms": round((t1 - t0) * 1000, 2),
            "peak_mem_mb": round(peak / (1024 * 1024), 4),
            "n_candidates": len(explore),
            "n_rules_materialized": materialised,
        }

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
        masks_list = row_masks
        hits_list = row_hits
        base_rate = sum(1 for h in row_hits if h) / len(row_hits) if row_hits else 0.25

        def stats(cond_mask: int) -> tuple[int, int]:
            cov = 0
            n_hits = 0
            for idx, m in enumerate(masks_list):
                if (m & cond_mask) == cond_mask:
                    cov += 1
                    if hits_list[idx]:
                        n_hits += 1
            return cov, n_hits
        def _sort_key(cs: frozenset) -> list:
            return sorted((a, str(v)) for a, v in cs)

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


# ===========================================================================
# Block 1: Heuristic Pruning for COMPOSE (Academic Hardening)
# ===========================================================================

    # -----------------------------------------------------------------------
    # 1a. Dynamic Tree Pruning — early termination
    # -----------------------------------------------------------------------

    def _should_prune_branch(
        self,
        cond_set: frozenset[tuple[str, str]],
        cov: int,
        n_hits: int,
        base_rate: float,
        entropy_before: float | None = None,
    ) -> bool:
        """
        Dynamic tree pruning: terminate a branch if:
          1. Binomial p-value already exceeds alpha (no statistical significance)
          2. Precision is too low to ever reach theta even with full data
          3. Entropy reduction is negligible (uncertainty not reduced)
        """
        if cov == 0:
            return True

        prec = n_hits / cov

        # If precision is already below theta - margin, even perfect future
        # data cannot save this branch (assuming coverage grows)
        if prec < self.params.validate_accuracy - self.params.validate_margin:
            return True

        # Binomial p-value check (early termination)
        pval = self._binom_sf(n_hits - 1, cov, base_rate)
        if pval > self.params.alpha * 10:  # 10x alpha as pruning threshold
            return True

        # Entropy reduction check
        if entropy_before is not None:
            entropy_after = self._compute_entropy(cond_set)
            if entropy_before - entropy_after < 0.01:  # negligible reduction
                return True

        return False

    def _compute_entropy(
        self,
        cond_set: frozenset[tuple[str, str]],
    ) -> float:
        """Compute Shannon entropy of coverage distribution for a candidate."""
        import math
        # Approximate: entropy of success rate
        cov, n_hits = self._quick_stats(cond_set)
        if cov == 0:
            return 1.0
        p = n_hits / cov
        if p <= 0 or p >= 1:
            return 0.0
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    def _quick_stats(
        self,
        cond_set: frozenset[tuple[str, str]],
    ) -> tuple[int, int]:
        """Quick coverage/hits computation for a cond_set."""
        # This is a simplified version; real impl uses bitmask
        cov = 0
        hits = 0
        explore = [t for t in self.trials if t[3]]
        for c, a, s, _ in explore:
            if a != self._current_action:
                continue
            if all(c.get(attr) == val for attr, val in cond_set):
                cov += 1
                if s:
                    hits += 1
        return cov, hits

    # -----------------------------------------------------------------------
    # 1b. Sparse Attention Mask over Predicates — Top-K attribute selection
    # -----------------------------------------------------------------------

    def _compute_predicate_attention(
        self,
        explore: list[tuple[dict, Action, bool, bool]],
        action: Action,
    ) -> dict[str, float]:
        """
        Compute attention weights (correlation) for each attribute-value
        predicate with respect to the target action's success.

        Returns a dict mapping attribute -> weight (higher = more predictive).
        """
        import math
        weights: dict[str, float] = {}
        base_rate = self._base_rate_for_action(action, explore)

        for attr in self.attributes:
            # Mutual information between attribute and success
            mi = 0.0
            for value in self.domains.get(attr, []):
                # P(success | attr=value)
                cov_v = sum(1 for c, a, s, _ in explore if a == action and c.get(attr) == value)
                if cov_v == 0:
                    continue
                hits_v = sum(1 for c, a, s, _ in explore if a == action and c.get(attr) == value and s)
                p_v = hits_v / cov_v
                # Contribution to MI: p(attr=value) * KL(p(success|attr=value) || p(success))
                p_attr = cov_v / max(1, sum(1 for c, a, _, _ in explore if a == action))
                if p_v > 0 and p_v < 1 and base_rate > 0 and base_rate < 1:
                    mi += p_attr * (p_v * math.log2(p_v / base_rate) + (1 - p_v) * math.log2((1 - p_v) / (1 - base_rate)))
            weights[attr] = max(0.0, mi)

        return weights

    def _select_top_k_attributes(
        self,
        explore: list[tuple[dict, Action, bool, bool]],
        action: Action,
        k: int = 5,
    ) -> list[str]:
        """
        Select Top-K most predictive attributes using attention weights.
        Reduces search from O(2^N) to O(K * N log N).
        """
        weights = self._compute_predicate_attention(explore, action)
        sorted_attrs = sorted(weights.items(), key=lambda x: -x[1])
        top_k = [attr for attr, w in sorted_attrs[:k] if w > 0]
        # Always include at least the top 1
        if not top_k and sorted_attrs:
            top_k = [sorted_attrs[0][0]]
        return top_k

    def _base_rate_for_action(
        self,
        action: Action,
        explore: list[tuple[dict, Action, bool, bool]],
    ) -> float:
        """Base success rate for an action."""
        action_trials = [t for t in explore if t[1] == action]
        if not action_trials:
            return 0.25  # default uniform
        hits = sum(1 for t in action_trials if t[2])
        return hits / len(action_trials)

    # -----------------------------------------------------------------------
    # 1c. Incremental Rule Update — patch_rule_graph(new_event)
    # -----------------------------------------------------------------------

    def patch_rule_graph(
        self,
        new_event: tuple[dict, Action, bool],
    ) -> dict[str, Any]:
        """
        Incrementally update the rule graph with a single new event,
        without re-running full induction.

        Returns a dict describing what was changed:
          - "created": list of new rule IDs
          - "updated": list of updated rule IDs
          - "retired": list of retired rule IDs
          - "operations": list of operations performed
        """
        context, action, success = new_event
        changes = {
            "created": [],
            "updated": [],
            "retired": [],
            "operations": [],
        }

        # Step 1: Record the trial
        self.trials.append((dict(context), action, success, False))

        # Step 2: Find matching rules
        matching = [r for r in self.rules if r.condition.matches(context)]

        # Step 3: Update evidence for matching rules
        for rule in matching:
            if rule.action == action:
                rule.evidence.record(success)
                self._maybe_activate(rule)
                changes["updated"].append(rule.id)
                changes["operations"].append(f"UPDATE_EVIDENCE({rule.id})")

        # Step 4: Handle success without covering rule → CREATE
        if success and not any(
            r.condition.matches(context) and r.action == action
            for r in self.rules
        ):
            condition = Condition(
                [Literal(k, "=", v) for k, v in context.items()]
            )
            new_rule = self.create(condition, action)
            new_rule.status = "ACTIVE"  # immediately active for single success
            changes["created"].append(new_rule.id)
            changes["operations"].append(f"CREATE({new_rule.id})")

        # Step 5: Check for revision triggers on failed rules
        if not success:
            for rule in matching:
                if rule.action == action and self._should_revise(rule):
                    feature = self._find_discriminating_feature(rule)
                    if feature is not None:
                        attr, value = feature
                        self.specialize(rule, attr, value)
                        changes["operations"].append(f"SPECIALIZE({rule.id}, {attr}={value})")
                    else:
                        self._handle_undifferentiated(rule)
                        changes["retired"].append(rule.id)
                        changes["operations"].append(f"RETIRE({rule.id})")

        # Step 6: Log the observation
        self._log_observation(
            matching[0] if matching else None,
            context,
            success,
        ) if matching else None

        return changes

    def _log_observation(self, rule, context, success):
        """Log observation for discriminating feature search."""
        if rule is not None:
            self.history.append({
                "step": self.step,
                "operation": "OBSERVE",
                "rule_id": rule.id,
                "success": success,
                "context": dict(context),
            })

    # -----------------------------------------------------------------------
    # Override develop() to use pruning and Top-K
    # -----------------------------------------------------------------------

    def develop_with_pruning(self, top_k_attrs: int = 5) -> int:
        """
        Enhanced induction with dynamic pruning and sparse attention.

        Uses Top-K attribute selection to reduce search space,
        and dynamic tree pruning to terminate unpromising branches early.
        """
        explore = [t for t in self.trials if t[3]]
        if not explore:
            return 0

        materialised = self._demote_stale_generalizations_v3(explore)

        for action_name in self.actions:
            action = Action("primitive", action_name)
            # Select Top-K most predictive attributes
            top_attrs = self._select_top_k_attributes(explore, action, k=top_k_attrs)
            if not top_attrs:
                top_attrs = self.attributes[:top_k_attrs]

            # Restrict domains to Top-K attributes
            restricted_domains = {attr: self.domains.get(attr, []) for attr in top_attrs}
            restricted_attrs = top_attrs

            # Run induction with restricted attribute set
            self._induct_action_pruned(explore, action, restricted_attrs, restricted_domains)
            materialised += self.last_materialised

        return materialised

    def _induct_action_pruned(
        self,
        explore: list[tuple[dict, Action, bool, bool]],
        action: Action,
        attributes: list[str],
        domains: dict[str, list[str]],
        *,
        min_hits: int = 5,
        theta: float = 0.95,
        alpha: float = 1e-3,
        max_rules_per_action: int = 40,
    ) -> None:
        """Pruned Apriori induction for one action."""
        self.last_materialised = 0
        min_cov = self.params.min_coverage

        atoms: list[tuple[str, str]] = []
        seen = set()
        for attr in attributes:
            for value in domains.get(attr, []):
                key = (attr, value)
                if key not in seen:
                    seen.add(key)
                    atoms.append(key)

        atom_ix = {atom: i for i, atom in enumerate(atoms)}

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

        masks_list = row_masks
        hits_list = row_hits
        base_rate = sum(1 for h in row_hits if h) / len(row_hits) if row_hits else 0.25

        def stats(cond_mask: int) -> tuple[int, int]:
            cov = 0
            n_hits = 0
            for i, m in enumerate(masks_list):
                if (m & cond_mask) == cond_mask:
                    cov += 1
                    if hits_list[i]:
                        n_hits += 1
            return cov, n_hits

        def _sort_key(cs: frozenset) -> list:
            return sorted((a, str(v)) for a, v in cs)

        supported_prev: list[frozenset] = []
        entropy_before = 1.0

        for k in range(1, self.max_arity + 1):
            if k == 1:
                candidates = [frozenset({atom}) for atom in atoms]
            else:
                built: set[frozenset] = set()
                for base in supported_prev:
                    for atom in atoms:
                        if atom in base:
                            continue
                        built.add(frozenset(base | {atom}))
                candidates = list(built)
                candidates.sort(key=_sort_key)

            if not candidates:
                break

            supported: list[frozenset] = []
            for cond_set in candidates:
                cond_mask = 0
                for attr, value in cond_set:
                    cond_mask |= 1 << atom_ix[(attr, value)]
                cov, n_hits = stats(cond_mask)

                # Dynamic tree pruning
                if self._should_prune_branch(cond_set, cov, n_hits, base_rate, entropy_before):
                    continue

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
            # Update entropy for next level
            entropy_before = self._compute_entropy(frozenset()) if not supported_prev else entropy_before

    # -----------------------------------------------------------------------
    # Override _induct_action to use Top-K (backward compatible)
    # -----------------------------------------------------------------------

    def _induct_action_with_topk(
        self,
        explore: list[tuple[dict, Action, bool, bool]],
        action: Action,
        *,
        top_k: int = 5,
        min_hits: int = 5,
        theta: float = 0.95,
        alpha: float = 1e-3,
        max_rules_per_action: int = 40,
    ) -> None:
        """Wrapper that selects Top-K attributes before induction."""
        top_attrs = self._select_top_k_attributes(explore, action, k=top_k)
        if not top_attrs:
            top_attrs = self.attributes[:top_k]

        restricted_domains = {attr: self.domains.get(attr, []) for attr in top_attrs}
        self._induct_action_pruned(
            explore, action, top_attrs, restricted_domains,
            min_hits=min_hits, theta=theta, alpha=alpha,
            max_rules_per_action=max_rules_per_action,
        )
