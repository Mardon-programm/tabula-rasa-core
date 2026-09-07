"""
Attribute-task environment v3 (Phase 3): rare compositional rules.

Motivation (Phase 1 finding): on v0.1, behavioral-program induction ties/loses
to non-parametric kNN on the asymptotic regime. Phase 3 targets the known
failure mode of non-parametric memory: *sparsity*. We introduce a larger
attribute space and a set of **rare compositional rules** — conjunctions of
3-4 literals that cover only a fraction of contexts. kNN cannot generalise
from such rare conjunctions (its local neighbourhood is empty), whereas
induction can discover them via COMPOSE/SPECIALIZE.

Key design constraint: rare rules must still have *enough* coverage under a
uniform-random explore policy to be validated (min_coverage), but each single
literal of a rare rule must be individually ambiguous (P(success | single
lit) ≈ chance) so that one-literal GENERALIZE cannot capture it — composition
is required.

Vocabulary: 8 attributes, 2-4 values each (>= 36 contexts base, more when
combined). Oracle = priority list of simple + rare compositional rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Callable

# ---------------------------------------------------------------------------
# Vocabulary (Phase 3).
# ---------------------------------------------------------------------------

ATTRIBUTE_DOMAINS_V3: dict[str, list[str]] = {
    "role": ["adult", "child", "elder"],
    "friendliness": ["warm", "cold"],
    "status": ["insider", "stranger"],
    "mood": ["calm", "upset"],
    "context": ["greeting", "request", "conflict"],
    # Rare axes — small values but combinatorially informative.
    "season": ["spring", "summer", "autumn", "winter"],
    "location": ["home", "work", "public", "school"],
    "time": ["dawn", "day", "dusk", "night"],
}

ACTIONS_V3: list[str] = ["approach", "avoid", "ask", "obey", "offer"]

V3_ATTRIBUTES: tuple[str, ...] = tuple(ATTRIBUTE_DOMAINS_V3.keys())

ContextV3 = dict[str, str]


def all_contexts_v3() -> list[ContextV3]:
    contexts: list[ContextV3] = [{}]
    for attribute in V3_ATTRIBUTES:
        next_ctx: list[ContextV3] = []
        for ctx in contexts:
            for value in ATTRIBUTE_DOMAINS_V3[attribute]:
                next_ctx.append({**ctx, attribute: value})
        contexts = next_ctx
    return contexts


def domain_size() -> int:
    total = 1
    for values in ATTRIBUTE_DOMAINS_V3.values():
        total *= len(values)
    return total


# ---------------------------------------------------------------------------
# Oracle v3 — priority list of simple + rare compositional rules.
# ---------------------------------------------------------------------------
#
# Design targets:
#   * common simple rules give a baseline the omniscient baseline can use;
#   * RARE rules are conjunctions of 3-4 literals that cover a small share of
#     the space. Their single literals are individually ambiguous, so only
#     composition captures them. This is where kNN (sparse neighbourhood)
#     and one-literal GENERALIZE both fail.
#
# Rare rule "signature" applies to a fraction of all contexts:

from itertools import product


def _ctx_has(ctx: ContextV3, rules: list[tuple[str, str]]) -> bool:
    return all(ctx.get(k) == v for k, v in rules)


def oracle_action_v3(context: ContextV3) -> str:
    """Deterministic priority-ordered oracle for v3 environment.

    Rare compositional rules take priority. Each is a conjunction of 3–4
    literals whose single literals are individually ambiguous — so any
    faithful learner must compose them. The simple rules below are imbalanced
    on single literals too (they only become decisive as conjunctions), and
    the fallback covers the remainder.
    """

    # ---- Rare compositional rules (require 3-4 literals) ----
    # Each single literal below is ambiguous on its own; only the conjunction
    # disambiguates. Coverage is a small fraction of the full space.
    if context["season"] == "winter" and context["location"] == "public" and context["time"] == "night":
        return "offer"
    if context["location"] == "school" and context["role"] == "child" and context["mood"] == "upset":
        return "ask"
    if context["friendliness"] == "warm" and context["status"] == "stranger" and context["season"] == "spring" and context["time"] == "dawn":
        return "approach"
    if context["role"] == "adult" and context["context"] == "greeting" and context["season"] == "autumn":
        return "offer"

    # ---- Common simple rules ----
    if context["status"] == "insider" and context["mood"] == "calm":
        return "approach"
    if context["context"] == "conflict":
        return "avoid"
    if context["friendliness"] == "cold" and context["role"] == "elder":
        return "obey"

    # ---- Fallback ----
    return "ask"


def is_success_v3(context: ContextV3, action: str) -> bool:
    return oracle_action_v3(context) == action


def validate_oracle_v3() -> None:
    for ctx in all_contexts_v3():
        action = oracle_action_v3(ctx)
        assert action in ACTIONS_V3, f"invalid action {action} for {ctx}"


# ---------------------------------------------------------------------------
# Samplers (Phase 3).
# ---------------------------------------------------------------------------

def _sample_v3(
    rng: Random,
    predicate: Callable[[ContextV3], bool],
    n: int,
    allow_repeat: bool = True,
) -> list[ContextV3]:
    pool = [ctx for ctx in all_contexts_v3() if predicate(ctx)]
    chosen: list[ContextV3] = []
    for _ in range(n):
        if not pool:
            break
        idx = rng.randrange(len(pool))
        chosen.append(pool[idx])
        if not allow_repeat:
            pool.pop(idx)
    return chosen


@dataclass(slots=True)
class GeneratedSetsV3:
    train: list[ContextV3] = field(default_factory=list)
    known: list[ContextV3] = field(default_factory=list)
    t1: list[ContextV3] = field(default_factory=list)
    t2: list[ContextV3] = field(default_factory=list)
    t3: list[ContextV3] = field(default_factory=list)

    @property
    def test_all(self) -> list[ContextV3]:
        return self.t1 + self.t2 + self.t3


class AttributeTaskGeneratorV3:
    """Plenty-of-attributes generator with rare compositional rules.

    Splits:
      * train/test subsets of the FULL space (compositional transfer is
        achieved because rare-rule contexts appear partly in train and partly
        in test, and they are only a small fraction of the whole — so the
        model must compose, not memorise specific conjunctions).
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        train_size: int = 2000,
        known_size: int = 200,
        t1_size: int = 80,
        t2_size: int = 80,
        t3_size: int = 80,
    ) -> None:
        self.rng = Random(seed)
        self.train_size = train_size
        self.known_size = known_size
        self.t1_size = t1_size
        self.t2_size = t2_size
        self.t3_size = t3_size

    def _split(
        self,
        pool: list[ContextV3],
        sizes: dict[str, int],
    ) -> dict[str, list[ContextV3]]:
        rng = Random(self.rng.randint(0, 10 ** 9))
        for key in ("train", "known", "t1", "t2", "t3"):
            if key in sizes:
                sizes[key] = max(0, min(sizes[key], len(pool)))
        # draw disjoint subsets
        idx_pool = list(range(len(pool)))
        rng.shuffle(idx_pool)
        split: dict[str, list[ContextV3]] = {}
        cursor = 0
        for key, n in sizes.items():
            split[key] = [pool[i] for i in idx_pool[cursor:cursor + n]]
            cursor += n
        return split

    def generate(self) -> GeneratedSetsV3:
        validate_oracle_v3()

        # Keep two disjoint pools for a basic train/test separation over the
        # rare rule space. Use a simple 60/40 split of all contexts.
        pool = all_contexts_v3()
        rng = Random(self.rng.randint(0, 10 ** 9))
        rng.shuffle(pool)
        cut = int(len(pool) * 0.65)
        pool_train = pool[:cut]
        pool_test = pool[cut:]

        # Resample (with replacement, since sizes may exceed pool) into sets.
        def sample_from(p: list[ContextV3], n: int) -> list[ContextV3]:
            return [p[rng.randrange(len(p))] for _ in range(n)] if n else []

        train = sample_from(pool_train, self.train_size)
        known = sample_from(pool_train, self.known_size)
        t1 = sample_from(pool_test, self.t1_size)
        t2 = sample_from(pool_test, self.t2_size)
        t3 = sample_from(pool_test, self.t3_size)

        return GeneratedSetsV3(train=train, known=known, t1=t1, t2=t2, t3=t3)
