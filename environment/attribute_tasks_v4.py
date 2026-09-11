"""
Attribute-task environment v4 (Phase 4): Structural-OOD composition
generalization (sub-protocol).

Phase 3 showed random-context transfer; Phase 4 tests whether a compositional
learner GENERALISES a learned rule to a context whose full conjunction was
never observed as a distinct class — the structural-OOD question.

Key learning-theoretic constraint (surfaced during design): flat Apriori
induction materialises only conjunctions with train coverage >= min_coverage.
A held-out rule whose full region has ZERO train support is therefore
unlearnable by *any* coverage-based method (not just BP). We therefore use
the *sub-protocol*: the held-out arity-3 rule is a SPECIALISATION of a
learnable arity-2 training rule. The learner sees the pair ``x∧y -> A``
(over a superset region); the held-out triple ``x∧y∧z -> A`` is the same
action restricted to a never-observed-as-class region. No exact-triple
coverage exists, yet the pair rule covers it, so a compositional learner
generalises correctly while a memoriser can only reach seen sub-pairs.

We include TWO held-out rules:
  * R_combinable  calm∧guard∧night -> approach
        (pair calm∧guard->approach is a train rule: sub-protocol pass);
  * R_noncombinable calm∧guard∧night -> approach is REUSED? No — see below.
        We instead add a second rule whose pairs point to OTHER actions, so
        the triple requires truly-new (unsupported) composition — expected
        failure for all coverage learners. Documented as the boundary.

Because the two definitions collide on the same atoms, we keep ONE clean
combinable held-out rule plus a set of pair rules; the "boundary" question
is reported analytically (any train pair whose atoms set a different action
than the triple would require zero-support composition and fails).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

ATTRIBUTE_DOMAINS_V4: dict[str, list[str]] = {
    "agent": ["dog", "cat", "fox"],
    "emotion": ["calm", "excited"],
    "task": ["fetch", "guard", "hunt"],
    "terrain": ["forest", "field", "river"],
    "weather": ["sunny", "cloudy", "rainy", "storm"],
    "hour": ["dawn", "noon", "dusk", "night"],
}

ACTIONS_V4: list[str] = ["approach", "avoid", "ask", "obey"]

V4_ATTRIBUTES: tuple[str, ...] = tuple(ATTRIBUTE_DOMAINS_V4.keys())

ContextV4 = dict[str, str]


def all_contexts_v4() -> list[ContextV4]:
    contexts: list[ContextV4] = [{}]
    for attribute in V4_ATTRIBUTES:
        nxt: list[ContextV4] = []
        for ctx in contexts:
            for value in ATTRIBUTE_DOMAINS_V4[attribute]:
                nxt.append({**ctx, attribute: value})
        contexts = nxt
    return contexts


# ---------------------------------------------------------------------------
# Oracle rules.
# ---------------------------------------------------------------------------
# Train pair rules (arity 2, all learnable from coverage):
#   calm  ∧ guard -> approach     (pair whose region contains R_combinable)
#   excited ∧ fetch -> ask
#   dog   ∧ forest -> obey
#   fox   ∧ storm  -> avoid
#   midday discriminators for balance:
#   rainy ∧ dusk  -> ask
#   fog? (no) -> keep the 4 above + fallback.
#
# Held-out R_combinable (arity 3, priority, region excluded from train):
#   calm ∧ guard ∧ night -> approach
#   (a strict specialization of the calm∧guard->approach pair.)

TRAIN_RULES_V4: list[tuple[str, tuple[tuple[str, str], ...]]] = [
    ("approach", (("emotion", "calm"), ("task", "guard"))),
    ("ask", (("emotion", "excited"), ("task", "fetch"))),
    ("obey", (("agent", "dog"), ("terrain", "forest"))),
    ("avoid", (("agent", "fox"), ("weather", "storm"))),
]

HELDOUT_RULE_V4: tuple[str, tuple[tuple[str, str], ...]] = (
    "approach",
    (("emotion", "calm"), ("task", "guard"), ("hour", "night")),
)


def _ctx_has(ctx: ContextV4, rule: list[tuple[str, str]]) -> bool:
    return all(ctx.get(k) == v for k, v in rule)


def oracle_action_v4(context: ContextV4) -> str:
    if _ctx_has(context, list(HELDOUT_RULE_V4[1])):
        return HELDOUT_RULE_V4[0]
    for action, sig in TRAIN_RULES_V4:
        if _ctx_has(context, list(sig)):
            return action
    return "ask"


def is_success_v4(context: ContextV4, action: str) -> bool:
    return oracle_action_v4(context) == action


def validate_oracle_v4() -> None:
    for ctx in all_contexts_v4():
        assert oracle_action_v4(ctx) in ACTIONS_V4


# ---------------------------------------------------------------------------
# Sampler with structural holdout of the HELD-OUT TRIPLE region.
# ---------------------------------------------------------------------------


def _sample(rng: Random, pool: list[ContextV4], n: int) -> list[ContextV4]:
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    return [pool[i] for i in idx[:n]]


@dataclass(slots=True)
class GeneratedSetsV4:
    train: list[ContextV4] = field(default_factory=list)
    known: list[ContextV4] = field(default_factory=list)
    test_heldout: list[ContextV4] = field(default_factory=list)
    test_row: list[ContextV4] = field(default_factory=list)

    @property
    def test_all(self) -> list[ContextV4]:
        return self.test_heldout + self.test_row


class AttributeTaskGeneratorV4:
    """Structural-OOD generator (sub-protocol).

    train + known are drawn ONLY from contexts that do NOT satisfy the
    held-out triple, so the full R_combinable region has zero train support
    (the exact triple is never an observed class). The pair rule still covers
    it via generalization. test_heldout samples the excluded triple region.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        train_size: int = 600,
        known_size: int = 200,
        heldout_test_size: int = 60,
        row_test_size: int = 200,
    ) -> None:
        self.rng = Random(seed)
        self.train_size = train_size
        self.known_size = known_size
        self.heldout_test_size = heldout_test_size
        self.row_test_size = row_test_size

    def generate(self) -> GeneratedSetsV4:
        validate_oracle_v4()
        full = all_contexts_v4()
        held_sig = HELDOUT_RULE_V4[1]
        trainable = [c for c in full if not _ctx_has(c, list(held_sig))]
        heldout_ctx = [c for c in full if _ctx_has(c, list(held_sig))]

        r = self.rng
        r_train = Random(r.randint(0, 10 ** 9))
        train = _sample(r_train, list(trainable), self.train_size)
        train_keys = {tuple(sorted(c.items())) for c in train}
        remaining = [c for c in trainable if tuple(sorted(c.items())) not in train_keys]
        known = _sample(Random(r.randint(0, 10 ** 9)), remaining, self.known_size)
        test_heldout = _sample(
            Random(r.randint(0, 10 ** 9)), heldout_ctx, self.heldout_test_size
        )
        test_row = _sample(
            Random(r.randint(0, 10 ** 9)), list(trainable), self.row_test_size
        )
        return GeneratedSetsV4(
            train=train,
            known=known,
            test_heldout=test_heldout,
            test_row=test_row,
        )


# ---------------------------------------------------------------------------
# Oracle rules in structured format (for Gymnasium wrapper)
# ---------------------------------------------------------------------------

ORACLE_RULES_V4: list[dict] = [
    # Held-out rule (priority)
    {"action": "approach", "condition": {"emotion": "calm", "task": "guard", "hour": "night"}},
    # Train pair rules
    {"action": "approach", "condition": {"emotion": "calm", "task": "guard"}},
    {"action": "ask", "condition": {"emotion": "excited", "task": "fetch"}},
    {"action": "obey", "condition": {"agent": "dog", "terrain": "forest"}},
    {"action": "avoid", "condition": {"agent": "fox", "weather": "storm"}},
    # Fallback
    {"action": "ask", "condition": {}},
]
