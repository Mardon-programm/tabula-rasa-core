"""
Symbolic attribute-task generator for the TR-Core behavioral-program experiment.

Each task is a full context over a fixed vocabulary of attributes. The
agent selects one action; success is determined by a deterministic oracle
(target program) that is unknown to the agent.

The generator can build train experience, a known-holdout (retention
control), and three genuinely-unseen test structures (T1/T2/T3) with
train/test disjointness guaranteed per the pre-registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

ATTRIBUTE_DOMAINS: dict[str, list[str]] = {
    "role": ["adult", "child", "elder"],
    "friendliness": ["warm", "cold"],
    "status": ["insider", "stranger"],
    "mood": ["calm", "upset"],
    "context": ["greeting", "request", "conflict"],
}

ACTIONS: list[str] = ["approach", "avoid", "ask", "obey"]

ALL_ATTRIBUTES: tuple[str, ...] = tuple(ATTRIBUTE_DOMAINS.keys())

Context = dict[str, str]


def all_contexts() -> list[Context]:
    """Enumerate the full context space (72 contexts)."""

    contexts: list[Context] = [{}]

    for attribute in ALL_ATTRIBUTES:
        contexts = [
            {**ctx, attribute: value}
            for ctx in contexts
            for value in ATTRIBUTE_DOMAINS[attribute]
        ]

    return contexts


# ---------------------------------------------------------------------------
# Oracle (target program) — priority-ordered, deterministic.
# ---------------------------------------------------------------------------

def oracle_action(context: Context) -> str:
    """Return the single correct action for a context (deterministic oracle)."""

    role = context["role"]
    friendliness = context["friendliness"]
    status = context["status"]
    mood = context["mood"]
    ctx = context["context"]

    # 1. Warm adults/elders are approachable.
    if role in ("adult", "elder") and friendliness == "warm":
        return "approach"
    # 2. Children are always approachable.
    if role == "child":
        return "approach"
    # 3. Strangers in conflict must be avoided.
    if ctx == "conflict" and status == "stranger":
        return "avoid"
    # 4. Upset in conflict → avoid.
    if mood == "upset" and ctx == "conflict":
        return "avoid"
    # 5. Upset requests → ask.
    if mood == "upset" and ctx == "request":
        return "ask"
    # 6. Cold in conflict → obey (defer).
    if ctx == "conflict" and friendliness == "cold":
        return "obey"
    # 7. Adult/elder requests → obey (needs composition of role + context).
    if role in ("adult", "elder") and ctx == "request":
        return "obey"
    # 8. Fallback.
    return "ask"


def validate_oracle() -> None:
    """Assert the oracle is total and consistent (no symmetric conflicts)."""

    for ctx in all_contexts():
        action = oracle_action(ctx)
        assert action in ACTIONS, f"oracle returned invalid action for {ctx}"


def is_success(context: Context, action: str) -> bool:
    return oracle_action(context) == action


# ---------------------------------------------------------------------------
# Eligibility filters for train / unseen axes.
# ---------------------------------------------------------------------------

def is_train_eligible(context: Context) -> bool:
    """Train excludes role=elder and the (mood=upset, context=conflict) pair."""

    if context["role"] == "elder":
        return False
    if context["mood"] == "upset" and context["context"] == "conflict":
        return False
    return True


def is_t1(context: Context) -> bool:
    """>T1: new value role=elder, excluding request (which is T3 territory)."""

    return context["role"] == "elder" and context["context"] != "request"


def is_t2(context: Context) -> bool:
    """>T2: new combination (mood=upset, context=conflict), known roles."""

    return (
        context["role"] != "elder"
        and context["mood"] == "upset"
        and context["context"] == "conflict"
    )


def is_t3(context: Context) -> bool:
    """>T3: composition — role=elder with context=request."""

    return context["role"] == "elder" and context["context"] == "request"


def classify(context: Context) -> str | None:
    """Return the axis membership ('t1'/'t2'/'t3') or None if in train space."""

    if is_t3(context):
        return "t3"
    if is_t1(context):
        return "t1"
    if is_t2(context):
        return "t2"
    return None


# ---------------------------------------------------------------------------
# Samplers.
# ---------------------------------------------------------------------------

def _sample(
    rng: Random,
    predicate: Callable[[Context], bool],
    n: int,
    allow_repeat: bool = True,
) -> list[Context]:
    pool = [ctx for ctx in all_contexts() if predicate(ctx)]

    if not allow_repeat or len(pool) >= n:
        chosen: list[Context] = []
        for _ in range(n):
            if not pool:
                break
            idx = rng.randrange(len(pool))
            chosen.append(pool[idx])
            if not allow_repeat:
                pool.pop(idx)
        return chosen

    # With replacement.
    return [pool[rng.randrange(len(pool))] for _ in range(n)]


@dataclass(slots=True)
class GeneratedSets:
    train: list[Context] = field(default_factory=list)
    known: list[Context] = field(default_factory=list)
    t1: list[Context] = field(default_factory=list)
    t2: list[Context] = field(default_factory=list)
    t3: list[Context] = field(default_factory=list)

    @property
    def test_all(self) -> list[Context]:
        return self.t1 + self.t2 + self.t3

    def snapshot(self) -> dict[str, Any]:
        return {
            "train": len(self.train),
            "known": len(self.known),
            "t1": len(self.t1),
            "t2": len(self.t2),
            "t3": len(self.t3),
        }


class AttributeTaskGenerator:
    """Generate train / known / unseen test sets from the symbolic space."""

    def __init__(
        self,
        *,
        seed: int = 0,
        train_size: int = 600,
        known_size: int = 120,
        t1_size: int = 60,
        t2_size: int = 60,
        t3_size: int = 40,
    ) -> None:
        self.rng = Random(seed)
        self.train_size = train_size
        self.known_size = known_size
        self.t1_size = t1_size
        self.t2_size = t2_size
        self.t3_size = t3_size

    def generate(self) -> GeneratedSets:
        validate_oracle()

        train = _sample(self.rng, is_train_eligible, self.train_size)
        known = _sample(self.rng, is_train_eligible, self.known_size)
        t1 = _sample(self.rng, is_t1, self.t1_size)
        t2 = _sample(self.rng, is_t2, self.t2_size)
        t3 = _sample(self.rng, is_t3, self.t3_size)

        return GeneratedSets(train=train, known=known, t1=t1, t2=t2, t3=t3)