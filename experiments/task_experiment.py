"""
Task experiment harness (attribute-tasks + behavioral program).

Implements the pre-registered protocol:

- Configurations: Base / Full / Null-bp / Ab-1..3 (ablation of the seven
  development operations) plus the external DecisionList baseline.
- Per-seed evaluation on T1/T2/T3 (unseen) and known (retention).
- Developmental curve: transfer as a function of train volume fraction.
- Chance (random) and ceiling (oracle) anchors on the same test sets.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Callable

from behavioral import Action, BehavioralProgram, Params
from behavioral.agent import BehavioralTaskAgent
from environment.attribute_tasks import (
    ACTIONS,
    AttributeTaskGenerator,
    Context,
    is_success,
    oracle_action,
)
from experiments.decision_list_baseline import DecisionListBaseline

SEED_BASE = 1000
REPLICATION_BASE = 9000

ALL_OPS = [
    "CREATE",
    "MODIFY",
    "GENERALIZE",
    "SPECIALIZE",
    "COMPOSE",
    "SPLIT",
    "DELETE",
]

# config_id -> granted operations (None = all).
ABLATIONS: dict[str, set[str] | None] = {
    "full": None,
    "ab_no_generalize": set(ALL_OPS) - {"GENERALIZE"},
    "ab_no_specialize": set(ALL_OPS) - {"SPECIALIZE"},
    "ab_no_compose": set(ALL_OPS) - {"COMPOSE"},
    "null_bp": None,  # same mechanism but receives no train outcomes
}


@dataclass(slots=True)
class RunResult:
    config: str
    seed: int
    transfer: dict[str, float] = field(default_factory=dict)
    overall: float = 0.0
    known_acc: float = 0.0
    complexity: float = 0.0
    n_rules: int = 0
    revisions: int = 0
    trace: list[dict] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "seed": self.seed,
            "transfer": self.transfer,
            "overall": self.overall,
            "known_acc": self.known_acc,
            "complexity": self.complexity,
            "n_rules": self.n_rules,
            "revisions": self.revisions,
        }


def _action(act: str) -> Action:
    return Action("primitive", act)


def run_behavioral_config(
    config_id: str,
    train: list[Context],
    test_sets: dict[str, list[Context]],
    known: list[Context],
    *,
    seed: int,
    params: Params | None = None,
    train_frac: float = 1.0,
    noise_eps: float | None = None,
) -> RunResult:
    """Train a bp agent on a (possibly fractional) train set and evaluate.

    When ``noise_eps`` is set (robustness diagnostic), train outcomes are
    flipped with probability ``noise_eps`` while ``null_bp`` is untouched:
    ``null_bp`` remains "random outcomes", ``noise_eps`` is "deterministic
    oracle outcome, occasionally distorted" (prereg §3, §10).
    """

    allowed = ABLATIONS[config_id]
    agent = BehavioralTaskAgent(
        seed=seed,
        params=params,
        allowed_operations=allowed,
    )

    size = int(len(train) * train_frac)
    experience = train[:size]

    revisions = 0
    noise = config_id == "null_bp"  # bp mechanism runs, outcomes are random
    noise_rng = Random(seed)
    eps = noise_eps or 0.0
    for ctx in experience:
        action = agent.select_action(ctx, explore=True)
        if noise:
            success = noise_rng.random() < 0.5
        else:
            success = is_success(ctx, action)
            if eps and noise_rng.random() < eps:
                success = not success
        trigger = agent.evaluate_and_learn(ctx, action, success)
        if trigger == "revised":
            revisions += 1

    # Frozen evaluation (no further learning on test).
    result = RunResult(config=config_id, seed=seed)
    overall_correct = 0
    overall_total = 0
    for axis, ctxs in test_sets.items():
        correct = 0
        for ctx in ctxs:
            act = agent.select_action(ctx, explore=False)
            if is_success(ctx, act):
                correct += 1
        result.transfer[axis] = correct / max(len(ctxs), 1)
        overall_correct += correct
        overall_total += len(ctxs)
    result.overall = overall_correct / max(overall_total, 1)

    known_correct = 0
    for ctx in known:
        act = agent.select_action(ctx, explore=False)
        if is_success(ctx, act):
            known_correct += 1
    result.known_acc = known_correct / max(len(known), 1)

    result.complexity = agent.program.complexity()
    result.n_rules = len(agent.program.rules)
    result.revisions = revisions
    result.trace = agent.program.trace()
    return result


def run_static_memory_baseline(
    train: list[Context],
    test_sets: dict[str, list[Context]],
    known: list[Context],
    *,
    seed: int,
) -> RunResult:
    """Base configuration: memorized context→action lookup, no bp.

    The baseline memorizes the majority action per exact context from train
    outcomes (memory + prediction without a revising behavioral program).
    """

    rng = Random(seed)
    memory: dict[str, dict[str, int]] = {}
    for ctx in train:
        for action in ACTIONS:
            if is_success(ctx, action):
                key = tuple(sorted(ctx.items()))
                memory.setdefault(key, {})
                memory[key][action] = memory[key].get(action, 0) + 1

    def predict(ctx: Context) -> str:
        key = tuple(sorted(ctx.items()))
        vote = memory.get(key)
        if vote:
            return max(vote.items(), key=lambda kv: kv[1])[0]
        return ACTIONS[rng.randrange(len(ACTIONS))]

    result = RunResult(config="base", seed=seed)
    correct = 0
    total = 0
    for axis, ctxs in test_sets.items():
        c = sum(1 for ctx in ctxs if is_success(ctx, predict(ctx)))
        result.transfer[axis] = c / max(len(ctxs), 1)
        correct += c
        total += len(ctxs)
    result.overall = correct / max(total, 1)

    kc = sum(1 for ctx in known if is_success(ctx, predict(ctx)))
    result.known_acc = kc / max(len(known), 1)
    return result


def chance_accuracy(test_sets: dict[str, list[Context]], seed: int) -> dict[str, float]:
    """Chance anchor: uniform random action over every test context."""

    rng = Random(seed)
    return {
        axis: sum(
            1 for ctx in ctxs if is_success(ctx, ACTIONS[rng.randrange(len(ACTIONS))])
        ) / max(len(ctxs), 1)
        for axis, ctxs in test_sets.items()
    }


def ceiling_accuracy(test_sets: dict[str, list[Context]]) -> dict[str, float]:
    """Ceiling anchor: perfect oracle performance (by definition 1.0)."""

    return {axis: 1.0 for axis in test_sets}


def run_single_seed(
    i: int,
    *,
    train_size: int,
    known_size: int,
    axis_sizes: dict[str, int],
    params: Params | None = None,
    train_frac: float = 1.0,
    seed_base: int = SEED_BASE,
) -> dict[str, Any]:
    """Run all configurations for one seed, return collected results."""

    env_seed = seed_base + i
    agent_seed = seed_base + 10_000 + i
    rng_seed = seed_base + 20_000 + i

    gen = AttributeTaskGenerator(
        seed=env_seed,
        train_size=train_size,
        known_size=known_size,
        t1_size=axis_sizes.get("t1", 60),
        t2_size=axis_sizes.get("t2", 60),
        t3_size=axis_sizes.get("t3", 40),
    )
    sets = gen.generate()
    test_sets = {"t1": sets.t1, "t2": sets.t2, "t3": sets.t3}

    results: dict[str, RunResult] = {}

    # Base (static memory).
    results["base"] = run_static_memory_baseline(
        sets.train, test_sets, sets.known, seed=agent_seed
    )

    # Behavioral configurations.
    for config_id in ABLATIONS:
        results[config_id] = run_behavioral_config(
            config_id,
            sets.train,
            test_sets,
            sets.known,
            seed=agent_seed,
            params=params,
            train_frac=train_frac,
        )

    # External baseline (DecisionList).
    dl = DecisionListBaseline(seed=env_seed)
    dl.fit(sets.train)
    dl_result = RunResult(config="decision_list", seed=env_seed)
    for axis, ctxs in test_sets.items():
        acc = dl.evaluate(ctxs)["accuracy"]
        dl_result.transfer[axis] = acc
    dl_result.overall = dl.evaluate(test_sets["t1"] + test_sets["t2"] + test_sets["t3"])["accuracy"]
    kc = dl.evaluate(sets.known)["accuracy"]
    dl_result.known_acc = kc
    results["decision_list"] = dl_result

    return {
        "seed": i,
        "results": {k: v.to_dict() for k, v in results.items()},
        "chance": chance_accuracy(test_sets, rng_seed),
        "ceiling": ceiling_accuracy(test_sets),
    }


def run_developmental_sample(
    i: int,
    *,
    train_size: int,
    known_size: int,
    axis_sizes: dict[str, int],
    params: Params | None = None,
    seed_base: int = SEED_BASE,
    fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
) -> dict[str, Any]:
    """Developmental curve: transfer vs fraction of train for the full config."""

    env_seed = seed_base + i
    agent_seed = seed_base + 10_000 + i
    gen = AttributeTaskGenerator(
        seed=env_seed,
        train_size=train_size,
        known_size=known_size,
        t1_size=axis_sizes.get("t1", 60),
        t2_size=axis_sizes.get("t2", 60),
        t3_size=axis_sizes.get("t3", 40),
    )
    sets = gen.generate()
    test_sets = {"t1": sets.t1, "t2": sets.t2, "t3": sets.t3}

    curve = {}
    for frac in fractions:
        r = run_behavioral_config(
            "full",
            sets.train,
            test_sets,
            sets.known,
            seed=agent_seed,
            params=params,
            train_frac=frac,
        )
        curve[str(frac)] = {
            "overall": r.overall,
            "transfer": r.transfer,
            "known_acc": r.known_acc,
        }
    return {"seed": i, "curve": curve}


def run_pilot(
    *,
    n_seeds: int = 20,
    train_size: int = 600,
    known_size: int = 120,
    axis_sizes: dict[str, int] | None = None,
    params: Params | None = None,
    out_dir: str | Path = "results",
) -> tuple[list[dict[str, Any]], Path]:
    """Small pilot run to sanity-check the experiment end-to-end."""

    axis_sizes = axis_sizes or {"t1": 60, "t2": 60, "t3": 40}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = [
        run_single_seed(
            i,
            train_size=train_size,
            known_size=known_size,
            axis_sizes=axis_sizes,
            params=params,
        )
        for i in range(1, n_seeds + 1)
    ]

    path = out / "pilot_task_run.json"
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False))
    return runs, path