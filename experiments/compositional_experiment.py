"""
Compositional experiment (Phase 1): honest comparison on genuinely-unseen tests.

Every learner — behavioral-program agent and value-classifier baselines —
receives the SAME single-seed uniform experience. No learner sees oracle
labels; success is observed only for the randomly-chosen action.

Test sets use the original compositional splits (T1: role=elder without
request, T2: mood=upset ∧ context=conflict, T3: elder ∧ request). These
are disjoint from train by full-attribute combinations.

The experiment asks: does the behavioral program outperform (or underperform)
simple baselines on compositional transfer, given the same information?

Also collects sample-complexity curves (full vs baselines) over train
fractions 0.1 / 0.25 / 0.5 / 1.0.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from random import Random
from typing import Any

import json
import time

import numpy as np

from behavioral.agent import BehavioralTaskAgent
from behavioral.program import Params
from behavioral.rules import Action as A
from environment.attribute_tasks import (
    ACTIONS,
    AttributeTaskGenerator,
    GeneratedSets,
    Context,
    is_success,
)
from experiments.compositional_baselines import (
    ValueLogisticRegression,
    ValueKNN,
    ValueDecisionTree,
    ValueDecisionList,
    build_experience,
    evaluate_predictor,
    balanced_accuracy,
)


# ---------------------------------------------------------------------------
# Result container.
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    config: str
    seed: int
    transfer: dict[str, float] = field(default_factory=dict)
    overall: float = 0.0
    balanced: float = 0.0
    axis_balanced: dict[str, float] = field(default_factory=dict)
    known_acc: float = 0.0
    n_rules: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "seed": self.seed,
            "transfer": dict(self.transfer),
            "overall": self.overall,
            "balanced": self.balanced,
            "axis_balanced": dict(self.axis_balanced),
            "known_acc": self.known_acc,
            "n_rules": self.n_rules,
        }


# ---------------------------------------------------------------------------
# Seed / size constants.
# ---------------------------------------------------------------------------

SEED_BASE = 1000
REPLICATION_BASE = 9000
TRAIN_SIZE = 2400
KNOWN_SIZE = 120
AXIS_SIZES = {"t1": 60, "t2": 60, "t3": 40}
FRACTIONS = (0.1, 0.25, 0.5, 1.0)
DEVELOP_INTERVAL = 50


# ---------------------------------------------------------------------------
# Agent training helper.
# ---------------------------------------------------------------------------

def _train_agent(
    train: list[Context],
    agent_seed: int,
    params: Params | None = None,
    develop_interval: int = DEVELOP_INTERVAL,
) -> BehavioralTaskAgent:
    """Train a BP agent on uniform random trials (all explored).

    The agent sees (context, random_action, oracle_outcome) exactly once
    per train context. develop() is called every ``develop_interval`` steps.
    No select_action is called during training — only observe_outcome.
    """
    agent = BehavioralTaskAgent(
        seed=agent_seed,
        params=params or Params(develop_interval=develop_interval),
    )
    rng = Random(agent_seed + 100_000)
    for i, ctx in enumerate(train):
        action = ACTIONS[rng.randrange(len(ACTIONS))]
        success = is_success(ctx, action)
        agent.program.observe_outcome(ctx, A("primitive", action), success, explored=True)
        if develop_interval and (i + 1) % develop_interval == 0:
            agent.program.develop()

    # Final develop after last interval chunk (if train_size % interval != 0).
    agent.program.develop()
    return agent


def _train_agent_with_curve(
    train: list[Context],
    agent_seed: int,
    train_frac: float,
    params: Params | None = None,
    develop_interval: int = DEVELOP_INTERVAL,
) -> BehavioralTaskAgent:
    """Train agent on a fractional subset of train (for sample-complexity)."""
    n = int(len(train) * train_frac)
    return _train_agent(train[:n], agent_seed, params, develop_interval)


# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------

def _bp_eval(
    agent: BehavioralTaskAgent,
    test_sets: dict[str, list[Context]],
    known: list[Context],
    seed: int,
) -> RunResult:
    """Evaluate agent on test + known; no further learning."""
    r = RunResult(config="full_bp", seed=seed)
    correct = 0
    total = 0
    for axis, ctxs in test_sets.items():
        c = 0
        for ctx in ctxs:
            act = agent.select_action(ctx, explore=False)
            if is_success(ctx, act):
                c += 1
        r.transfer[axis] = c / max(len(ctxs), 1)
        r.axis_balanced[axis] = _balanced_for(agent, ctxs, seed)
        correct += c
        total += len(ctxs)
    r.overall = correct / max(total, 1)
    r.balanced = _balanced_for(agent, test_sets["t1"] + test_sets["t2"] + test_sets["t3"], seed)
    kc = sum(
        1 for ctx in known
        if is_success(ctx, agent.select_action(ctx, explore=False))
    )
    r.known_acc = kc / max(len(known), 1)
    r.n_rules = len(agent.program.rules)
    return r


def _balanced_for(predictable, contexts: list[Context], seed: int) -> float:
    class _Wrap:
        def __init__(self, predictable):
            self._p = predictable
        def predict(self, ctx):
            return self._p.select_action(ctx, explore=False)
    return balanced_accuracy(_Wrap(predictable), contexts)


def _baseline_eval(
    model,
    name: str,
    test_sets: dict[str, list[Context]],
    known: list[Context],
    seed: int,
) -> RunResult:
    r = RunResult(config=name, seed=seed)
    correct = 0
    total = 0
    for axis, ctxs in test_sets.items():
        c = sum(1 for ctx in ctxs if is_success(ctx, model.predict(ctx)))
        r.transfer[axis] = c / max(len(ctxs), 1)
        r.axis_balanced[axis] = balanced_accuracy(model, ctxs)
        correct += c
        total += len(ctxs)
    r.overall = correct / max(total, 1)
    r.balanced = balanced_accuracy(model, test_sets["t1"] + test_sets["t2"] + test_sets["t3"])
    kc = sum(1 for ctx in known if is_success(ctx, model.predict(ctx)))
    r.known_acc = kc / max(len(known), 1)
    return r


# ---------------------------------------------------------------------------
# Baseline constructors.
# ---------------------------------------------------------------------------

def _make_baselines(
    experience: list[tuple[Context, str, bool]],
    n_train: int,
) -> dict[str, Any]:
    """Fit all baselines on the same uniform experience."""
    k_val = max(3, min(15, int(np.sqrt(n_train)) + 1))
    dt_depth = max(3, min(10, int(np.sqrt(n_train)) + 1))
    models = {
        "ValueLogReg": ValueLogisticRegression(seed=42),
        "ValuekNN": ValueKNN(k=k_val),
        "ValueTree": ValueDecisionTree(max_depth=dt_depth, min_samples=max(4, n_train // 30)),
        "ValueDList": ValueDecisionList(min_coverage=max(3, n_train // 60), ratio=0.5),
    }
    for m in models.values():
        m.fit(experience)
    return models


# ---------------------------------------------------------------------------
# Per-seed full run (compositional, fair).
# ---------------------------------------------------------------------------

def run_compositional_seed(
    i: int,
    *,
    seed_base: int = SEED_BASE,
    train_size: int = TRAIN_SIZE,
    known_size: int = KNOWN_SIZE,
    axis_sizes: dict[str, int] | None = None,
    train_frac: float = 1.0,
) -> dict[str, Any]:
    """Run one seed: agent + baselines on identical uniform experience."""
    axis_sizes = axis_sizes or AXIS_SIZES
    env_seed = seed_base + i
    agent_seed = seed_base + 10_000 + i
    rng_seed = seed_base + 20_000 + i

    gen = AttributeTaskGenerator(
        seed=env_seed,
        train_size=train_size,
        known_size=known_size,
        t1_size=axis_sizes["t1"],
        t2_size=axis_sizes["t2"],
        t3_size=axis_sizes["t3"],
    )
    sets = gen.generate()
    test_sets = {"t1": sets.t1, "t2": sets.t2, "t3": sets.t3}

    n = int(len(sets.train) * train_frac)
    train_sub = sets.train[:n]

    # --- Uniform experience for all learners ---
    experience = build_experience(train_sub, agent_seed + 55555)

    # --- Behavioral program agent ---
    bp = _train_agent_with_curve(sets.train, agent_seed, train_frac)
    bp_result = _bp_eval(bp, test_sets, sets.known, env_seed)
    bp_result.config = "full_bp"

    # --- Baselines ---
    models = _make_baselines(experience, n)
    baseline_results = {}
    for name, model in models.items():
        r = _baseline_eval(model, name, test_sets, sets.known, env_seed)
        baseline_results[name] = r

    # --- Chance ---
    class _Chance:
        def __init__(self, rng):
            self.rng = rng
        def predict(self, ctx):
            return ACTIONS[self.rng.randrange(len(ACTIONS))]
    chance_model = _Chance(Random(rng_seed))
    chance_bal = balanced_accuracy(chance_model, test_sets["t1"] + test_sets["t2"] + test_sets["t3"])

    # --- Ceiling (1.0 by definition) ---
    class _Ceiling:
        def predict(self, ctx):
            from environment.attribute_tasks import oracle_action
            return oracle_action(ctx)
    ceiling_bal = balanced_accuracy(_Ceiling(), test_sets["t1"] + test_sets["t2"] + test_sets["t3"])

    results = {
        "full_bp": bp_result.to_dict(),
        **{n: r.to_dict() for n, r in baseline_results.items()},
        "chance": {"overall": chance_bal, "balanced": chance_bal},
        "ceiling": {"overall": ceiling_bal, "balanced": ceiling_bal},
    }

    return {
        "seed": i,
        "train_size": n,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Aggregate runs.
# ---------------------------------------------------------------------------

def run_compositional_study(
    *,
    seed_base: int = SEED_BASE,
    n_seeds: int = 24,
    out_dir: str | Path = "results",
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "main" if seed_base == SEED_BASE else "replication"

    t0 = time.time()
    runs = []
    for i in range(1, n_seeds + 1):
        runs.append(run_compositional_seed(i, seed_base=seed_base))
        if i % 4 == 0:
            elapsed = time.time() - t0
            print(f"  [{tag}] {i}/{n_seeds}  {elapsed:.0f}s", flush=True)

    path = out / f"compositional_{tag}.json"
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False))
    print(f"  saved {path.name} ({time.time()-t0:.0f}s total)")
    return path


def run_sample_complexity(
    *,
    seed_base: int = SEED_BASE,
    n_seeds: int = 24,
    fractions: tuple[float, ...] = FRACTIONS,
    train_size: int = 1200,
    out_dir: str | Path = "results",
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "main" if seed_base == SEED_BASE else "replication"

    t0 = time.time()
    runs = []
    for i in range(1, n_seeds + 1):
        curves: dict[str, dict[str, Any]] = {}
        for frac in fractions:
            r = run_compositional_seed(
                i, seed_base=seed_base, train_frac=frac, train_size=train_size
            )
            curves[str(frac)] = {
                "overall": r["results"]["full_bp"]["overall"],
                "balanced": r["results"]["full_bp"].get("balanced", 0.0),
                "known_acc": r["results"]["full_bp"].get("known_acc", 0),
                "baselines": {
                    n: res["balanced"]
                    for n, res in r["results"].items()
                    if n not in ("chance", "ceiling", "full_bp")
                },
                "baselines_overall": {
                    n: res["overall"]
                    for n, res in r["results"].items()
                    if n not in ("chance", "ceiling", "full_bp")
                },
            }
        runs.append({"seed": i, "curves": curves})
        if i % 4 == 0:
            elapsed = time.time() - t0
            print(f"  [{tag}] {i}/{n_seeds}  {elapsed:.0f}s", flush=True)

    path = out / f"complexity_{tag}.json"
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False))
    print(f"  saved {path.name} ({time.time()-t0:.0f}s total)")
    return path


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Compositional Phase 1 study ===")
    run_compositional_study(seed_base=SEED_BASE)
    run_compositional_study(seed_base=REPLICATION_BASE)
    run_sample_complexity(seed_base=SEED_BASE, n_seeds=8, train_size=1200, fractions=FRACTIONS)
    run_sample_complexity(seed_base=REPLICATION_BASE, n_seeds=8, train_size=1200, fractions=FRACTIONS)
    print("Done.")
