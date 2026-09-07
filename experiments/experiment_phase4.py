"""
Phase 4 experiment: Structural-OOD composition generalization (sub-protocol).

Design constraint (see environment.attribute_tasks_v4): a held-out arity-3
rule whose full region has zero train support is unlearnable by any
coverage-based inducer. The sub-protocol instead holds out the exact triple
as a "distinct class" while the *pair* covering it is a training rule
(``calm∧guard -> approach``). The never-observed triple region
``calm∧guard∧night -> approach`` is then transferable only by compositional
generalisation of the pair — memorisers (kNN) reach only seen sub-patterns.

Metrics (per seed):
  * heldout_acc  — accuracy on the NEVER-observed triple region (the OOD probe);
  * row balanced — balanced accuracy on a random reference row of the pool;
  * known_acc    — accuracy on the held-out internal known set;
  * pair_recovered — was the covering pair rule (calm∧guard -> approach)
    materialised as an ACTIVE conjunction (interpretability);
  * ablations: `ab_no_compose` (pair induction impossible) and `null_bp`
    (random outcomes) — expected to FAIL heldout_acc, isolating composition
    as the mechanism of OOD generalisation.

N=24 main + 24 replication (SEED_BASE / REPLICATION_BASE).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from random import Random
from typing import Any

import json
import time

from behavioral.program import Params
from behavioral.rules import Action as A, Literal
from behavioral.program_v3 import BehavioralProgramV3
from environment.attribute_tasks_v4 import (
    ACTIONS_V4,
    ATTRIBUTE_DOMAINS_V4,
    V4_ATTRIBUTES,
    ContextV4,
    AttributeTaskGeneratorV4,
    HELDOUT_RULE_V4,
    TRAIN_RULES_V4,
    all_contexts_v4,
    is_success_v4,
    oracle_action_v4,
)
from experiments.compositional_baselines import (
    Encoder,
    ValueDecisionList,
    ValueDecisionTree,
    ValueKNN,
    ValueLogisticRegression,
    ValueMultiDecisionList,
    ValueRandomForest,
    balanced_accuracy,
    evaluate_predictor,
)

ENCODER_V4 = Encoder(
    attributes=list(V4_ATTRIBUTES),
    domains=ATTRIBUTE_DOMAINS_V4,
    actions=ACTIONS_V4,
)

PAIR_RULE_V4 = ("approach", (("emotion", "calm"), ("task", "guard")))

_ALL_OPS = ["CREATE", "MODIFY", "GENERALIZE", "SPECIALIZE", "COMPOSE", "SPLIT", "DELETE"]


@dataclass
class RunResultV4:
    config: str
    seed: int
    heldout_acc: float = 0.0
    row_overall: float = 0.0
    row_balanced: float = 0.0
    known_acc: float = 0.0
    n_active: int = 0


def _majority_action(experience) -> str:
    """Rational fallback: action maximising expected success over experience.

    (Phase 3 used count-majority; here the four action counts are a uniform
    four-way tie, so count-majority would be arbitrary luck. Expected-success
    picks the action whose marginal success is best — by construction the
    environmental default — and is deterministic per seed.)
    """
    tally: dict[str, list[int]] = {}
    for _, a, s, *_ in experience:
        t = tally.setdefault(a, [0, 0])
        t[0] += 1
        t[1] += 1 if s else 0
    best = max(tally, key=lambda a: (tally[a][1] / tally[a][0] if tally[a][0] else 0.0, a))
    return best


def _train_program_v4(
    train: list[ContextV4],
    agent_seed: int,
    experience: list[tuple[ContextV4, str, bool, bool]],
    *,
    allowed_operations: list[str] | None = None,
    randomize_outcomes: bool = False,
) -> BehavioralProgramV3:
    if randomize_outcomes:
        rng = Random(agent_seed + 77_777)
        experience = [
            (ctx, action, rng.random() < 0.25, True)
            for ctx, action, _success, _flg in experience
        ]
    prog = BehavioralProgramV3(
        attributes=list(V4_ATTRIBUTES),
        domains=ATTRIBUTE_DOMAINS_V4,
        actions=ACTIONS_V4,
        params=Params(
            develop_interval=250,
            min_coverage=4,
            validate_accuracy=0.85,
            validate_margin=0.05,
            explore_rate=0.0,
        ),
        seed=agent_seed,
        max_arity=4,
        allowed_operations=allowed_operations,
    )
    prog.trials = experience
    prog.develop()
    return prog


class _ChancePredictor:
    """True chance: a fresh random action per context (not one fixed action)."""

    def __init__(self, seed: int, actions: list[str]) -> None:
        self._rng = Random(seed + 0xC0FFEE)
        self._actions = actions

    def predict(self, _ctx) -> str:
        return self._actions[self._rng.randrange(len(self._actions))]


class _Ceil:
    def predict(self, ctx) -> str:
        return oracle_action_v4(ctx)


class _AgentPredictorV4:
    """Deterministic agent policy: ACTIVE rules, specificity-first, majority fallback."""

    def __init__(self, prog: BehavioralProgramV3, fallback: str) -> None:
        self.prog = prog
        self.fallback = fallback

    def predict(self, ctx: ContextV4) -> str:
        cands = [r for r in self.prog.active_rules if r.condition.matches(ctx)]
        if not cands:
            return self.fallback
        best = max(cands, key=lambda r: (-r.condition.specificity(), -r.evidence.total))
        return best.action.name


def _accuracy(model, ctxs: list[ContextV4]) -> float:
    if not ctxs:
        return 0.0
    return sum(1 for c in ctxs if is_success_v4(c, model.predict(c))) / len(ctxs)


def _eval_v4(model, test_heldout, test_row, known, name: str, seed: int) -> RunResultV4:
    r = RunResultV4(config=name, seed=seed)
    r.heldout_acc = _accuracy(model, test_heldout)
    ev = evaluate_predictor(model, test_row, success=is_success_v4, oracle=oracle_action_v4)
    r.row_overall = ev["overall"]
    r.row_balanced = ev["balanced"]
    r.known_acc = _accuracy(model, known)
    return r


def _pair_recovered(prog: BehavioralProgramV3) -> dict[str, Any]:
    """Interpretability probe: the covering pair rule of the held-out triple."""
    action, sig = PAIR_RULE_V4
    sig_lits = frozenset(Literal(k, "=", v) for k, v in sig)
    found = None
    for r in prog.active_rules:
        if r.action.name != action:
            continue
        rlits = frozenset(r.condition.literals)
        if rlits.issuperset(sig_lits) and len(rlits) == 2:
            found = f"{action}[{sorted(str(l) for l in r.condition.literals)}]"
            break
    # count how often full-population held-out region would be correctly
    # covered by ACTIVE rules firing (specificity-first), for interpretability.
    held_ctxs = [c for c in all_contexts_v4()
                 if all(c.get(k) == v for k, v in HELDOUT_RULE_V4[1])]
    correct = sum(1 for c in held_ctxs
                  if oracle_action_v4(c) == _AgentPredictorV4(prog, _majority_fallback(prog)).predict(c))
    return {"found": found, "heldout_region_correct_full": correct / len(held_ctxs)}


def _majority_fallback(prog):
    return Counter(a for _, a, *_ in prog.trials).most_common(1)[0][0]


SEED_BASE = 7000
REPLICATION_BASE = 17000
N_SEEDS = 24


def run_phase4_seed(
    i: int,
    *,
    seed_base: int = SEED_BASE,
    train_size: int = 600,
    known_size: int = 200,
    heldout_test_size: int = 60,
    row_test_size: int = 200,
) -> dict[str, Any]:
    env_seed = seed_base + i
    agent_seed = seed_base + 10_000 + i

    gen = AttributeTaskGeneratorV4(
        seed=env_seed,
        train_size=train_size,
        known_size=known_size,
        heldout_test_size=heldout_test_size,
        row_test_size=row_test_size,
    )
    sets = gen.generate()

    rng = Random(agent_seed + 55_555)
    experience = []
    for ctx in sets.train:
        action = ACTIONS_V4[rng.randrange(len(ACTIONS_V4))]
        experience.append((ctx, action, is_success_v4(ctx, action), True))

    fallback = _majority_action(experience)

    def _build(predictor, name):
        return _eval_v4(predictor, sets.test_heldout, sets.test_row, sets.known, name, i)

    prog = _train_program_v4(sets.train, agent_seed, experience)
    agent = _AgentPredictorV4(prog, fallback)
    bp = _build(agent, "full_bp")
    bp.n_active = len(prog.rules)

    no_compose = [op for op in _ALL_OPS if op != "COMPOSE"]
    nc_prog = _train_program_v4(
        sets.train, agent_seed + 1, experience, allowed_operations=no_compose
    )
    null_prog = _train_program_v4(
        sets.train, agent_seed + 2, experience, randomize_outcomes=True
    )
    nc = _build(_AgentPredictorV4(nc_prog, fallback), "ab_no_compose")
    null = _build(_AgentPredictorV4(null_prog, fallback), "null_bp")

    n_train = len(experience)
    k_val = max(3, min(21, int(n_train ** 0.5)))
    compo_min_cov = max(8, n_train // 200)
    models = {
        "ValuekNN": ValueKNN(k=k_val, encoder=ENCODER_V4),
        "ValueDListML": ValueMultiDecisionList(
            max_arity=4, min_coverage=compo_min_cov, ratio=0.85, encoder=ENCODER_V4
        ),
        "ValueTree": ValueDecisionTree(
            max_depth=10, min_samples=max(8, n_train // 40), encoder=ENCODER_V4
        ),
        "ValueForest": ValueRandomForest(
            n_estimators=25, max_depth=10,
            min_samples=max(8, n_train // 40), max_features=6,
            seed=42, encoder=ENCODER_V4,
        ),
    }
    baseline = {}
    for name, model in models.items():
        model.fit(experience)
        baseline[name] = _build(model, name)

    def _chance_predict(_ctx: ContextV4) -> str:
        return ACTIONS_V4[rng.randrange(len(ACTIONS_V4))]

    chance = _accuracy(_ChancePredictor(env_seed, ACTIONS_V4), sets.test_heldout)
    ceiling = _accuracy(_Ceil(), sets.test_heldout)

    return {
        "seed": i,
        "results": {
            "full_bp": _serialize(bp),
            "ab_no_compose": _serialize(nc),
            "null_bp": _serialize(null),
            **{n: _serialize(r) for n, r in baseline.items()},
            "chance": {"heldout_acc": chance},
            "ceiling": {"heldout_acc": ceiling},
        },
        "audit": _pair_recovered(prog),
    }


def _serialize(r: RunResultV4) -> dict[str, Any]:
    return {
        "config": r.config,
        "seed": r.seed,
        "heldout_acc": r.heldout_acc,
        "row_overall": r.row_overall,
        "row_balanced": r.row_balanced,
        "known_acc": r.known_acc,
        "n_active": r.n_active,
    }


def run_phase4_study(
    *,
    seed_base: int = SEED_BASE,
    n_seeds: int = N_SEEDS,
    out_dir: str | Path = "results",
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "main" if seed_base == SEED_BASE else "replication"
    t0 = time.time()
    runs = []
    for i in range(1, n_seeds + 1):
        runs.append(run_phase4_seed(i, seed_base=seed_base))
        if i % 4 == 0:
            print(f"  [phase4-{tag}] {i}/{n_seeds}  {time.time() - t0:.0f}s", flush=True)
    path = out / f"phase4_{tag}.json"
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False))
    print(f"  saved {path.name} ({time.time() - t0:.0f}s total)")
    return path


def summarize_study(path: Path | str) -> None:
    runs = json.loads(Path(path).read_text(encoding="utf-8"))
    names = [
        "full_bp", "ab_no_compose", "null_bp",
        "ValuekNN", "ValueDListML", "ValueTree", "ValueForest",
        "chance", "ceiling",
    ]
    print(f"\n=== Phase 4 summary ({len(runs)} seeds) ===")
    print(f"{'model':<14}{'heldout':>10}{'rowBal':>10}{'rowOA':>10}{'known':>10}")
    for name in names:
        vals = defaultdict(list)
        for run in runs:
            res = run["results"][name]
            for k in ("heldout_acc", "row_balanced", "row_overall", "known_acc"):
                if k in res:
                    vals[k].append(res[k])
        if not vals:
            continue
        row = " ".join(
            f"{sum(vals[k]) / len(vals[k]):>10.3f}" if vals[k] else "-".rjust(10)
            for k in ("heldout_acc", "row_balanced", "row_overall", "known_acc")
        )
        print(f"{name:<14}{row}")
    print("\npair rule cover probe (seeds):")
    print("  pair calm^guard->approach materialised:", sum(1 for r in runs if r["audit"]["found"]), "/", len(runs))
    mean_cov = sum(r["audit"]["heldout_region_correct_full"] for r in runs) / len(runs)
    print(f"  full-pop held-out region correct (BP policy): {mean_cov:.3f}")


def paired_stats(path: Path | str, n_perm: int = 20_000) -> None:
    import numpy as np
    runs = json.loads(Path(path).read_text(encoding="utf-8"))
    bp = np.array([r["results"]["full_bp"]["heldout_acc"] for r in runs])
    names = ["ValuekNN", "ValueDListML", "ValueTree", "ValueForest",
             "ab_no_compose", "null_bp"]

    def perm_p(a, b, rng):
        d = np.abs((a - b).mean())
        obs = np.concatenate([a, b])
        cnt = 0
        for _ in range(n_perm):
            p = rng.permutation(obs)
            s = p[:a.size].mean() - p[a.size:].mean()
            cnt += abs(s) >= d
        return (cnt + 1) / (n_perm + 1)

    rng = np.random.default_rng(0)
    print(f"\n== paired heldout-acc stats ({len(runs)} seeds, {path}) ==")
    for name in names:
        v = np.array([r["results"][name]["heldout_acc"] for r in runs])
        diff = (bp - v).mean()
        dz = diff / (bp - v).std(ddof=1) if (bp - v).std(ddof=1) > 0 else float("inf")
        p = perm_p(bp, v, rng)
        print(f"full_bp vs {name:<13} diff {diff:+.3f}  dz {dz:+5.2f}  p {p:.4f}")


if __name__ == "__main__":
    print("=== Phase 4 study ===")
    run_phase4_study(seed_base=SEED_BASE, n_seeds=N_SEEDS)
    run_phase4_study(seed_base=REPLICATION_BASE, n_seeds=N_SEEDS)
    summarize_study("results/phase4_main.json")
    paired_stats("results/phase4_main.json")
    paired_stats("results/phase4_replication.json")
    print("Done.")