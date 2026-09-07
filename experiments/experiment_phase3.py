"""
Phase 3 experiment: rare compositional rules demand composition (arity >= 2).

Environment v3 (attribute_tasks_v3): 8 attributes, 4608 contexts, 5 actions.
The oracle is priority-ordered:

  * 4 rare rules of arity 3-4 whose *single* literals are individually
    ambiguous (precision 0.16-0.68 at arity 1) — only the full conjunction
    disambiguates them (precision >= 0.95 at discovered arity);
  * 3 simple rules that are ALSO only decisive as pairs (e.g. insider∧calm),
    so an honest learner must compose everywhere.

Every learner — BehavioralProgramV3 and the value baselines (LogReg, kNN,
Tree, DecisionList) — receives the SAME uniform random experience
(context, uniform action, oracle outcome), one trial per train context,
drawn from a 65% pool of the full space. Evaluation happens on contexts
drawn from the disjoint 35% test pool (T1/T2/T3), plus the ``known`` set.

The experiment asks:
  * does the agent discover the rare conjunctions (COMPOSE, arity 3-4)?
  * does it beat the four baselines (esp. the one-literal DecisionList and
    sparse-neighbourhood kNN) on balanced accuracy over genuinely-unseen
    contexts?
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

import json
import time

from behavioral.program import Params
from behavioral.rules import Action as A, Literal
from behavioral.program_v3 import BehavioralProgramV3
from environment.attribute_tasks_v3 import (
    ACTIONS_V3,
    ATTRIBUTE_DOMAINS_V3,
    V3_ATTRIBUTES,
    AttributeTaskGeneratorV3,
    ContextV3,
    is_success_v3,
    oracle_action_v3,
)
from experiments.compositional_baselines import (
    Encoder,
    ValueCompoRuleLearner,
    ValueDecisionList,
    ValueDecisionTree,
    ValueKNN,
    ValueLogisticRegression,
    ValueMultiDecisionList,
    ValueRandomForest,
    balanced_accuracy,
    evaluate_predictor,
)


# ---------------------------------------------------------------------------
# Schema / encoder (injected into every baseline).
# ---------------------------------------------------------------------------

ENCODER_V3 = Encoder(
    attributes=list(V3_ATTRIBUTES),
    domains=ATTRIBUTE_DOMAINS_V3,
    actions=ACTIONS_V3,
)


# ---------------------------------------------------------------------------
# Oracle rule signatures (mirrors attribute_tasks_v3) — for composition audit.
# ---------------------------------------------------------------------------

RARE_RULES_V3: list[tuple[str, tuple[tuple[str, str], ...]]] = [
    ("offer", (("season", "winter"), ("location", "public"), ("time", "night"))),
    ("ask", (("location", "school"), ("role", "child"), ("mood", "upset"))),
    (
        "approach",
        (
            ("friendliness", "warm"),
            ("status", "stranger"),
            ("season", "spring"),
            ("time", "dawn"),
        ),
    ),
    ("offer", (("role", "adult"), ("context", "greeting"), ("season", "autumn"))),
]

SIMPLE_RULES_V3: list[tuple[str, tuple[tuple[str, str], ...]]] = [
    ("approach", (("status", "insider"), ("mood", "calm"))),
    ("avoid", (("context", "conflict"),)),
    ("obey", (("friendliness", "cold"), ("role", "elder"))),
]

# Formal "oracle compositional rules": every oracle rule of arity >= 2 that is
# representable in the hypothesis space (equality literals). These are the
# targets of rule recovery; ``avoid <- conflict`` (arity 1) is non-compositional
# and excluded. ``cold∧elder -> obey`` is representable but (as shown) not
# discoverable because of priority overlap — it stays in the denominator to
# keep recall honest.
COMPOSITIONAL_TARGETS_V3: list[tuple[str, tuple[tuple[str, str], ...]]] = [
    *RARE_RULES_V3,
    ("approach", (("status", "insider"), ("mood", "calm"))),
    ("obey", (("friendliness", "cold"), ("role", "elder"))),
]


def matches_signature(ctx: ContextV3, signature: tuple[tuple[str, str], ...]) -> bool:
    return all(ctx.get(k) == v for k, v in signature)


def is_rare_context_v3(ctx: ContextV3) -> bool:
    return any(matches_signature(ctx, sig) for _, sig in RARE_RULES_V3)


# ---------------------------------------------------------------------------
# Result container.
# ---------------------------------------------------------------------------

@dataclass
class RunResultV3:
    config: str
    seed: int
    transfer: dict[str, float] = field(default_factory=dict)
    overall: float = 0.0
    balanced: float = 0.0
    rare_balanced: float = 0.0
    nonrare_balanced: float = 0.0
    rare_accuracy: float = 0.0
    known_acc: float = 0.0
    n_active: int = 0

    @property
    def name(self) -> str:
        return self.config


FALLBACK_ACTION = "ask"  # oracle default branch and largest class

_ALL_OPS = ["CREATE", "MODIFY", "GENERALIZE", "SPECIALIZE", "COMPOSE", "SPLIT", "DELETE"]


def _majority_action(experience) -> str:
    c = Counter(a for _, a, *_ in experience)
    return c.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Agent training (same uniform experience; direct induction, no exemplars).
# ---------------------------------------------------------------------------

def _train_program_v3(
    train: list[ContextV3],
    agent_seed: int,
    experience: list[tuple[ContextV3, str, bool, bool]],
    *,
    params: Params | None = None,
    develop_interval: int = 250,
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
        attributes=list(V3_ATTRIBUTES),
        domains=ATTRIBUTE_DOMAINS_V3,
        actions=ACTIONS_V3,
        params=params or Params(
            develop_interval=develop_interval,
            min_coverage=4,
            validate_accuracy=0.85,
            validate_margin=0.05,
            explore_rate=0.0,
        ),
        seed=agent_seed,
        max_arity=4,
        allowed_operations=allowed_operations,
    )
    # Uniform experience exactly as baselines see it (no oracle labels).
    prog.trials = experience
    # A single induction pass over all uniform explore data (batch setting).
    prog.develop()
    return prog


class _AgentPredictor:
    """Deterministic agent policy: ACTIVE rules, specificity-first, majority fallback."""

    def __init__(self, prog: BehavioralProgramV3, fallback: str) -> None:
        self.prog = prog
        self.fallback = fallback

    def predict(self, ctx: ContextV3) -> str:
        cands = [r for r in self.prog.active_rules if r.condition.matches(ctx)]
        if not cands:
            return self.fallback
        best = max(cands, key=lambda r: (-r.condition.specificity(), -r.evidence.total))
        return best.action.name


# ---------------------------------------------------------------------------
# Composition audit + formal rule-recovery metrics.
# ---------------------------------------------------------------------------

def rule_recovery(
    prog: BehavioralProgramV3,
    contexts: list[ContextV3] | None = None,
    oracle=oracle_action_v3,
    fallback: str = FALLBACK_ACTION,
) -> dict[str, Any]:
    """Lab-grade recovery + false-discovery metrics of compositional rules.

    Two orthogonal measurements over ACTIVE rules of arity >= 2. Both use the
    FULL context population (not just the train sample), so "discovery" means
    structure that is true in the population, not in the training batch.

    **Structural call (recall).** Target set = every oracle compositional rule
    (arity >= 2, equality literals, COMPOSITIONAL_TARGETS_V3):

      * exact recovery   — some ACTIVE rule equals the target signature;
      * subset recovery  — some ACTIVE rule's literal set is a subset of the
                           signature (same action) — the region is captured,
                           not necessarily by its maximal signature.

    **Behavioral FDR (policy-level precision).** Under the agent's actual
    decision procedure (specificity-first, tie-break by evidence — identical
    to ``_AgentPredictor``), a rule is a *false discovery* if it ever wins the
    resolution on a full-population context and the oracle differs from its
    action. This is the overfitting floor of the retained composition: it
    counts rules whose firing actively costs accuracy, including cases where
    a broad rule wins only because a more specific competitor was not
    materialised in this seed (priority-rule overlap makes plain region
    purity meaningless here). ``fdr = harmful / compositional-active``.
    """
    if contexts is None:
        from environment.attribute_tasks_v3 import all_contexts_v3 as _all
        contexts = _all()

    targets: list[tuple[str, frozenset[Literal]]] = [
        (action, frozenset(Literal(k, "=", v) for k, v in sig))
        for action, sig in COMPOSITIONAL_TARGETS_V3
    ]
    compo = [r for r in prog.active_rules if len(r.condition.literals) >= 2]

    # --- resolution exactly as the agent predictor does it ---
    def _winner(ctx: ContextV3):
        cands = [r for r in prog.active_rules if r.condition.matches(ctx)]
        if not cands:
            return None
        return max(cands, key=lambda r: (-r.condition.specificity(), -r.evidence.total))

    win_right: dict[int, int] = defaultdict(int)
    win_wrong: dict[int, int] = defaultdict(int)
    winners = [None] * len(contexts)
    for i, ctx in enumerate(contexts):
        w = _winner(ctx)
        winners[i] = w
        if w is None:
            continue
        key = id(w)
        if oracle(ctx) == w.action.name:
            win_right[key] += 1
        else:
            win_wrong[key] += 1

    exact = [False] * len(targets)
    subset = [False] * len(targets)
    n_harmful = 0
    total_fire = 0
    total_fire_correct = 0
    for rule in compo:
        key = id(rule)
        w = win_right[key] + win_wrong[key]
        total_fire += w
        total_fire_correct += win_right[key]
        if win_wrong[key] > 0:
            n_harmful += 1
        rlits = frozenset(rule.condition.literals)
        for i, (action, sig) in enumerate(targets):
            if rule.action.name != action:
                continue
            if rlits == sig:
                exact[i] = True
                subset[i] = True
            elif rlits.issubset(sig):
                subset[i] = True

    n_active_all = len(prog.active_rules)
    mean_arity = (
        sum(len(r.condition.literals) for r in prog.active_rules) / max(n_active_all, 1)
    )
    n_atomic = sum(1 for r in prog.active_rules if len(r.condition.literals) == 1)
    n_compo = len(compo)

    return {
        "n_targets": len(targets),
        "n_compositional_active": n_compo,
        "n_active": n_active_all,
        "n_atomic": n_atomic,
        "n_consistent": n_compo - n_harmful,
        "n_inconsistent": n_harmful,
        "precision": 1.0 - n_harmful / max(n_compo, 1),
        "fdr": n_harmful / max(n_compo, 1),
        "winner_accuracy": total_fire_correct / max(total_fire, 1),
        "n_fire_contexts": total_fire,
        "mean_active_arity": mean_arity,
        "exact_recovered": exact,
        "subset_recovered": subset,
        "n_exact_recovered": sum(exact),
        "n_subset_recovered": sum(subset),
        "exact_rate": sum(exact) / len(targets),
        "subset_rate": sum(subset) / len(targets),
        "targets": [f"{action}[{len(sig)}]" for action, sig in targets],
        "targets_exact": [
            f"{action}_{str(tuple(k + '=' + v for k, v in sig))}"
            for action, sig in COMPOSITIONAL_TARGETS_V3
        ],
    }


def audit_composition(prog: BehavioralProgramV3) -> dict[str, Any]:
    """Backward-compatible per-rule audit (seen/not seen across seeds)."""
    recovery = rule_recovery(prog)
    active = prog.active_rules
    result = {
        "n_active": len(active),
        "n_total": len(prog.rules),
        "arity_histogram": Counter(len(r.condition.literals) for r in active),
        "mean_arity": (
            sum(len(r.condition.literals) for r in active) / max(len(active), 1)
        ),
        "rare_discovered": {},
        "simple_discovered": {},
    }

    def captured(sig_lits, action: str) -> bool:
        sig = {lit for lit in sig_lits}
        for r in active:
            if r.action.name != action:
                continue
            rlits = set(r.condition.literals)
            if rlits and rlits.issubset(sig) and len(rlits) >= 2:
                return True
        return False

    for action, sig in RARE_RULES_V3:
        sig_lits = tuple(Literal(k, "=", v) for k, v in sig)
        result["rare_discovered"][action + str(sig)] = captured(sig_lits, action)
    for action, sig in SIMPLE_RULES_V3:
        sig_lits = tuple(Literal(k, "=", v) for k, v in sig)
        result["simple_discovered"][action + str(sig)] = captured(sig_lits, action)
    result["rule_recovery"] = recovery
    return result


# ---------------------------------------------------------------------------
# Per-seed full run.
# ---------------------------------------------------------------------------

SEED_BASE = 7000
REPLICATION_BASE = 17000
TRAIN_SIZE = 2500
KNOWN_SIZE = 200
AXIS_SIZES = {"t1": 200, "t2": 200, "t3": 200}
N_SEEDS = 24


def run_phase3_seed(
    i: int,
    *,
    seed_base: int = SEED_BASE,
    train_size: int = TRAIN_SIZE,
    known_size: int = KNOWN_SIZE,
    axis_sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    axis_sizes = axis_sizes or AXIS_SIZES
    env_seed = seed_base + i
    agent_seed = seed_base + 10_000 + i

    gen = AttributeTaskGeneratorV3(
        seed=env_seed,
        train_size=train_size,
        known_size=known_size,
        t1_size=axis_sizes["t1"],
        t2_size=axis_sizes["t2"],
        t3_size=axis_sizes["t3"],
    )
    sets = gen.generate()
    test_sets = {"t1": sets.t1, "t2": sets.t2, "t3": sets.t3}
    test_all = test_sets["t1"] + test_sets["t2"] + test_sets["t3"]

    # --- The single source of experience (uniform random actions) ---
    rng = Random(agent_seed + 55_555)
    experience: list[tuple[ContextV3, str, bool, bool]] = []
    for ctx in sets.train:
        action = ACTIONS_V3[rng.randrange(len(ACTIONS_V3))]
        experience.append((ctx, action, is_success_v3(ctx, action), True))

    # --- Behavioral program ---
    prog = _train_program_v3(sets.train, agent_seed, experience)
    fallback = _majority_action(experience)
    agent_model = _AgentPredictor(prog, fallback)
    bp_result = _eval_model(
        agent_model, test_sets, test_all, sets.known, "full_bp", i
    )
    audit = audit_composition(prog)
    bp_result.n_active = len(prog.rules)

    # --- Ablations: null outcomes (mechanism control) and no-COMPOSE ---
    no_compose = [op for op in _ALL_OPS if op != "COMPOSE"]
    null_prog = _train_program_v3(
        sets.train, agent_seed + 1, experience, randomize_outcomes=True
    )
    nc_prog = _train_program_v3(
        sets.train, agent_seed + 2, experience, allowed_operations=no_compose
    )
    null_result = _eval_model(
        _AgentPredictor(null_prog, fallback),
        test_sets, test_all, sets.known, "null_bp", i,
    )
    null_result.n_active = len(null_prog.rules)
    nc_result = _eval_model(
        _AgentPredictor(nc_prog, fallback),
        test_sets, test_all, sets.known, "ab_no_compose", i,
    )
    nc_result.n_active = len(nc_prog.rules)

    # --- Baselines (exactly the same experience) ---
    n_train = len(experience)
    k_val = max(3, min(21, int(n_train ** 0.5)))
    compo_min_cov = max(8, n_train // 200)
    models = {
        "ValueLogReg": ValueLogisticRegression(seed=42, encoder=ENCODER_V3),
        "ValuekNN": ValueKNN(k=k_val, encoder=ENCODER_V3),
        "ValueTree": ValueDecisionTree(
            max_depth=10, min_samples=max(8, n_train // 40), encoder=ENCODER_V3
        ),
        "ValueDList": ValueDecisionList(
            min_coverage=max(4, n_train // 80), ratio=0.85, encoder=ENCODER_V3
        ),
        # Strong compositional / rule-learning controllers:
        "ValueCompoLift": ValueCompoRuleLearner(
            max_arity=4,
            min_coverage=compo_min_cov,
            min_lift=2.0,
            encoder=ENCODER_V3,
        ),
        "ValueDListML": ValueMultiDecisionList(
            max_arity=4,
            min_coverage=compo_min_cov,
            ratio=0.85,
            encoder=ENCODER_V3,
        ),
        "ValueForest": ValueRandomForest(
            n_estimators=25,
            max_depth=10,
            min_samples=max(8, n_train // 40),
            max_features=6,
            seed=42,
            encoder=ENCODER_V3,
        ),
    }
    baseline_results = {}
    for name, model in models.items():
        model.fit(experience)
        baseline_results[name] = _eval_model(
            model, test_sets, test_all, sets.known, name, i
        )

    # --- Chance & ceiling ---
    class _Chance:
        def __init__(self, rng): self.rng = rng
        def predict(self, ctx): return ACTIONS_V3[self.rng.randrange(len(ACTIONS_V3))]

    chance = balanced_accuracy(
        _Chance(Random(seed_base + 30_000 + i)), test_all, oracle_action_v3
    )
    class _Ceiling:
        def predict(self, ctx): return oracle_action_v3(ctx)
    ceiling = balanced_accuracy(_Ceiling(), test_all, oracle_action_v3)

    return {
        "seed": i,
        "train_size": n_train,
        "results": {
            "full_bp": _serialize(bp_result),
            "null_bp": _serialize(null_result),
            "ab_no_compose": _serialize(nc_result),
            **{n: _serialize(r) for n, r in baseline_results.items()},
            "chance": {"overall": chance, "balanced": chance},
            "ceiling": {"overall": ceiling, "balanced": ceiling},
        },
        "audit": audit,
    }


def _eval_model(
    model,
    test_sets: dict[str, list[ContextV3]],
    test_all: list[ContextV3],
    known: list[ContextV3],
    name: str,
    seed: int,
) -> RunResultV3:
    r = RunResultV3(config=name, seed=seed)
    for axis, ctxs in test_sets.items():
        r.transfer[axis] = evaluate_predictor(
            model, ctxs, success=is_success_v3, oracle=oracle_action_v3
        )["overall"]
    ev = evaluate_predictor(model, test_all, success=is_success_v3, oracle=oracle_action_v3)
    r.overall = ev["overall"]
    r.balanced = ev["balanced"]
    rare = [c for c in test_all if is_rare_context_v3(c)]
    nonrare = [c for c in test_all if not is_rare_context_v3(c)]
    r.rare_accuracy = _accuracy_on(model, rare)
    r.rare_balanced = (
        balanced_accuracy(model, rare, oracle_action_v3) if rare else 0.0
    )
    r.nonrare_balanced = balanced_accuracy(model, nonrare, oracle_action_v3)
    r.known_acc = _accuracy_on(model, known)
    return r


def _accuracy_on(model, ctxs: list[ContextV3]) -> float:
    if not ctxs:
        return 0.0
    return sum(1 for c in ctxs if is_success_v3(c, model.predict(c))) / len(ctxs)


def _serialize(r: RunResultV3) -> dict[str, Any]:
    return {
        "config": r.config,
        "seed": r.seed,
        "transfer": dict(r.transfer),
        "overall": r.overall,
        "balanced": r.balanced,
        "rare_balanced": r.rare_balanced,
        "nonrare_balanced": r.nonrare_balanced,
        "rare_accuracy": r.rare_accuracy,
        "known_acc": r.known_acc,
        "n_active": r.n_active,
    }


# ---------------------------------------------------------------------------
# Aggregate runs.
# ---------------------------------------------------------------------------

def run_phase3_study(
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
        runs.append(run_phase3_seed(i, seed_base=seed_base))
        if i % 4 == 0:
            print(f"  [phase3-{tag}] {i}/{n_seeds}  {time.time() - t0:.0f}s", flush=True)
    path = out / f"phase3_{tag}.json"
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False))
    print(f"  saved {path.name} ({time.time() - t0:.0f}s total)")
    return path


def summarize_study(path: Path | str) -> None:
    runs = json.loads(Path(path).read_text(encoding="utf-8"))
    models = [
        "full_bp", "null_bp", "ab_no_compose",
        "ValueLogReg", "ValuekNN", "ValueTree", "ValueDList",
        "ValueCompoLift", "ValueDListML", "ValueForest",
        "chance", "ceiling",
    ]
    acc = {m: defaultdict(list) for m in models}
    aud = {"rare": Counter(), "simple": Counter()}
    for run in runs:
        for m in models:
            res = run["results"][m]
            acc[m]["balanced"].append(res["balanced"])
            acc[m]["overall"].append(res["overall"])
            if m == "full_bp":
                acc[m]["rare_balanced"].append(res["rare_balanced"])
                acc[m]["rareraw"].append(res["rare_accuracy"])
                acc[m]["known_acc"].append(res["known_acc"])
                for name, ok in run["audit"]["rare_discovered"].items():
                    if ok:
                        aud["rare"][name] += 1
                for name, ok in run["audit"]["simple_discovered"].items():
                    if ok:
                        aud["simple"][name] += 1
    print(f"\n=== Phase 3 summary ({len(runs)} seeds) ===")
    print(f"{'model':<14}{'balanced':>10}{'overall':>10}{'rare':>10}{'known':>10}{'rules':>8}")
    for m in models:
        row = acc[m]
        b = sum(row["balanced"]) / len(row["balanced"])
        o = sum(row["overall"]) / len(row["overall"])
        rb = sum(row.get("rare_balanced", [0] * len(runs))) / len(runs) if m == "full_bp" else float("nan")
        k = sum(row.get("known_acc", [0] * len(runs))) / len(runs) if m == "full_bp" else float("nan")
        na = sum(r["results"][m].get("n_active", 0) for r in runs) / len(runs)
        frb = f"{rb:>10.3f}" if m == "full_bp" else "-".rjust(10)
        fk = f"{k:>10.3f}" if m == "full_bp" else "-".rjust(10)
        print(f"{m:<14}{b:>10.3f}{o:>10.3f}{frb}{fk}{na:>8.1f}")
    print("\ncomposition audit (seeds where the rule was recovered / total):")
    for name, cnt in aud["rare"].items():
        print(f"  rare   {name}: {cnt}/{len(runs)}")
    for name, cnt in aud["simple"].items():
        print(f"  simple {name}: {cnt}/{len(runs)}")
    print("\nrule-recovery metrics (mean over seeds, full_bp):")
    recov = [r["audit"]["rule_recovery"] for r in runs]
    n_targets = recov[0]["n_targets"]
    print(f"  targets: {len(recov)} seeds x {n_targets} oracle compositional rules")
    print(
        f"  precise    : {sum(x['precision'] for x in recov) / len(recov):.3f}"
        f"   (FDR {sum(x['fdr'] for x in recov) / len(recov):.3f};"
        f" winner-acc {sum(x['winner_accuracy'] for x in recov) / len(recov):.3f})"
    )
    print(
        f"  exact rate : {sum(x['exact_rate'] for x in recov) / len(recov):.3f}"
        f"   subset rate {sum(x['subset_rate'] for x in recov) / len(recov):.3f}"
    )
    print(
        f"  consistent / inconsistent : {sum(x['n_consistent'] for x in recov) / len(recov):.1f}"
        f" / {sum(x['n_inconsistent'] for x in recov) / len(recov):.1f}"
        f"   (compositional active {sum(x['n_compositional_active'] for x in recov) / len(recov):.1f})"
    )
    for t in range(n_targets):
        ex = sum(1 for x in recov if x["exact_recovered"][t]) / len(recov)
        sub = sum(1 for x in recov if x["subset_recovered"][t]) / len(recov)
        print(
            f"  target {recov[0]['targets_exact'][t]:<40} exact {ex:5.2f}   subset {sub:5.2f}"
        )


def paired_stats(path: Path | str, n_perm: int = 20_000) -> None:
    """Paired permutation p-values + Cohen's d_z (full_bp vs each baseline)."""
    import numpy as np

    runs = json.loads(Path(path).read_text(encoding="utf-8"))
    bp = np.array([r["results"]["full_bp"]["balanced"] for r in runs])
    names = [
        "ValueLogReg", "ValuekNN", "ValueTree", "ValueDList",
        "ValueCompoLift", "ValueDListML", "ValueForest",
        "null_bp", "ab_no_compose",
    ]

    def _perm_p(a, b, rng) -> float:
        d = np.abs((a - b).mean())
        obs = np.concatenate([a, b])
        cnt = 0
        for _ in range(n_perm):
            p = rng.permutation(obs)
            s = p[: a.size].mean() - p[a.size:].mean()
            if abs(s) >= d:
                cnt += 1
        return (cnt + 1) / (n_perm + 1)

    rng = np.random.default_rng(0)
    print(f"\n== paired stats ({len(runs)} seeds, {path}) ==")
    for name in names:
        v = np.array([r["results"][name]["balanced"] for r in runs])
        diff = (bp - v).mean()
        dz = diff / (bp - v).std(ddof=1) if (bp - v).std(ddof=1) > 0 else float("inf")
        p = _perm_p(bp, v, rng)
        lo, hi = _bootstrap_ci(bp - v, rng)
        print(
            f"full_bp vs {name:<12} diff {diff:+.3f} 95%CI [{lo:+.3f},{hi:+.3f}]"
            f"  dz {dz:+5.2f}  p {p:.4f}"
        )


def _bootstrap_ci(diffs: np.ndarray, rng, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean paired difference."""
    import numpy as np
    if diffs.size < 2:
        return float(diffs[0]), float(diffs[0])
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, diffs.size, size=diffs.size)
        means[b] = diffs[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


if __name__ == "__main__":
    print("=== Phase 3 study ===")
    run_phase3_study(seed_base=SEED_BASE, n_seeds=N_SEEDS)
    run_phase3_study(seed_base=REPLICATION_BASE, n_seeds=N_SEEDS)
    summarize_study(f"results/phase3_main.json")
    paired_stats(f"results/phase3_main.json")
    paired_stats(f"results/phase3_replication.json")
    print("Done.")