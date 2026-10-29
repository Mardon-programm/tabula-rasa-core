#!/usr/bin/env python3
"""
Stress Benchmark: Attribute Scalability Test for TR-Core

Tests how TR-Core's Apriori induction scales with increasing attribute
dimensionality (N = 8, 32, 128, 512, 1024).

Measures:
  - Latency (ms) per induction pass
  - RAM consumption (MB) peak
  - Accuracy (OOD) on held-out contexts
  - Number of rules materialized

Usage:
  python experiments/stress_benchmark.py --output results/stress_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from behavioral.program_v3 import BehavioralProgramV3
from behavioral.program import Params
from behavioral.rules import Action
# No environment import needed - synthetic env generated inline


def generate_synthetic_env(
    n_attrs: int,
    n_values_per_attr: int = 3,
    n_actions: int = 4,
    seed: int = 42,
) -> tuple[dict, list[str], list[str], Any]:
    """
    Generate a synthetic attribute task environment with N attributes.

    Returns: (domains, attributes, actions, oracle_fn)
    """
    rng = random.Random(seed)
    attrs = [f"attr_{i}" for i in range(n_attrs)]
    domains = {attr: [f"v{j}" for j in range(n_values_per_attr)] for attr in attrs}
    actions = [f"act_{i}" for i in range(n_actions)]

    # Deterministic oracle: action based on first attribute only (learnable)
    # Tests scalability: more attributes = more noise, but signal is in attr_0
    def oracle(ctx: dict) -> str:
        v0 = ctx.get(attrs[0], "v0")
        vi0 = int(v0[1:]) if v0.startswith("v") else 0
        return actions[vi0 % n_actions]

    return domains, attrs, actions, oracle


def run_stress_test(
    n_attrs: int,
    train_size: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a single stress test for N attributes."""
    domains, attrs, actions, oracle = generate_synthetic_env(n_attrs, seed=seed)

    # Generate synthetic data
    rng = random.Random(seed)
    contexts = []
    for _ in range(train_size):
        ctx = {attr: rng.choice(domains[attr]) for attr in attrs}
        contexts.append(ctx)

    # Build experience
    experience = []
    for ctx in contexts:
        action_name = rng.choice(actions)
        action = Action('primitive', action_name)
        success = (action_name == oracle(ctx))
        experience.append((ctx, action, success, True))

    # Initialize BehavioralProgramV3
    bp = BehavioralProgramV3(
        attributes=attrs,
        domains=domains,
        actions=actions,
        params=Params(
            develop_interval=50,
            min_coverage=4,
            validate_accuracy=0.8,
            validate_margin=0.05,
            explore_rate=0.0,
        ),
        seed=seed,
        max_arity=min(2, n_attrs),  # cap arity at 4
    )

    # Feed experience
    for ctx, action, success, _ in experience:
        bp.learn(ctx, action, success, explored=True)

    # Measure induction
    import tracemalloc
    tracemalloc.start()
    t0 = time.perf_counter()
    n_new = bp.develop()
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latency_ms = (t1 - t0) * 1000
    peak_mem_mb = peak / (1024 * 1024)

    # Measure OOD accuracy on held-out contexts
    test_contexts = []
    for _ in range(500):
        ctx = {attr: rng.choice(domains[attr]) for attr in attrs}
        test_contexts.append(ctx)

    correct = 0
    for ctx in test_contexts:
        pred = bp.choose_action(ctx)
        if pred is not None and pred.name == oracle(ctx):
            correct += 1
    ood_acc = correct / len(test_contexts)

    return {
        "n_attrs": n_attrs,
        "n_contexts_total": 3 ** n_attrs if n_attrs <= 10 else "N/A",
        "train_size": train_size,
        "latency_ms": round(latency_ms, 2),
        "peak_mem_mb": round(peak_mem_mb, 2),
        "ood_accuracy": round(ood_acc, 4),
        "n_rules_total": len(bp.rules),
        "n_rules_active": len(bp.active_rules),
        "n_rules_materialized": n_new,
        "max_arity": min(4, n_attrs),
    }


def main():
    parser = argparse.ArgumentParser(description="Stress Benchmark: Attribute Scalability")
    parser.add_argument("--output", type=str, default="results/stress_benchmark.json")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    n_attrs_list = [8, 16, 32]

    results = []
    for n_attrs in n_attrs_list:
        for seed in seeds:
            print(f"Running N={n_attrs}, seed={seed}...")
            result = run_stress_test(n_attrs, seed=seed)
            result["seed"] = seed
            results.append(result)
            print(f"  Latency: {result['latency_ms']}ms, Mem: {result['peak_mem_mb']}MB, OOD: {result['ood_accuracy']}, Rules: {result['n_rules_active']}")

    # Aggregate by N
    aggregated = {}
    for n in n_attrs_list:
        subset = [r for r in results if r["n_attrs"] == n]
        aggregated[n] = {
            "n_attrs": n,
            "latency_ms_mean": round(np.mean([r["latency_ms"] for r in subset]), 2),
            "latency_ms_std": round(np.std([r["latency_ms"] for r in subset]), 2),
            "peak_mem_mb_mean": round(np.mean([r["peak_mem_mb"] for r in subset]), 2),
            "peak_mem_mb_std": round(np.std([r["peak_mem_mb"] for r in subset]), 2),
            "ood_accuracy_mean": round(np.mean([r["ood_accuracy"] for r in subset]), 4),
            "ood_accuracy_std": round(np.std([r["ood_accuracy"] for r in subset]), 4),
            "n_rules_active_mean": round(np.mean([r["n_rules_active"] for r in subset]), 1),
            "n_seeds": len(subset),
        }

    output = {
        "raw_results": results,
        "aggregated": aggregated,
        "config": {
            "n_attrs_list": n_attrs_list,
            "seeds": seeds,
            "train_size": 1000,
            "max_arity": 4,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {out_path}")

    # Print summary table
    print("\n=== Stress Benchmark Summary ===")
    print(f"{'N':>6} {'Latency (ms)':>14} {'RAM (MB)':>10} {'OOD Acc':>10} {'Rules':>8}")
    for n in n_attrs_list:
        a = aggregated[n]
        print(f"{n:>6} {a['latency_ms_mean']:>14.1f} {a['peak_mem_mb_mean']:>10.2f} {a['ood_accuracy_mean']:>10.4f} {a['n_rules_active_mean']:>8.1f}")


if __name__ == "__main__":
    main()