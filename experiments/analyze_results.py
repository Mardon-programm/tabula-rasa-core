#!/usr/bin/env python3
"""
Results Analysis Script for TR-Core Experiments.

Loads experiment results and computes statistical summaries,
paired comparisons, and effect sizes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_results(path: Path) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    return json.loads(path.read_text())


def summarize_study(runs: List[Dict[str, Any]], model_names: List[str]) -> Dict[str, Any]:
    """Compute summary statistics across seeds."""
    summary = {}
    
    for name in model_names:
        metrics = defaultdict(list)
        for run in runs:
            if name in run.get("results", {}):
                res = run["results"][name]
                for k, v in res.items():
                    if isinstance(v, (int, float)):
                        metrics[k].append(v)
        
        if metrics:
            summary[name] = {
                "n_seeds": len(runs),
                "metrics": {
                    k: {
                        "mean": float(np.mean(v)),
                        "std": float(np.std(v, ddof=1)),
                        "min": float(np.min(v)),
                        "max": float(np.max(v)),
                        "median": float(np.median(v)),
                    }
                    for k, v in metrics.items()
                }
            }
    
    return summary


def paired_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 20000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired permutation test for two sets of scores."""
    rng = np.random.default_rng(seed)
    
    diff = a - b
    observed_mean_diff = np.mean(diff)
    observed_abs_diff = np.abs(observed_mean_diff)
    
    # Cohen's dz
    dz = observed_mean_diff / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else float('inf')
    
    # Permutation test
    combined = np.concatenate([a, b])
    n = len(a)
    count = 0
    
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        perm_diff = np.mean(perm[:n]) - np.mean(perm[n:])
        if np.abs(perm_diff) >= observed_abs_diff:
            count += 1
    
    p_value = (count + 1) / (n_perm + 1)
    
    # Bootstrap CI for mean difference
    n_boot = 2000
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_diffs.append(np.mean(a[idx] - b[idx]))
    
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    
    return {
        "mean_diff": float(observed_mean_diff),
        "cohens_dz": float(dz),
        "p_value": float(p_value),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
    }


def compare_models(runs: List[Dict[str, Any]], baseline: str, targets: List[str]) -> Dict[str, Any]:
    """Run paired comparisons between baseline and target models."""
    comparisons = {}
    
    for target in targets:
        a_vals = []
        b_vals = []
        
        for run in runs:
            if baseline in run.get("results", {}) and target in run.get("results", {}):
                # Use balanced accuracy or heldout_acc depending on experiment
                a = (run["results"][baseline].get("row_balanced") 
                     or run["results"][baseline].get("heldout_acc")
                     or run["results"][baseline].get("balanced"))
                b = (run["results"][target].get("row_balanced") 
                     or run["results"][target].get("heldout_acc")
                     or run["results"][target].get("balanced"))
                if a is not None and b is not None:
                    a_vals.append(a)
                    b_vals.append(b)
        
        if a_vals and b_vals:
            a_arr = np.array(a_vals)
            b_arr = np.array(b_vals)
            comparisons[f"{baseline}_vs_{target}"] = paired_permutation_test(a_arr, b_arr)
    
    return comparisons


def audit_pair_rule_recovery(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze pair rule recovery from audit data."""
    recovered_count = sum(1 for r in runs if r.get("audit", {}).get("found"))
    total = len(runs)
    
    coverages = [r.get("audit", {}).get("heldout_region_correct_full", 0) for r in runs]
    
    return {
        "pair_rule_recovered": recovered_count,
        "total_seeds": total,
        "recovery_rate": recovered_count / total if total > 0 else 0,
        "heldout_region_coverage_mean": float(np.mean(coverages)) if coverages else 0,
        "heldout_region_coverage_std": float(np.std(coverages, ddof=1)) if len(coverages) > 1 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze TR-Core experiment results")
    parser.add_argument("--input", type=str, required=True, help="Input results JSON file")
    parser.add_argument("--replication", type=str, help="Replication results JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output analysis JSON file")
    parser.add_argument("--baseline", type=str, default="full_bp", help="Baseline model name")
    parser.add_argument("--targets", type=str, nargs="+", help="Target models to compare")
    args = parser.parse_args()
    
    # Load main results
    runs = load_results(Path(args.input))
    
    # Load replication if provided
    repl_runs = []
    if args.replication:
        repl_runs = load_results(Path(args.replication))
    
    # Determine model names from first run
    if runs:
        model_names = list(runs[0].get("results", {}).keys())
    else:
        model_names = []
    
    # Default targets for comparison
    if args.targets is None:
        args.targets = [m for m in model_names if m != args.baseline and not m.startswith("chance") and m != "ceiling"]
    
    print(f"Analyzing {len(runs)} seeds, models: {model_names}")
    print(f"Baseline: {args.baseline}, Targets: {args.targets}")
    
    # Compute summaries
    main_summary = summarize_study(runs, model_names)
    repl_summary = summarize_study(repl_runs, model_names) if repl_runs else {}
    
    # Paired comparisons
    comparisons = compare_models(runs, args.baseline, args.targets)
    repl_comparisons = compare_models(repl_runs, args.baseline, args.targets) if repl_runs else {}
    
    # Audit analysis
    audit = audit_pair_rule_recovery(runs)
    repl_audit = audit_pair_rule_recovery(repl_runs) if repl_runs else {}
    
    # Combine all
    analysis = {
        "main": main_summary,
        "replication": repl_summary,
        "comparisons_main": comparisons,
        "comparisons_replication": repl_comparisons,
        "audit_main": audit,
        "audit_replication": repl_audit,
    }
    
    # Save
    output_path = Path(args.output)
    output_path.write_text(json.dumps(analysis, indent=2))
    print(f"Saved analysis to {output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    for name, data in main_summary.items():
        metrics = data.get("metrics", {})
        bal_acc = metrics.get("row_balanced", metrics.get("heldout_acc", {}))
        if bal_acc:
            print(f"  {name}: balanced_acc = {bal_acc['mean']:.3f} ± {bal_acc['std']:.3f}")
    
    print("\n=== Paired Comparisons (Main) ===")
    for name, comp in comparisons.items():
        print(f"  {name}: diff={comp['mean_diff']:+.3f}, dz={comp['cohens_dz']:+.2f}, p={comp['p_value']:.4f}, CI=[{comp['ci_95_lower']:.3f}, {comp['ci_95_upper']:.3f}]")
    
    if repl_comparisons:
        print("\n=== Paired Comparisons (Replication) ===")
        for name, comp in repl_comparisons.items():
            print(f"  {name}: diff={comp['mean_diff']:+.3f}, dz={comp['cohens_dz']:+.2f}, p={comp['p_value']:.4f}, CI=[{comp['ci_95_lower']:.3f}, {comp['ci_95_upper']:.3f}]")
    
    print(f"\n=== Pair Rule Recovery ===")
    print(f"  Main: {audit['pair_rule_recovered']}/{audit['total_seeds']} ({audit['recovery_rate']:.2f})")
    print(f"  Heldout region coverage: {audit['heldout_region_coverage_mean']:.3f} ± {audit['heldout_region_coverage_std']:.3f}")
    if repl_audit:
        print(f"  Replication: {repl_audit['pair_rule_recovered']}/{repl_audit['total_seeds']} ({repl_audit['recovery_rate']:.2f})")


if __name__ == "__main__":
    main()