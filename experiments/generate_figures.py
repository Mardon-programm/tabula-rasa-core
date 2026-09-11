#!/usr/bin/env python3
"""
Figure Generation Script for TR-Core Paper.

Generates publication-ready figures from experiment analysis results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',
})

sns.set_style("whitegrid")
sns.set_context("paper")


def load_analysis(path: Path) -> Dict[str, Any]:
    """Load analysis JSON."""
    return json.loads(path.read_text())


def plot_balanced_accuracy_comparison(
    main_analysis: Dict[str, Any],
    repl_analysis: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot balanced accuracy comparison across models (Phase 3)."""
    
    main = main_analysis.get("main", {})
    repl = repl_analysis.get("main", {})
    
    # Model order for consistent plotting
    model_order = [
        "full_bp",
        "ValueDListML",
        "ValueTree",
        "ValueKNN",
        "ValueRandomForest",
        "ValueCompoLift",
        "ValueLogisticRegression",
        "ValueDecisionList",
        "null_bp",
        "ab_no_compose",
        "chance",
        "ceiling",
    ]
    
    # Filter to available models
    models = [m for m in model_order if m in main]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Phase 3 uses "balanced" key
    main_means = [main[m]["metrics"]["balanced"]["mean"] for m in models]
    main_stds = [main[m]["metrics"]["balanced"]["std"] for m in models]
    
    repl_means = [repl[m]["metrics"]["balanced"]["mean"] for m in models]
    repl_stds = [repl[m]["metrics"]["balanced"]["std"] for m in models]
    
    bars1 = ax.bar(x - width/2, main_means, width, label='Main (N=24)', 
                   yerr=main_stds, capsize=3, color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, repl_means, width, label='Replication (N=24)', 
                   yerr=repl_stds, capsize=3, color='#A23B72', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Phase 3: Balanced Accuracy Comparison (Main vs Replication)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('Value', '').replace('Decision', 'D').replace('Multi', 'M').replace('Compo', 'C').replace('LogisticRegression', 'LogReg') for m in models], 
                       rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Chance')
    
    # Add significance markers
    for i, m in enumerate(models):
        if m == "full_bp":
            ax.text(i, main_means[i] + main_stds[i] + 0.02, '★', ha='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / "phase3_balanced_accuracy.pdf")
    plt.savefig(output_dir / "phase3_balanced_accuracy.png")
    plt.close()


def plot_phase4_heldout_accuracy(
    main_analysis: Dict[str, Any],
    repl_analysis: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot Phase 4 held-out accuracy (OOD generalization)."""
    
    main = main_analysis.get("main", {})
    repl = repl_analysis.get("main", {})
    
    model_order = [
        "full_bp",
        "ValueMultiDecisionList",
        "ValueDecisionTree",
        "ValueKNN",
        "ValueRandomForest",
        "ab_no_compose",
        "null_bp",
        "chance",
        "ceiling",
    ]
    
    models = [m for m in model_order if m in main]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    main_means = [main[m]["metrics"]["heldout_acc"]["mean"] for m in models]
    main_stds = [main[m]["metrics"]["heldout_acc"]["std"] for m in models]
    
    repl_means = [repl[m]["metrics"]["heldout_acc"]["mean"] for m in models]
    repl_stds = [repl[m]["metrics"]["heldout_acc"]["std"] for m in models]
    
    bars1 = ax.bar(x - width/2, main_means, width, label='Main (N=24)', 
                   yerr=main_stds, capsize=3, color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, repl_means, width, label='Replication (N=24)', 
                   yerr=repl_stds, capsize=3, color='#A23B72', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Held-out Accuracy (OOD)')
    ax.set_title('Phase 4: Structural-OOD Generalization (Held-out Triple)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('Value', '').replace('Decision', 'D').replace('Multi', 'M') for m in models], 
                       rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (1/4)')
    
    # Highlight full_bp
    for i, m in enumerate(models):
        if m == "full_bp":
            ax.text(i, main_means[i] + main_stds[i] + 0.02, '★', ha='center', fontsize=14, color='#2E86AB')
            ax.text(i, repl_means[i] + repl_stds[i] + 0.02, '★', ha='center', fontsize=14, color='#A23B72')
    
    plt.tight_layout()
    plt.savefig(output_dir / "phase4_heldout_accuracy.pdf")
    plt.savefig(output_dir / "phase4_heldout_accuracy.png")
    plt.close()


def plot_paired_comparisons(
    analysis: Dict[str, Any],
    phase: str,
    output_dir: Path,
) -> None:
    """Plot paired comparison effect sizes."""
    
    # Try different key formats
    comps = analysis.get(f"comparisons_{phase}", {})
    if not comps:
        comps = analysis.get("comparisons_main", {})
    if not comps:
        return
    
    names = []
    diffs = []
    dzs = []
    pvals = []
    ci_lowers = []
    ci_uppers = []
    
    for name, comp in comps.items():
        names.append(name.replace(f"full_bp_vs_", "").replace("Value", "").replace("Decision", "D"))
        diffs.append(comp["mean_diff"])
        dzs.append(comp["cohens_dz"])
        pvals.append(comp["p_value"])
        ci_lowers.append(comp["ci_95_lower"])
        ci_uppers.append(comp["ci_95_upper"])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mean difference with CI
    colors = ['green' if d > 0 else 'red' for d in diffs]
    ax1.barh(range(len(names)), diffs, xerr=[np.array(diffs) - np.array(ci_lowers), np.array(ci_uppers) - np.array(diffs)], 
             color=colors, alpha=0.7, capsize=3, edgecolor='black')
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    ax1.set_xlabel('Mean Difference (full_bp - target)')
    ax1.set_title(f'Phase {phase.upper()}: Paired Mean Differences (95% CI)')
    
    # Cohen's dz
    colors_dz = ['green' if d > 0 else 'red' for d in dzs]
    ax2.barh(range(len(names)), dzs, color=colors_dz, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names)
    ax2.set_xlabel("Cohen's dz")
    ax2.set_title(f'Phase {phase.upper()}: Effect Sizes')
    
    # Add p-value annotations
    for i, (d, p) in enumerate(zip(diffs, pvals)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax1.text(d + (0.01 if d >= 0 else -0.01), i, f' {sig}', va='center', 
                 ha='left' if d >= 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"phase{phase}_paired_comparisons.pdf")
    plt.savefig(output_dir / f"phase{phase}_paired_comparisons.png")
    plt.close()


def plot_rule_recovery(
    analysis: Dict[str, Any],
    phase: str,
    output_dir: Path,
) -> None:
    """Plot rule recovery rates."""
    
    audit = analysis.get(f"audit_{phase}", {})
    if not audit:
        audit = analysis.get("audit_main", {})
    repl_audit = analysis.get("audit_replication", {})
    
    if not audit:
        return
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    categories = ['Main', 'Replication']
    recovered = [audit.get('pair_rule_recovered', 0), repl_audit.get('pair_rule_recovered', 0)]
    total = [audit.get('total_seeds', 24), repl_audit.get('total_seeds', 24)]
    rates = [r/t if t > 0 else 0 for r, t in zip(recovered, total)]
    
    bars = ax.bar(categories, rates, color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black')
    
    for bar, r, t in zip(bars, recovered, total):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{r}/{t}', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Recovery Rate')
    ax.set_title(f'Phase {phase.upper()}: Pair Rule Recovery (calm∧guard→approach)')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"phase{phase}_rule_recovery.pdf")
    plt.savefig(output_dir / f"phase{phase}_rule_recovery.png")
    plt.close()


def plot_ablation_effect(
    analysis: Dict[str, Any],
    phase: str,
    output_dir: Path,
) -> None:
    """Plot ablation study results."""
    
    main = analysis.get("main", {})
    
    # Key ablation comparisons
    ablation_models = ["full_bp", "ab_no_compose", "null_bp", "chance"]
    # Filter to models that exist in main
    ablation_models = [m for m in ablation_models if m in main]
    
    # Use appropriate metric key for each phase
    if phase == "3":
        metric_key = "balanced"
    elif phase == "4":
        metric_key = "row_balanced"
    else:
        metric_key = "row_balanced"
    
    # Filter to models that have this metric
    available = [m for m in ablation_models if metric_key in main[m]["metrics"]]
    
    if len(available) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    means = [main[m]["metrics"][metric_key]["mean"] for m in available]
    stds = [main[m]["metrics"][metric_key]["std"] for m in available]
    
    labels = [m.replace('_', ' ').title() for m in available]
    colors = ['#2E86AB' if m == 'full_bp' else '#E74C3C' if 'ab_no' in m or 'null' in m else 'gray' for m in available]
    
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.01, 
                f'{mean:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    ylabel = 'Balanced Accuracy' if metric_key in ('row_balanced', 'balanced') else 'Held-out Accuracy'
    ax.set_ylabel(ylabel)
    ax.set_title(f'Phase {phase.upper()}: Ablation Study')
    ax.set_ylim(0, max(means) + max(stds) + 0.15)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"phase{phase}_ablation.pdf")
    plt.savefig(output_dir / f"phase{phase}_ablation.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--phase3", type=str, required=True, help="Phase 3 main analysis JSON")
    parser.add_argument("--phase3-repl", type=str, help="Phase 3 replication analysis JSON")
    parser.add_argument("--phase4", type=str, required=True, help="Phase 4 main analysis JSON")
    parser.add_argument("--phase4-repl", type=str, help="Phase 4 replication analysis JSON")
    parser.add_argument("--output", type=str, default="figures", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    phase3 = load_analysis(Path(args.phase3))
    phase3_repl = load_analysis(Path(args.phase3_repl)) if args.phase3_repl else phase3
    phase4 = load_analysis(Path(args.phase4))
    phase4_repl = load_analysis(Path(args.phase4_repl)) if args.phase4_repl else phase4
    
    print("Generating figures...")
    
    # Phase 3 figures
    plot_balanced_accuracy_comparison(phase3, phase3_repl, output_dir)
    plot_paired_comparisons(phase3, "3", output_dir)
    plot_rule_recovery(phase3, "3", output_dir)
    plot_ablation_effect(phase3, "3", output_dir)
    
    # Phase 4 figures
    plot_phase4_heldout_accuracy(phase4, phase4_repl, output_dir)
    plot_paired_comparisons(phase4, "4", output_dir)
    plot_rule_recovery(phase4, "4", output_dir)
    plot_ablation_effect(phase4, "4", output_dir)
    
    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()