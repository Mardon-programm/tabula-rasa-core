from __future__ import annotations

import json
import sys
import os
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.sandbox_sim import SandboxEnvironment
from main import TabulaRasaCore
from experiments.baselines import BasePredictor, BASELINES

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def _fresh_env() -> SandboxEnvironment:
    """Create a fresh deterministic sandbox starting from state A."""
    env = SandboxEnvironment(seed=42)
    env.current_state = "A"
    env.reset()
    return env


def run_tr_core(
    seed: int,
    total_steps: int,
    drift_step: int,
    fixed_action: str = "X",
    **ablation_flags: bool,
) -> list[dict[str, Any]]:
    core = TabulaRasaCore(
        seed=seed,
        enable_metrics=False,
        verbose=False,
        **ablation_flags,
    )
    env = _fresh_env()

    # Align core's notion of current state with the environment.
    obs = env.observe()
    core.perceive(obs.state)

    trace: list[dict[str, Any]] = []
    for _ in range(total_steps):
        if core.step_count == drift_step:
            env.change_fundamental_rule()

        # Force a fixed action so every model observes the SAME transitions.
        # This isolates prediction + adaptation, ignoring differences in action
        # policy. TR-Core's internal action selection is overridden here.
        core.previous_action = fixed_action
        before_state = env.current_state
        env_result = env.step(fixed_action)

        result = core.step(
            env_observation=before_state,
            actual_next_state=env_result.state,
            reward=env_result.reward,
        )

        error = 1.0 if result["predicted_state"] != env_result.state else 0.0
        trace.append(
            {
                "step": core.step_count,
                "state": before_state,
                "actual_state": env_result.state,
                "predicted_state": result["predicted_state"],
                "action": fixed_action,
                "error": error,
            }
        )

    return trace


def run_baseline(
    cls: type[BasePredictor],
    seed: int,
    total_steps: int,
    drift_step: int,
) -> list[dict[str, Any]]:
    if cls.name == "Static":
        model = cls(seed=seed, freeze_after=drift_step)
    else:
        model = cls(seed=seed)
    env = _fresh_env()
    model.reset()
    model.current_state = "A"

    trace: list[dict[str, Any]] = []
    for _ in range(total_steps):
        if model.step_count == drift_step:
            env.change_fundamental_rule()
        step = model.run_step(env)
        trace.append(step)

    return trace


def summarize(trace: list[dict[str, Any]], drift_step: int) -> dict[str, float]:
    errors = [t["error"] for t in trace]
    pre = errors[:drift_step]
    post = errors[drift_step:]

    mean_error = sum(errors) / len(errors) if errors else 1.0
    mean_pre = sum(pre) / len(pre) if pre else 0.0
    mean_post = sum(post) / len(post) if post else 0.0
    cum_error = sum(errors)

    # Reach baseline accuracy: fraction of steps correct *overall*
    correct = sum(1 for e in errors if e == 0.0)
    accuracy = correct / len(errors) if errors else 0.0

    # Recovery: first post-drift step where prediction is correct.
    # Require correctness to hold for a confirmation window to avoid a spurious
    # lucky single correct prediction counting as recovery.
    recovery_step: Optional[int] = None
    win = 3
    for i in range(len(post) - win + 1):
        window = post[i : i + win]
        if all(e == 0.0 for e in window):
            recovery_step = i
            break
    if recovery_step is None:
        recovery_step = len(post)

    return {
        "mean_error": round(mean_error, 4),
        "mean_error_pre_drift": round(mean_pre, 4),
        "mean_error_post_drift": round(mean_post, 4),
        "cumulative_error": round(cum_error, 4),
        "accuracy": round(accuracy, 4),
        "recovery_steps": recovery_step,
    }


def run_benchmark(
    seeds: int = 20,
    total_steps: int = 100,
    drift_step: int = 50,
) -> dict[str, Any]:
    models: list[tuple[str, Any]] = [("Random", None), ("Static", None), ("Adaptive", None)]

    core_variants: list[tuple[str, dict[str, bool]]] = [
        ("TR-Core (Full)", {}),
        ("TR-Core - Memory", {"enable_memory": False}),
        ("TR-Core - Curiosity", {"enable_curiosity": False}),
        ("TR-Core - Self-model", {"enable_self_model": False}),
        ("TR-Core - Adaptation", {"enable_adaptation": False}),
    ]

    results: dict[str, dict[str, Any]] = {}
    per_trace: dict[str, list[dict[str, Any]]] = {}

    for name, flags in models:
        cls = BASELINES[name]
        agg: list[dict[str, float]] = []
        for seed in range(seeds):
            trace = run_baseline(cls, seed, total_steps, drift_step)
            per_trace.setdefault(name, []).append(trace)
            agg.append(summarize(trace, drift_step))
        results[name] = _aggregate(name, agg)

    for name, flags in core_variants:
        agg: list[dict[str, float]] = []
        for seed in range(seeds):
            trace = run_tr_core(seed, total_steps, drift_step, **flags)
            per_trace.setdefault(name, []).append(trace)
            agg.append(summarize(trace, drift_step))
        results[name] = _aggregate(name, agg)

    # Keep per-trace error curves for analysis (compact).
    error_curves = {
        name: [[t["error"] for t in trace] for trace in traces]
        for name, traces in per_trace.items()
    }

    return {
        "experiment": {
            "seeds": seeds,
            "total_steps": total_steps,
            "drift_step": drift_step,
            "environment": "SandboxEnvironment with rule switch at drift_step",
            "research_question": (
                "Can a non-pretrained cognitive architecture detect changes in "
                "environmental dynamics and adapt its internal transition model "
                "using prediction error and experience-driven learning?"
            ),
            "hypothesis": (
                "TR-Core with adaptive learning will recover from concept drift "
                "faster and maintain lower prediction error than non-adaptive baselines."
            ),
        },
        "results": results,
        "error_curves": error_curves,
    }


def _aggregate(name: str, runs: list[dict[str, float]]) -> dict[str, Any]:
    keys = [
        "mean_error",
        "mean_error_pre_drift",
        "mean_error_post_drift",
        "cumulative_error",
        "accuracy",
        "recovery_steps",
    ]
    out: dict[str, Any] = {"runs": len(runs)}
    for key in keys:
        values = [r[key] for r in runs]
        out[key] = round(sum(values) / len(values), 4) if values else 0.0
        if len(values) > 1:
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            out[f"{key}_std"] = round(var**0.5, 4)
        else:
            out[f"{key}_std"] = 0.0
    return out


def print_table(report: dict[str, Any]) -> None:
    exp = report["experiment"]
    print("=" * 78)
    print("TR-CORE BENCHMARK — Concept Drift")
    print(f"  {exp['seeds']} seeds x {exp['total_steps']} steps, drift at step {exp['drift_step']}")
    print("=" * 78)
    print(
        f"{'Model':<24}{'MeanErr':>10}{'PostDrift':>10}{'Accuracy':>10}{'CumErr':>10}{'Recovery':>10}"
    )
    print("-" * 78)
    for name, res in report["results"].items():
        print(
            f"{name:<24}{res['mean_error']:>10.3f}{res['mean_error_post_drift']:>10.3f}"
            f"{res['accuracy']:>10.3f}{res['cumulative_error']:>10.1f}{res['recovery_steps']:>10.1f}"
        )
    print("=" * 78)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="TR-Core concept drift benchmark")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--drift", type=int, default=60)
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "benchmark_report.json"))
    parser.add_argument("--no-save", action="store_true", help="Do not write report file")
    args = parser.parse_args(argv)

    t0 = time.time()
    report = run_benchmark(seeds=args.seeds, total_steps=args.steps, drift_step=args.drift)
    print_table(report)

    if not args.no_save:
        out = Path(args.out)
        out.parent.mkdir(exist_ok=True, parents=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[REPORT] Saved to {out}")

    print(f"\n[elapsed] {time.time() - t0:.1f}s")
    return report


if __name__ == "__main__":
    main()
