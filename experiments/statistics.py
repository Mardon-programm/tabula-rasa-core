"""
Statistical analysis for the TR-Core experiment (protocol v0.1 §6).

Without scipy, we use:
  - paired permutation test  (exact H0 distribution, seeded Monte-Carlo)
  - Cohen's d_z              (paired-effect size)
  - bias-corrected percentile bootstrap CI  (protocol: >=2000 resamples)
  - Holm-Bonferroni          (family-wise correction across comparisons)

Unit of analysis: the seed (per-seed accuracy), so pairs are matched.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from statistics import mean, pstdev, stdev
from typing import Any, Callable, Iterable

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20240905
PERMUTATION_SAMPLES = 20000

Config = str
Comparisons = list[tuple[str, str]]  # (baseline pair label handled by caller)


@dataclass(slots=True)
class CompareResult:
    a: str
    b: str
    n: int
    mean_a: float
    mean_b: float
    diff: float
    p_one_sided: float
    p_two_sided: float
    bootstrap_ci: tuple[float, float]
    cohens_d: float
    significant_holm: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "n": self.n,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "diff": round(self.diff, 4),
            "p_one_sided": round(self.p_one_sided, 4),
            "p_two_sided": round(self.p_two_sided, 4),
            "ci95": [round(self.bootstrap_ci[0], 4), round(self.bootstrap_ci[1], 4)],
            "cohens_d": round(self.cohens_d, 4),
            "significant_holm": self.significant_holm,
        }


def _permutation_p_value(
    diffs: list[float],
    rng: random.Random,
    samples: int = PERMUTATION_SAMPLES,
) -> tuple[float, float]:
    """Two- and one-sided (mean>0) p-values for paired diffs, permutation test."""
    obs = mean(diffs)
    count_ge = 0
    count_two = 0
    n = len(diffs)
    for _ in range(samples):
        signs = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
        perm_mean = mean(s * d for s, d in zip(signs, diffs))
        if perm_mean >= obs:
            count_ge += 1
        if abs(perm_mean) >= abs(obs):
            count_two += 1
    return (count_ge + 1) / (samples + 1), (count_two + 1) / (samples + 1)


def _bootstrap_ci(
    diffs: list[float],
    rng: random.Random,
    resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
) -> tuple[float, float]:
    n = len(diffs)
    means: list[float] = []
    for _ in range(resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo = means[int(alpha / 2 * len(means))]
    hi = means[int((1 - alpha / 2) * len(means)) - 1]
    return lo, hi


def paired_compare(
    a_name: str,
    b_name: str,
    a_scores: list[float],
    b_scores: list[float],
    rng: random.Random | None = None,
) -> CompareResult:
    """Compare two configs across paired seeds."""
    assert len(a_scores) == len(b_scores) > 1, "per-seed scores required"
    diffs = [x - y for x, y in zip(a_scores, b_scores)]
    rng = rng or random.Random(BOOTSTRAP_SEED)
    p1, p2 = _permutation_p_value(diffs, rng)
    ci = _bootstrap_ci(diffs, rng)
    sd = stdev(diffs) if len(diffs) > 1 else 0.0
    d = mean(diffs) / sd if sd > 0 else 0.0
    return CompareResult(
        a=a_name,
        b=b_name,
        n=len(a_scores),
        mean_a=mean(a_scores),
        mean_b=mean(b_scores),
        diff=mean(diffs),
        p_one_sided=p1,
        p_two_sided=p2,
        bootstrap_ci=ci,
        cohens_d=d,
    )


def holm_bonferroni(results: list[CompareResult]) -> list[CompareResult]:
    """Apply Holm correction to two-sided p... it's Holm-Bonferroni; keep 1-sided p."""
    order = sorted(range(len(results)), key=lambda i: results[i].p_one_sided)
    m = len(results)
    for rank, idx in enumerate(order):
        results[idx].significant_holm = results[idx].p_one_sided <= 0.05 / (m - rank)
    return results


def config_scores(
    runs: Iterable[dict[str, Any]],
    config: str,
    metric: str = "overall",
) -> list[float]:
    return [r["results"][config][metric] for r in runs]


def summarize(
    runs: list[dict[str, Any]],
    comparisons: Comparisons | None = None,
    metric: str = "overall",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Summarize a list of per-seed runs: descriptive + hypothesis tests."""
    rng = random.Random(BOOTSTRAP_SEED)

    configs = list(runs[0]["results"].keys())
    descr: dict[str, dict[str, Any]] = {}
    for c in configs:
        scores = config_scores(runs, c, metric)
        descr[c] = {
            "mean": mean(scores),
            "sd": pstdev(scores),
            "min": min(scores),
            "max": max(scores),
            "n": len(scores),
            "scores": [round(s, 4) for s in scores],
        }

    if comparisons is None:
        comparisons = [
            ("full", "base"),
            ("full", "null_bp"),
            ("full", "ab_no_generalize"),
            ("full", "ab_no_specialize"),
            ("full", "ab_no_compose"),
        ]

    tested: list[CompareResult] = []
    for a, b in comparisons:
        if a in configs and b in configs:
            tested.append(
                paired_compare(a, b, config_scores(runs, a, metric),
                               config_scores(runs, b, metric), rng)
            )
    tested = holm_bonferroni(tested)
    order = sorted(range(len(tested)), key=lambda i: tested[i].p_one_sided)
    tested = [tested[i] for i in order]

    return {
        "metric": metric,
        "n_seeds": len(runs),
        "configs": descr,
        "comparisons": [t.to_dict() for t in tested],
    }


def load(path: str | Any) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def report(runs: list[dict[str, Any]], out_path: str | None = None, **kw: Any) -> dict[str, Any]:
    summary = summarize(runs, **kw)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def cli_main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="TR-Core statistics analyzer")
    ap.add_argument("input", help="path to per-seed runs JSON (results/...json)")
    ap.add_argument("-o", "--out", default=None, help="output report JSON path")
    ap.add_argument("-m", "--metric", default="overall")
    args = ap.parse_args()

    runs = load(args.input)
    s = report(runs, args.out, metric=args.metric)

    print(f"n seeds: {s['n_seeds']}  metric: {s['metric']}\n")
    for c, d in s["configs"].items():
        print(f"  {c:22s} mean={d['mean']:.3f} sd={d['sd']:.3f} "
              f"[{d['min']:.3f}..{d['max']:.3f}]")
    print()
    for t in s["comparisons"]:
        sig = "***" if t["significant_holm"] else ""
        print(f'  {t["a"]:<20s} vs {t["b"]:<20s} d={t["diff"]:+.3f} '
              f'p1={t["p_one_sided"]:.4f} p2={t["p_two_sided"]:.4f} '
              f'dz={t["cohens_d"]:.2f} ci={t["ci95"]} {sig}')


def developmental_trend(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Linear regression of overall ~ log2(frac) across fractions.

    Accumuates per-seed (overall, log2 frac) pairs; fits slope via
    least squares (naive, no scipy). Returns slope, intercept, and a
    permutation p-value for slope>0 across seeds.
    """
    import math as _math

    fracs = sorted(
        float(fr) for fr in data[0]["curve"] if fr != "seed"
    )
    xs = [_math.log2(fr) for fr in fracs]
    pairs: list[tuple[float, float]] = []
    for seed in data:
        for fr in fracs:
            y = seed["curve"][str(fr)]["overall"]
            pairs.append((_math.log2(fr), y))
    # least squares
    n = len(pairs)
    mx = sum(p for p, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    return {
        "fracs": fracs,
        "log2_xs": xs,
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "n_points": n,
    }


if __name__ == "__main__":
    cli_main()