#!/usr/bin/env python3
"""
Programmatic Hydra runner for TR-Core experiments.

Avoids @hydra.main decorator which has a Python 3.14 compatibility bug.
"""

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.experiment_phase3 import run_phase3_seed, summarize_study as summarize3, paired_stats as paired3
from experiments.experiment_phase4 import run_phase4_seed, summarize_study as summarize4, paired_stats as paired4


def run_experiment(experiment_name: str, seeds: list[int], seed_base: int = 7000, **kwargs) -> list[dict]:
    """Run experiment for multiple seeds."""
    results = []
    for seed in seeds:
        print(f"Running {experiment_name} seed={seed}...")
        if experiment_name == "phase3":
            result = run_phase3_seed(
                i=seed,
                seed_base=seed_base,
                train_size=kwargs.get("train_size", 2500),
                known_size=kwargs.get("known_size", 200),
                axis_sizes=kwargs.get("axis_sizes", {"t1": 200, "t2": 200, "t3": 200}),
            )
        elif experiment_name == "phase4":
            result = run_phase4_seed(
                i=seed,
                seed_base=seed_base,
                train_size=kwargs.get("train_size", 600),
                known_size=kwargs.get("known_size", 200),
                heldout_test_size=kwargs.get("heldout_test_size", 60),
                row_test_size=kwargs.get("row_test_size", 200),
            )
        else:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        results.append(result)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TR-Core Experiment Runner")
    parser.add_argument("--experiment", choices=["phase3", "phase4"], default="phase4")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24")
    parser.add_argument("--seed-base", type=int, default=7000)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--paired-stats", action="store_true")
    args = parser.parse_args()
    
    def parse_seeds(s: str) -> list[int]:
        result = []
        for part in s.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                result.extend(range(start, end + 1))
            else:
                result.append(int(part))
        return result
    
    seeds = parse_seeds(args.seeds)
    
    # Run experiment
    results = run_experiment(args.experiment, seeds, args.seed_base)
    
    # Save results
    import json
    from pathlib import Path
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    tag = "main" if args.seed_base == 7000 else "replication"
    output_path = output_dir / f"{args.experiment}_{tag}.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} results to {output_path}")
    
    # Summarize if requested
    if args.summarize:
        if args.experiment == "phase3":
            summarize3(output_path)
            if args.paired_stats:
                paired3(output_path)
        else:
            summarize4(output_path)
            if args.paired_stats:
                paired4(output_path)


if __name__ == "__main__":
    main()