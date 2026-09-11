"""
Hydra-compatible experiment runner for TR-Core.

This module provides entry points that work with Hydra's multirun for
automated parameter sweeps across distributed environments.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf


def run_phase3_hydra(cfg: DictConfig) -> Dict[str, Any]:
    """Run Phase 3 experiment with Hydra config."""
    from experiments.experiment_phase3 import run_phase3_seed
    
    exp_cfg = cfg.experiment
    seed = cfg.seed
    
    result = run_phase3_seed(
        i=seed,
        seed_base=exp_cfg.seed_base,
        train_size=exp_cfg.train_size,
        known_size=exp_cfg.known_size,
        test_size=sum(exp_cfg.test_sizes.values()),
    )
    
    # Extract key metrics for Hydra logging
    metrics = {
        "balanced_accuracy": result["results"]["full_bp"]["row_balanced"],
        "overall_accuracy": result["results"]["full_bp"]["row_overall"],
        "known_accuracy": result["results"]["full_bp"]["known_acc"],
        "n_active_rules": result["results"]["full_bp"]["n_active"],
    }
    
    return {
        "metrics": metrics,
        "full_result": result,
    }


def run_phase4_hydra(cfg: DictConfig) -> Dict[str, Any]:
    """Run Phase 4 experiment with Hydra config."""
    from experiments.experiment_phase4 import run_phase4_seed
    
    exp_cfg = cfg.experiment
    seed = cfg.seed
    
    result = run_phase4_seed(
        i=seed,
        seed_base=exp_cfg.seed_base,
        train_size=exp_cfg.train_size,
        known_size=exp_cfg.known_size,
        heldout_test_size=exp_cfg.heldout_test_size,
        row_test_size=exp_cfg.row_test_size,
    )
    
    metrics = {
        "heldout_accuracy": result["results"]["full_bp"]["heldout_acc"],
        "row_balanced_accuracy": result["results"]["full_bp"]["row_balanced"],
        "row_overall_accuracy": result["results"]["full_bp"]["row_overall"],
        "known_accuracy": result["results"]["full_bp"]["known_acc"],
        "n_active_rules": result["results"]["full_bp"]["n_active"],
        "pair_rule_recovered": 1 if result["audit"]["found"] else 0,
        "heldout_region_coverage": result["audit"]["heldout_region_correct_full"],
    }
    
    return {
        "metrics": metrics,
        "full_result": result,
    }


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point - runs experiment based on config."""
    print("=" * 60)
    print(f"TR-Core Experiment: {cfg.experiment.name}")
    print(f"Seed: {cfg.seed}")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    
    start_time = time.time()
    
    if cfg.experiment.name == "phase3":
        output = run_phase3_hydra(cfg)
    elif cfg.experiment.name == "phase4":
        output = run_phase4_hydra(cfg)
    else:
        raise ValueError(f"Unknown experiment: {cfg.experiment.name}")
        
    runtime = time.time() - start_time
    
    # Add runtime to metrics
    output["metrics"]["runtime_seconds"] = runtime
    
    # Print metrics for Hydra sweep logging
    print("\nMETRICS:")
    for k, v in output["metrics"].items():
        print(f"  {k}: {v}")
    
    # Save full results
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    results_file = output_dir / "results.json"
    results_file.write_text(json.dumps(output, indent=2, default=str))
    
    print(f"\nSaved results to {results_file}")
    print(f"Runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()