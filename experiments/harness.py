"""
Benchmark Harness for TR-Core Experiments.

Provides a unified interface for running experiments with Hydra configuration,
automated parameter sweeps, and structured output for DVC tracking.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@dataclass
class ExperimentResult:
    """Standardized experiment result container."""
    config_name: str
    seed: int
    metrics: Dict[str, float]
    artifacts: Dict[str, Any]
    runtime_seconds: float
    git_commit: Optional[str] = None


class BenchmarkHarness:
    """
    Main entry point for running TR-Core experiments.
    
    Usage:
        harness = BenchmarkHarness(config_path="conf")
        results = harness.run("phase4", seeds=[1,2,3], overrides={"agent.max_arity": 3})
    """
    
    def __init__(
        self,
        config_path: str = "conf",
        config_name: str = "config",
        output_dir: str = "results",
    ):
        self.config_path = Path(config_path).resolve()
        self.config_name = config_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Hydra with absolute path
        try:
            hydra.initialize_config_dir(config_dir=str(self.config_path), version_base=None)
            self.cfg = hydra.compose(config_name=config_name)
        except Exception as e:
            # Fallback: load configs manually without Hydra initialization
            import yaml
            self.cfg = OmegaConf.load(self.config_path / f"{config_name}.yaml")
        
    def run_single(
        self,
        experiment_name: str,
        seed: int,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ExperimentResult:
        """Run a single experiment seed."""
        start_time = time.time()
        
        # Load config manually (bypass Hydra compose issues)
        import yaml
        cfg = OmegaConf.load(self.config_path / f"{self.config_name}.yaml")
        exp_cfg = OmegaConf.load(self.config_path / "experiment" / f"{experiment_name}.yaml")
        agent_cfg = OmegaConf.load(self.config_path / "agent" / "behavioral_program_v3.yaml")
        env_cfg = OmegaConf.load(self.config_path / "env" / f"{experiment_name == 'phase3' and 'attribute_tasks_v3' or 'attribute_tasks_v4'}.yaml")
        
        # Apply overrides
        if overrides:
            for k, v in overrides.items():
                # Navigate to the right config section
                keys = k.split('.')
                target = cfg
                for key in keys[:-1]:
                    if key not in target:
                        target[key] = {}
                    target = target[key]
                target[keys[-1]] = v
        
        # Merge configs
        cfg = OmegaConf.merge(cfg, exp_cfg, agent_cfg, env_cfg)
        cfg.seed = seed
        
        # Instantiate environment generator (not gymnasium env)
        if experiment_name == "phase3":
            from environment.attribute_tasks_v3 import AttributeTaskGeneratorV3
            env = AttributeTaskGeneratorV3(
                seed=seed,
                train_size=cfg.train_size,
                known_size=cfg.known_size,
                t1_size=cfg.test_sizes.T1,
                t2_size=cfg.test_sizes.T2,
                t3_size=cfg.test_sizes.T3,
            )
        else:
            from environment.attribute_tasks_v4 import AttributeTaskGeneratorV4
            env = AttributeTaskGeneratorV4(
                seed=seed,
                train_size=cfg.train_size,
                known_size=cfg.known_size,
                heldout_test_size=cfg.heldout_test_size,
                row_test_size=cfg.row_test_size,
            )
        
        # Generate data splits
        sets = env.generate()
        
        # Run experiment (dispatch to appropriate runner)
        if experiment_name == "phase3":
            result = self._run_phase3(cfg, seed, sets, exp_cfg)
        elif experiment_name == "phase4":
            result = self._run_phase4(cfg, seed, sets, exp_cfg)
        else:
            raise ValueError(f"Unknown experiment: {experiment_name}")
            
        runtime = time.time() - start_time
        
        return ExperimentResult(
            config_name=experiment_name,
            seed=seed,
            metrics=result["metrics"],
            artifacts=result["artifacts"],
            runtime_seconds=runtime,
        )
    
    def _run_phase3(self, cfg: DictConfig, seed: int, sets: Any, exp_cfg: DictConfig) -> Dict[str, Any]:
        """Run Phase 3 experiment (delegates to existing implementation)."""
        from experiments.experiment_phase3 import run_phase3_seed
        
        # Convert Hydra config to args expected by run_phase3_seed
        result = run_phase3_seed(
            i=seed,
            seed_base=exp_cfg.seed_base,
            train_size=exp_cfg.train_size,
            known_size=exp_cfg.known_size,
            test_size=exp_cfg.test_sizes.get("T1", 200) * 3,
        )
        
        return {
            "metrics": {
                "balanced_accuracy": result["results"]["full_bp"]["row_balanced"],
                "overall_accuracy": result["results"]["full_bp"]["row_overall"],
                "known_accuracy": result["results"]["full_bp"]["known_acc"],
            },
            "artifacts": {
                "full_results": result,
            }
        }
    
    def _run_phase4(self, cfg: DictConfig, seed: int, sets: Any, exp_cfg: DictConfig) -> Dict[str, Any]:
        """Run Phase 4 experiment (delegates to existing implementation)."""
        from experiments.experiment_phase4 import run_phase4_seed
        
        result = run_phase4_seed(
            i=seed,
            seed_base=exp_cfg.seed_base,
            train_size=exp_cfg.train_size,
            known_size=exp_cfg.known_size,
            heldout_test_size=exp_cfg.heldout_test_size,
            row_test_size=exp_cfg.row_test_size,
        )
        
        return {
            "metrics": {
                "heldout_accuracy": result["results"]["full_bp"]["heldout_acc"],
                "row_balanced_accuracy": result["results"]["full_bp"]["row_balanced"],
                "row_overall_accuracy": result["results"]["full_bp"]["row_overall"],
                "known_accuracy": result["results"]["full_bp"]["known_acc"],
            },
            "artifacts": {
                "full_results": result,
            }
        }
    
    def run_sweep(
        self,
        experiment_name: str,
        param_grid: Dict[str, List[Any]],
        seeds: List[int],
        fixed_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[ExperimentResult]:
        """
        Run a parameter sweep across multiple seeds.
        
        Args:
            experiment_name: Name of experiment (phase3, phase4)
            param_grid: Dict mapping parameter paths to lists of values
            seeds: List of seeds to run
            fixed_overrides: Overrides applied to all runs
            
        Returns:
            List of ExperimentResult for all combinations
        """
        import itertools
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        results = []
        total = len(combinations) * len(seeds)
        current = 0
        
        print(f"Starting sweep: {len(combinations)} param combos × {len(seeds)} seeds = {total} runs")
        
        for combo in combinations:
            combo_overrides = dict(zip(param_names, combo))
            if fixed_overrides:
                combo_overrides.update(fixed_overrides)
                
            for seed in seeds:
                current += 1
                print(f"  [{current}/{total}] {experiment_name} seed={seed} params={combo_overrides}")
                
                try:
                    result = self.run_single(experiment_name, seed, combo_overrides)
                    results.append(result)
                    
                    # Save incrementally
                    self._save_incremental(results, experiment_name)
                    
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results.append(ExperimentResult(
                        config_name=experiment_name,
                        seed=seed,
                        metrics={"error": str(e)},
                        artifacts={},
                        runtime_seconds=0,
                    ))
                    
        return results
    
    def _save_incremental(self, results: List[ExperimentResult], experiment_name: str) -> None:
        """Save results incrementally to JSON."""
        output_path = self.output_dir / f"{experiment_name}_sweep_results.json"
        data = [asdict(r) for r in results]
        output_path.write_text(json.dumps(data, indent=2, default=str))
    
    def summarize(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Compute summary statistics across seeds."""
        import numpy as np
        from collections import defaultdict
        
        # Group by config (param combination)
        grouped = defaultdict(list)
        for r in results:
            key = r.config_name  # Could include param hash
            grouped[key].append(r)
            
        summary = {}
        for config_name, runs in grouped.items():
            metrics = defaultdict(list)
            for r in runs:
                for k, v in r.metrics.items():
                    if isinstance(v, (int, float)):
                        metrics[k].append(v)
                        
            summary[config_name] = {
                "n_seeds": len(runs),
                "metrics": {
                    k: {
                        "mean": float(np.mean(v)),
                        "std": float(np.std(v)),
                        "min": float(np.min(v)),
                        "max": float(np.max(v)),
                    }
                    for k, v in metrics.items()
                }
            }
            
        return summary


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """CLI entry point for running experiments via Hydra."""
    print(OmegaConf.to_yaml(cfg))
    
    harness = BenchmarkHarness()
    
    # Example: run single seed
    if cfg.get("single_seed"):
        result = harness.run_single(cfg.experiment.name, cfg.single_seed)
        print(f"Result: {result.metrics}")
        
    # Example: run sweep
    if cfg.get("sweep"):
        sweep_cfg = cfg.sweep
        param_grid = dict(sweep_cfg.param_grid)
        seeds = list(range(sweep_cfg.seeds.start, sweep_cfg.seeds.end + 1))
        results = harness.run_sweep(cfg.experiment.name, param_grid, seeds)
        summary = harness.summarize(results)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()