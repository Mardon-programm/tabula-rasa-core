#!/usr/bin/env python3
"""
Data Generation Script for TR-Core Experiments.

Generates train/known/test splits from attribute task environments
and saves them as JSON for DVC tracking and reproducible experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf


def context_to_dict(ctx: Any) -> Dict[str, Any]:
    """Convert context object to dictionary."""
    if hasattr(ctx, '__dict__'):
        return {k: v for k, v in ctx.__dict__.items() if not k.startswith('_')}
    elif isinstance(ctx, dict):
        return dict(ctx)
    else:
        return dict(ctx)


def generate_data(cfg: DictConfig, output_dir: Path) -> None:
    """Generate data splits from environment config."""
    # Instantiate environment
    env = hydra.utils.instantiate(cfg.env)
    
    # Generate splits
    sets = env.generate()
    
    # Convert to serializable format
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each split
    splits = {}
    for attr_name in dir(sets):
        if attr_name.startswith('_') or attr_name in ['generate']:
            continue
        value = getattr(sets, attr_name)
        if isinstance(value, list):
            splits[attr_name] = [context_to_dict(c) for c in value]
    
    # Save combined file
    combined_path = output_dir / "splits.json"
    combined_path.write_text(json.dumps(splits, indent=2))
    
    # Save individual files for convenience
    for name, data in splits.items():
        (output_dir / f"{name}.json").write_text(json.dumps(data, indent=2))
    
    print(f"Generated {len(splits)} splits in {output_dir}")
    for name, data in splits.items():
        print(f"  {name}: {len(data)} samples")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Experiment config name")
    parser.add_argument("--output", type=str, help="Output directory")
    args, _ = parser.parse_known_args()
    
    if args.config:
        # Override experiment config
        cfg = hydra.compose(config_name="config", overrides=[f"experiment={args.config}"])
    
    output_dir = Path(args.output) if args.output else Path("data") / cfg.experiment.name
    generate_data(cfg, output_dir)


if __name__ == "__main__":
    main()