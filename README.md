# TR-Core: Tabula Rasa Behavioral Program Induction

TR-Core is a **formal experimentation framework** for studying autonomous symbolic induction of compositional rules from experience. It implements a developmental cognitive architecture where a behavioral program (layer of changeable rules over fixed substrate) learns via seven operations: CREATE, MODIFY, GENERALIZE, SPECIALIZE, COMPOSE, SPLIT, DELETE.

**Key contribution**: An Apriori-style induction engine with binomial statistical filtering that discovers rare compositional rules (conjunctions of 3–4 literals) and generalizes to structurally held-out contexts — outperforming kNN, decision trees, random forests, and logistic regression on balanced accuracy.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ATTRIBUTE TASK ENVIRONMENT               │
│  (Gymnasium interface: V3 rare compositional, V4 Structural-OOD) │
└─────────────────────────┬───────────────────────────────────┘
                          │ observe(context, action, success)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    BEHAVIORAL PROGRAM (V3)                  │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │   RULE BASE     │  │     INDUCTION ENGINE (Apriori)  │  │
│  │  • LATENT       │  │  • Coverage-based pruning       │  │
│  │  • ACTIVE       │  │  • Binomial filter (α=1e-3)     │  │
│  │  • RETIRED      │  │  • Precision ≥ 0.95             │  │
│  │  • DELETED      │  │  • Max arity K=4                │  │
│  └────────┬────────┘  └──────────────┬──────────────────┘  │
│           │                          │                      │
│           └──────────┬───────────────┘                      │
│                      ▼                                     │
│              SUBPROGRAMS (COMPOSE)                          │
└─────────────────────────────────────────────────────────────┘
```

## Research Results (Phase 3 & 4)

| Experiment | Task | Key Finding |
|------------|------|-------------|
| **Phase 3** | Rare compositional rules (8 attrs, 4608 contexts) | BP balanced acc **0.78–0.80** vs kNN 0.70, RF 0.66, LogReg 0.20. COMPOSE necessary (ablation: 0.20). |
| **Phase 4** | Structural-OOD (held-out triple `calm∧guard∧night→approach`) | BP heldout acc **0.96** vs DListML 0.91, Tree 0.98, kNN 0.25. Pair rule `calm∧guard→approach` recovered **24/24** seeds. |

Mechanistic controls isolate COMPOSE as the causal mechanism:
- `null_bp` (randomized outcomes): collapses to chance (0.20)
- `ab_no_compose` (COMPOSE disabled): collapses to chance (0.20)

## Professional Lab Infrastructure

| Component | Status | Location |
|-----------|--------|----------|
| Formal complexity analysis (LaTeX) | ✅ | `docs/apriori_formal_analysis.tex` |
| Hydra configuration management | ✅ | `conf/` |
| DVC pipeline (data→exp→analysis→figures) | ✅ | `dvc.yaml` |
| Benchmark harness (sweeps, multi-seed) | ✅ | `experiments/harness.py` |
| Gymnasium environment interface | ✅ | `environment/gymnasium_env.py` |
| Publication-ready figure generation | ✅ | `experiments/generate_figures.py` |
| Statistical analysis (permutation tests, Cohen's dz) | ✅ | `experiments/analyze_results.py` |

## Installation

```bash
# Using uv (recommended)
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .[dev]

# Or with pip
pip install -e .[dev]
```

## Quick Start

### Run Phase 4 experiment (2 seeds for testing)
```bash
python -m experiments.run_hydra experiment=phase4 seed=1,2 --multirun
```

### Run with benchmark harness
```python
from experiments.harness import BenchmarkHarness

harness = BenchmarkHarness(config_path="conf", output_dir="results")

# Single run
result = harness.run_single("phase4", seed=1)
print(result.metrics)

# Parameter sweep
results = harness.run_sweep(
    "phase4",
    param_grid={"agent.max_arity": [2, 3, 4], "agent.min_coverage": [2, 4, 6]},
    seeds=[1, 2, 3, 4, 5],
)
summary = harness.summarize(results)
```

### Gymnasium interface (for RL baselines)
```python
import gymnasium as gym
from environment.gymnasium_env import register_envs

register_envs()
env = gym.make("TRCore-AttributeTaskV4-v0", seed=42)

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Success: {info['success']}, Oracle action: {info['oracle_action']}")
```

### DVC pipeline (full reproducible run)
```bash
dvc repro  # Runs data generation → experiments → analysis → figures
```

### Generate paper figures
```bash
python -m experiments.generate_figures \
    --phase3 results/phase3_analysis.json \
    --phase4 results/phase4_analysis.json \
    --output figures/
```

## Project Structure

```
tr-core/
├── behavioral/              # Behavioral Program (core induction)
│   ├── program.py          # Base BehavioralProgram (v0.2)
│   ├── program_v3.py       # Apriori induction (Phase 3/4)
│   └── rules.py            # Rule, Condition, Literal, Evidence
├── environment/             # Attribute task environments
│   ├── attribute_tasks_v3.py   # Phase 3: rare compositional
│   ├── attribute_tasks_v4.py   # Phase 4: Structural-OOD
│   └── gymnasium_env.py        # Gymnasium wrapper
├── experiments/             # Experimentation framework
│   ├── experiment_phase3.py    # Phase 3 runner
│   ├── experiment_phase4.py    # Phase 4 runner
│   ├── compositional_baselines.py  # kNN, Tree, Forest, DList, etc.
│   ├── harness.py              # BenchmarkHarness (sweeps, multi-seed)
│   ├── run_hydra.py            # Hydra entry point
│   ├── analyze_results.py      # Paired permutation tests
│   └── generate_figures.py     # Publication figures
├── conf/                      # Hydra configs
│   ├── config.yaml
│   ├── experiment/phase3.yaml, phase4.yaml
│   ├── agent/behavioral_program_v3.yaml
│   └── env/attribute_tasks_v3.yaml, v4.yaml
├── docs/
│   └── apriori_formal_analysis.tex  # Complexity proofs, guarantees
├── dvc.yaml                   # DVC pipeline
├── pyproject.toml             # Package config
└── tests/                     # Unit tests
```

## Configuration (Hydra)

```yaml
# conf/experiment/phase4.yaml
experiment:
  name: phase4
  n_seeds: 24
  seed_base: 7000
  train_size: 600
  heldout_test_size: 60
  
agent:
  max_arity: 4
  min_coverage: 4
  validate_accuracy: 0.85
  theta: 0.95
  alpha: 1e-3
```

Override via CLI:
```bash
python -m experiments.run_hydra experiment=phase4 agent.max_arity=3 agent.min_coverage=6
```

## Citation

```bibtex
@article{trcore2026,
  title={TR-Core: Autonomous Symbolic Induction of Compositional Rules via Apriori Search with Binomial Filtering},
  author={TR-Core Research Group},
  year={2026},
  note={arXiv preprint}
}
```

## License

MIT License