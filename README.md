# TR-Core: Autonomous Symbolic Induction for Structural-OOD Generalization

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
[![Paper](https://img.shields.io/badge/arXiv-2026.XXXXX-orange.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23619568.svg)](https://doi.org/10.5281/zenodo.23619568)

> **TR-Core** is a symbolic cognitive architecture that autonomously discovers compositional rules from raw interaction without pretrained weights, embeddings, or LLMs. It achieves structural out-of-distribution generalization through an Apriori-style induction engine with exact binomial statistical filtering.

## 🎯 Key Results

| Experiment | Task | TR-Core | Best Baseline | Ablation (no COMPOSE) |
|------------|------|---------|---------------|----------------------|
| **Phase 3** | Rare Compositional Rules (8 attrs, 4,608 contexts) | **0.78–0.80** balanced acc | 0.81 (ValueDListML) | **0.20** (chance) |
| **Phase 4** | Structural-OOD (held-out triple, zero support) | **0.96–1.00** held-out acc | 0.98 (Decision Tree) | **0.00** |

**Key Finding**: COMPOSE operation is the *causal mechanism* for structural-OOD generalization — ablations collapse to chance level.

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              ATTRIBUTE TASK ENVIRONMENT                     │
│       (Gymnasium: V3 Rare / V4 Structural-OOD)             │
└─────────────────────────┬───────────────────────────────────┘
                          │ observe(context, action, success)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  BEHAVIORAL PROGRAM V3                      │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │   RULE BASE     │  │     INDUCTION ENGINE            │  │
│  │  • LATENT       │  │  • Coverage pruning (c_min=4)   │  │
│  │  • ACTIVE       │  │  • Binomial filter (α=1e-3)     │  │
│  │  • RETIRED      │  │  • Precision ≥ 0.95             │  │
│  │  • DELETED      │  │  • Max arity K=4                │  │
│  └────────┬────────┘  └──────────────┬──────────────────┘  │
│           │                          │                      │
│           └──────────┬───────────────┘                      │
│                      ▼                                       │
│              SUBPROGRAMS (COMPOSE)                          │
└─────────────────────────────────────────────────────────────┘
```

**7 Development Operations**: `CREATE`, `GENERALIZE`, `SPECIALIZE`, `SPLIT`, `COMPOSE`, `MODIFY`, `DELETE`

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/Mardon-programm/tabula-rasa-core.git
cd tabula-rasa-core

# Using uv (recommended)
uv venv .venv && source .venv/bin/activate
uv pip install -e .[dev]

# Or with pip
pip install -e .[dev]
```

### Run Experiments
```bash
# Phase 3: Rare compositional rules
python -m experiments.run --experiment phase3 --seeds 1-24 --summarize --paired-stats

# Phase 4: Structural-OOD generalization
python -m experiments.run --experiment phase4 --seeds 1-24 --summarize --paired-stats

# Full DVC pipeline
dvc repro
```

### Run Demo
```bash
streamlit run web/app.py
# Select "Phase 4 Demo" in sidebar → Initialize → Collect → Develop → Test
```

### Gymnasium Environment
```python
import gymnasium as gym
from environment.gymnasium_env import register_envs

register_envs()
env = gym.make("TRCore-AttributeTaskV4-v0", seed=42)
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Success: {info['success']}, Oracle: {info['oracle_action']}")
```

## 📊 Reproducibility

| Component | Tool |
|-----------|------|
| Configuration | Hydra (YAML, sweep-ready) |
| Pipeline | DVC (14 stages) |
| Statistics | Permutation tests (20k), Cohen's dz, bootstrap CI |
| Environments | Gymnasium (`TRCore-AttributeTaskV3-v0`, `V4-v0`) |
| Seeds | 24 seeds × 2 replications (main + replication) |

## 📁 Project Structure

```
tabula-rasa-core/
├── behavioral/           # Core induction engine
│   ├── program.py        # BehavioralProgram (v0.2)
│   ├── program_v3.py     # Apriori induction (Phase 3/4)
│   └── rules.py          # Rule, Condition, Literal, Evidence
├── environment/          # Task environments
│   ├── attribute_tasks_v3.py
│   ├── attribute_tasks_v4.py
│   └── gymnasium_env.py  # Gymnasium interface
├── experiments/          # Experimentation framework
│   ├── experiment_phase3.py
│   ├── experiment_phase4.py
│   ├── analyze_results.py
│   ├── generate_figures.py
│   └── harness.py
├── conf/                 # Hydra configs
├── docs/                 # Paper, formal analysis
├── dvc.yaml              # Reproducible pipeline
├── pyproject.toml        # Package config
└── web/app.py           # Streamlit demo
```

## 📚 Citation

```bibtex
@article{trcore2026,
  title={TR-Core: Autonomous Symbolic Induction of Compositional Rules via Apriori Search with Binomial Filtering},
  author={TR-Core Research Group},
  year={2026},
  note={arXiv preprint}
}
```

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🔬 Related Work

- **Alan** — Next-gen embodied agent building on TR-Core: [github.com/Mardon-programm/alan](https://github.com/Mardon-programm/alan)
- **TR-Core Paper** — [docs/tr_core_paper.tex](docs/tr_core_paper.tex)
- **Formal Analysis** — [docs/apriori_formal_analysis.tex](docs/apriori_formal_analysis.tex)

---

*Independent research by Mardon (School No. 7, Osh, Kyrgyzstan)*
