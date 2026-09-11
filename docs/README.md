# TR-Core Documentation

**Tabula Rasa Core: Autonomous Symbolic Induction of Compositional Rules**

Current version: **v0.3** — Behavioral Program with Apriori Induction (Phase 3/4)

---

## 📚 Documentation Index

### For Researchers (Primary)

| Document | Description |
|----------|-------------|
| **[Root README](../README.md)** | Complete project overview, results, commands |
| **[Paper (LaTeX)](tr_core_paper.tex)** | arXiv-ready manuscript with formal appendix |
| **[Formal Analysis](apriori_formal_analysis.tex)** | Complexity proofs, statistical guarantees, recovery theorems |
| **[Bibliography](references.bib)** | 14 key citations |

### For Developers

| Document | Description |
|----------|-------------|
| **[Benchmark Harness](../experiments/harness.py)** | `BenchmarkHarness.run_sweep(param_grid, seeds)` |
| **[Experiment Runner](../experiments/run.py)** | CLI: `python -m experiments.run --experiment phase4 --seeds 1-24` |
| **[Statistical Analysis](../experiments/analyze_results.py)** | Permutation tests, Cohen's dz, bootstrap CI |
| **[Figure Generation](../experiments/generate_figures.py)** | Publication-ready PDF/PNG |
| **[Gymnasium Environments](../environment/gymnasium_env.py)** | `gym.make("TRCore-AttributeTaskV4-v0")` |

### Configuration (Hydra)

```
conf/
├── config.yaml                    # Main config
├── experiment/phase3.yaml         # Phase 3: rare compositional
├── experiment/phase4.yaml         # Phase 4: Structural-OOD
├── agent/behavioral_program_v3.yaml  # Apriori engine params
├── env/attribute_tasks_v3.yaml    # V3 environment
└── env/attribute_tasks_v4.yaml    # V4 environment
```

### DVC Pipeline

```bash
dvc repro  # Full reproducible: data → experiments → analysis → figures
```

**Stages (14):** data gen, main/replication runs, ablations, parameter sweeps, analysis, figures.

---

## 🔬 Current Research Status (Phase 3/4)

### Phase 3: Rare Compositional Rules
- **Environment:** 8 attributes, 4608 contexts, 5 actions
- **Result:** `full_bp` balanced acc **0.784 / 0.803** (main/replication)
- **Beats:** kNN (0.70), RF (0.66), LogReg (0.20) — all $p < 10^{-4}$
- **Ablation:** COMPOSE necessary — without it → chance (0.20)

### Phase 4: Structural-OOD Generalization
- **Environment:** 6 attributes, 864 contexts, 4 actions
- **Held-out:** `calm∧guard∧night → approach` (zero train support)
- **Result:** `full_bp` heldout acc **0.958 / 1.000**
- **Pair rule** `calm∧guard→approach` recovered **24/24** seeds
- **Ablation:** COMPOSE causal — without it → 0.000

---

## 🚀 Quick Commands

```bash
# Install
uv venv .venv && source .venv/bin/activate
uv pip install -e .[dev]

# Full experiments (24 seeds × 2 replications)
python -m experiments.run --experiment phase3 --seeds 1-24 --summarize --paired-stats
python -m experiments.run --experiment phase4 --seeds 1-24 --summarize --paired-stats

# Parameter sweep
python -c "
from experiments.harness import BenchmarkHarness
h = BenchmarkHarness()
r = h.run_sweep('phase4', {'agent.max_arity': [2,3,4], 'agent.min_coverage': [2,4,6]}, seeds=range(1,6))
print(h.summarize(r))
"

# Analysis & figures
python -m experiments.analyze_results --input results/phase4_main.json --replication results/phase4_replication.json --output results/phase4_analysis.json --baseline full_bp --targets ab_no_compose null_bp ValuekNN ValueDListML ValueTree ValueForest
python -m experiments.generate_figures --phase4 results/phase4_analysis.json --phase4-repl results/phase4_analysis.json --output figures

# Gymnasium baseline
python -c "
import gymnasium as gym
from environment.gymnasium_env import register_envs
register_envs()
env = gym.make('TRCore-AttributeTaskV4-v0')
"
```

---

## 📁 Key Source Files

```
behavioral/
├── program.py          # Base BehavioralProgram (v0.2 spec)
├── program_v3.py       # Apriori induction engine (Phase 3/4)
└── rules.py            # Rule, Condition, Literal, Evidence

environment/
├── attribute_tasks_v3.py   # Phase 3: rare compositional
├── attribute_tasks_v4.py   # Phase 4: Structural-OOD
└── gymnasium_env.py        # Gymnasium wrapper

experiments/
├── experiment_phase3.py
├── experiment_phase4.py
├── compositional_baselines.py  # kNN, Tree, RF, DList, etc.
├── harness.py
├── run.py
├── analyze_results.py
└── generate_figures.py
```

---

## 📄 Legacy Documentation (v0.1/v0.2)

The `en/` and `ru/` folders contain documentation for the **previous architecture** (10 cognitive systems: substrate, graph brain, world model, prediction, adaptation, memory, self-model, planner, curiosity, developmental stages).

**Not applicable to current Behavioral Program work** — kept for historical reference.

---

## 🎯 Target Venues

- **NeurIPS 2026** — "Cognitive Architectures for Compositional Generalization" workshop
- **AAAI 2026** — "Neuro-Symbolic AI" track
- **IROS 2026** — "Cognitive Robotics" workshop

---

## 📞 Contact

TR-Core Research Group  
Repository: `https://github.com/tabula-rasa/tr-core`  
arXiv: (pending LaTeX compilation)