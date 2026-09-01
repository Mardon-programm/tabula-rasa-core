# TR-Core Documentation Index

## Quick Navigation

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Installation and first run
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - System design overview
- **[README.md](../README.md)** - Project overview

### In-Depth Guides
- **[COMPLETION_REPORT.md](COMPLETION_REPORT.md)** - Project completion status and validation
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Detailed component descriptions
- **[RESEARCH_GUIDE.md](RESEARCH_GUIDE.md)** - Research applications and experimental design

## Project Structure

```
tabula_rasa_core/
├── main.py                 # Agent orchestrator
├── requirements.txt        # Dependencies
│
├── core/                   # Cognitive systems
│   ├── substrate.py       # Perception layer
│   ├── graph_brain.py     # Neural-like graph
│   ├── world_model.py     # Environment model
│   ├── prediction.py      # Forecasting engine
│   ├── adaptation.py      # Learning system
│   ├── curiosity_rl.py    # Exploration drive
│   ├── developmental_stage.py  # Stage progression
│   └── axiom_matrix.py    # Constraints
│
├── memory/                # Memory systems
│   ├── episodic.py       # Recent experiences
│   ├── semantic.py       # General knowledge
│   └── consolidation.py  # Episodic→semantic transfer
│
├── models/               # Agent models
│   ├── state.py         # State management
│   └── self_model.py    # Metacognition
│
├── planner/             # Planning systems
│   └── tree_search.py   # Tree search planner
│
├── environment/         # Test environments
│   ├── sandbox_sim.py   # Simple simulation
│   └── world_model.json # Learned model storage
│
├── experiments/         # Benchmarking
│   ├── metrics.py      # Metric collection
│   ├── experiments.py  # Benchmark definitions
│   └── runner.py       # Experiment runner
│
├── persistence/         # Storage
│   └── storage.py      # Checkpoint management
│
├── tests/              # Test suite
│   ├── conftest.py
│   ├── test_memory.py
│   ├── test_prediction.py
│   └── test_world_model.py
│
├── checkpoints/        # Agent checkpoints
├── results/            # Experiment results
└── docs/               # Documentation (this folder)
    ├── en/             # English documentation
    └── ru/             # Russian documentation
```

## Core Concepts

### Cognitive Systems (10 modules)

| System | File | Purpose |
|--------|------|---------|
| Substrate | `core/substrate.py` | Sensory perception & signal processing |
| Graph Brain | `core/graph_brain.py` | Neural-like association learning |
| World Model | `core/world_model.py` | Environment representation |
| Prediction Engine | `core/prediction.py` | State forecasting |
| Adaptation | `core/adaptation.py` | Error-driven learning |
| Curiosity | `core/curiosity_rl.py` | Exploration motivation |
| Memory | `memory/` | Episodic + semantic storage |
| Self-Model | `models/self_model.py` | Metacognition |
| Planner | `planner/tree_search.py` | Action planning |
| Development | `core/developmental_stage.py` | Stage progression |

### Agent Loop

Each step: **Perceive → Predict → Compare → Adapt → Memorize → Act**

### Key Metrics

- **Prediction Accuracy** - How well the agent predicts
- **Calibration Error** - Confidence vs actual performance
- **Concept Drift Signal** - Environmental change detection
- **Curiosity Score** - Exploration motivation
- **Memory Stats** - Episodic/semantic buffer usage
- **Self-Model State** - Agent's self-awareness

## Usage Patterns

### Basic Usage
```python
from main import TabulaRasaCore
from environment.sandbox_sim import SandboxEnvironment

agent = TabulaRasaCore(seed=42, enable_metrics=True)
env = SandboxEnvironment(seed=42)
stats = agent.run_episode(env, max_steps=100)
```

### Custom Experiments
```python
from experiments.runner import ExperimentRunner

runner = ExperimentRunner()
results = runner.run_comparative_experiments(
    conditions=["baseline", "no_curiosity"],
    episodes=10,
    steps_per_episode=100
)
```

### Analysis
```python
import json
import pandas as pd

with open("results/tr_core_metrics.json") as f:
    metrics = json.load(f)
df = pd.DataFrame(metrics["history"])
df.plot(x="step", y="prediction_accuracy")
```

## Research Topics

### Autonomous Learning
- World model formation
- Transition learning
- State discovery

### Concept Drift
- Drift detection
- Adaptation strategies
- Recovery measurement

### Plasticity vs Stability
- Learning rate dynamics
- Knowledge retention
- Interference handling

### Curiosity
- Intrinsic motivation
- Exploration efficiency
- Novelty and uncertainty

### Memory
- Consolidation
- Forgetting
- Knowledge extraction

### Metacognition
- Self-awareness
- Uncertainty estimates
- Decision support

## Configuration Reference

### Agent Initialization
```python
TabulaRasaCore(
    seed=42,              # Random seed
    enable_metrics=True,  # Collect metrics
    verbose=True          # Print logs
)
```

### Key Parameters

| Component | Parameter | Default | Range |
|-----------|-----------|---------|-------|
| Substrate | window_size | 4 | 2-16 |
| Substrate | entropy_threshold | 0.4 | 0.0-1.0 |
| Brain | learning_rate | 0.2 | 0.0-1.0 |
| Brain | decay_rate | 0.005 | 0.0-0.1 |
| Prediction | - | - | - |
| Adaptation | learning_rate | 0.1 | 0.0-1.0 |
| Adaptation | drift_threshold | 0.3 | 0.0-1.0 |
| Curiosity | novelty_weight | 0.4 | 0.0-1.0 |
| Planner | max_depth | 3 | 1-10 |

## Troubleshooting

### Agent not learning
- Check prediction accuracy metric
- Verify world model size growing
- Ensure environment is deterministic

### High concept drift signal
- Might indicate real environmental change
- Check drift detection is working
- Review learning rate settings

### Memory growing too fast
- Adjust episodic buffer limit
- Check semantic memory retention
- Review consolidation interval

## References

- See **README.md** for high-level overview
- See **ARCHITECTURE.md** for complete system design
- See **RESEARCH_GUIDE.md** for research methodology

## Further Reading

- Developmental Cognitive Science
- Reinforcement Learning theory
- World Models and Predictive Processing
- Metacognition in AI
