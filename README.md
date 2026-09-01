# TR-Core: Tabula Rasa Developmental Cognitive Agent

A complete research demonstration of autonomous continual learning, world model formation, and metacognition without pretrained neural weights.

## Overview

TR-Core is an experimental developmental cognitive architecture designed to investigate whether a persistent agent can autonomously:

- **Perceive** complex environments through sensory substrates
- **Learn** state transitions and build an internal world model
- **Predict** future states with calibrated confidence
- **Detect** concept drift and adapt beliefs without catastrophic forgetting
- **Explore** strategically using curiosity-driven reinforcement learning
- **Remember** through episodic→semantic consolidation
- **Plan** multi-step action sequences via tree search
- **Understand** its own capabilities and limitations (metacognition)
- **Develop** new capabilities through interaction

## Architecture

```
                  ENVIRONMENT
                      │
                      ▼
                  SUBSTRATE (perception)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
   WORLD MODEL                   MEMORY
   (what exists)            (what happened)
        │                           │
        └──────────┬────────────────┘
                   ▼
            PREDICTION ENGINE
            (what happens next)
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
    ERROR SIGNAL         CURIOSITY
         │                    │
         └──────┬─────────────┘
                ▼
            ADAPTATION
         (update beliefs)
                │
         ┌──────┴──────────┐
         ▼                 ▼
    SELF-MODEL      CONSOLIDATION
    (metacognition)   (episodic→semantic)
         │                 │
         └──────┬──────────┘
                ▼
            PLANNER
         (tree search)
                │
                ▼
             ACTION
                │
                └──────────→ ENVIRONMENT
```

## Project Structure

```
tabula_rasa_core/
│
├── core/                      # Core cognitive systems
│   ├── substrate.py          # Sensory processing
│   ├── graph_brain.py        # Neural-like graph
│   ├── world_model.py        # Environment model
│   ├── prediction.py         # Predictive engine
│   ├── adaptation.py         # Belief updating
│   ├── curiosity_rl.py       # Exploration drive
│   ├── developmental_stage.py # Development tracking
│   ├── axiom_matrix.py       # Constraints
│   └── __init__.py
│
├── memory/                    # Memory systems
│   ├── episodic.py           # Recent experiences
│   ├── semantic.py           # Extracted knowledge
│   ├── consolidation.py      # Episodic→semantic
│   └── __init__.py
│
├── models/                    # Agent models
│   ├── state.py              # State management
│   ├── self_model.py         # Metacognition
│   └── __init__.py
│
├── planner/                   # Planning & search
│   ├── tree_search.py        # Tree search planner
│   └── __init__.py
│
├── environment/              # Test environments
│   ├── sandbox_sim.py        # Simple simulation
│   ├── world_model.json      # Learned model
│   └── __init__.py
│
├── experiments/              # Benchmarking framework
│   ├── metrics.py            # Metric collection
│   ├── experiments.py        # Benchmark definitions
│   ├── runner.py             # Experiment runner
│   └── __init__.py
│
├── persistence/              # State persistence
│   ├── storage.py            # Checkpointing
│   └── __init__.py
│
├── tests/                    # Test suite
│   ├── test_substrate.py
│   ├── test_graph.py
│   ├── test_prediction.py
│   ├── test_adaptation.py
│   ├── test_memory.py
│   └── ...
│
├── main.py                   # Main entry point
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Core Systems

### 1. **Substrate** (`core/substrate.py`)
Sensory processing layer that quantizes raw observations into discrete states.
- Entropy-based segmentation
- Windowed observation buffering
- Modality tracking

### 2. **Graph Brain** (`core/graph_brain.py`)
Neural-like directed graph representing learned associations.
- Weighted synapses with learning rate
- Activation counting and age tracking
- Decay and pruning for stability

### 3. **World Model** (`core/world_model.py`)
Comprehensive representation of environment structure.
- Entity tracking with properties
- Relation modeling
- State transition statistics
- Causal link discovery
- Uncertainty quantification (epistemic, aleatoric, model)

### 4. **Prediction Engine** (`core/prediction.py`)
Generates predictions with calibrated confidence.
- Multi-state prediction with alternatives
- Prediction error evaluation
- Calibration tracking
- Concept drift signal detection

### 5. **Adaptation Engine** (`core/adaptation.py`)
Updates beliefs based on prediction errors.
- Plasticity vs. stability balance
- Concept drift handling
- Confidence adjustment
- Recovery after drift measurement

### 6. **Memory Systems**
Hierarchical memory organization:

- **Episodic** (`memory/episodic.py`): Recent experiences (FIFO buffer)
- **Semantic** (`memory/semantic.py`): Extracted facts with confidence
- **Consolidation** (`memory/consolidation.py`): Transfer + forgetting

### 7. **Self-Model** (`models/self_model.py`)
Metacognitive representation of agent capabilities.
- Prediction accuracy tracking
- Capability confidence estimation
- Knowledge vs. unknown estimation
- Exploration/exploitation decision

### 8. **Planner** (`planner/tree_search.py`)
Multi-step action planning via tree search.
- Depth-limited search
- Uncertainty-aware evaluation
- Goal-directed planning

### 9. **Curiosity Engine** (`core/curiosity_rl.py`)
Exploration drive combining novelty, uncertainty, and prediction error.
- State visit tracking
- Novelty decay
- Intrinsic reward calculation

## Running Experiments

### Basic Usage

```python
from main import TabulaRasaCore
from environment.sandbox_sim import SandboxEnvironment

# Create agent
agent = TabulaRasaCore(seed=42, enable_metrics=True)
env = SandboxEnvironment(seed=42)

# Run episode
stats = agent.run_episode(env, max_steps=100)
agent.print_status()
```

### Running Benchmarks

```python
from experiments.runner import ExperimentRunner

runner = ExperimentRunner(agent, output_dir="results")
results = runner.run_all_standard_experiments()
runner.print_summary()
runner.save_report()
```

### Available Experiments

1. **Learning Curve** - How fast does agent learn?
2. **Prediction Accuracy** - How accurate are predictions?
3. **Adaptation** - How fast after environment change?
4. **Memory Retention** - What's retained after other learning?
5. **Planning** - Can it plan multi-step sequences?
6. **Concept Drift** - Can it detect and adapt to rule changes?

## Metrics & Evaluation

The system tracks comprehensive metrics:

```
Learning:
  - Prediction accuracy
  - Calibration error
  - Learning curve

Adaptation:
  - Adaptation time
  - Plasticity vs. stability
  - Recovery after drift
  - Drift detection rate

Memory:
  - Episodic buffer utilization
  - Semantic fact count
  - Retention rate
  - Consolidation efficiency

Planning:
  - Planning success rate
  - Plan depth

Exploration:
  - Curiosity score
  - Exploration rate
  - Novelty

Self-Model:
  - Estimated knowledge
  - Uncertainty level
  - Capability confidence
```

## Key Features

### ✅ Continual Learning
Agent learns sequentially without catastrophic forgetting through:
- Memory consolidation (episodic → semantic)
- Stability-plasticity balance in adaptation
- Context-dependent belief marking

### ✅ World Model Formation
Builds internal representation through autonomous interaction:
- Entity and relation discovery
- Causal link extraction
- Uncertainty estimation
- Incremental refinement

### ✅ Prediction with Error
Error-driven learning cycle:
- Generate prediction with confidence
- Compare to actual outcome
- Calculate prediction error
- Update uncertainty and beliefs

### ✅ Concept Drift Detection
Autonomously detects when environment rules change:
- Tracks prediction error trends
- Signals drift magnitude
- Triggers adaptive learning
- Retains old knowledge

### ✅ Metacognition
Agent understands its own limits:
- Tracks capability confidence by task
- Estimates knowledge coverage
- Adjusts exploration vs. exploitation
- Recommends learning rates

### ✅ Curiosity-Driven Exploration
Exploration balances multiple drives:
- Novelty (unexplored states)
- Uncertainty (low confidence)
- Information gain (informative experiences)

### ✅ Developmental Stages
Agent progresses through learning stages:
- BLANK → SENSORY → ASSOCIATIVE → PREDICTIVE
- EXPLORATORY → MEMORIAL → ADAPTIVE → PLANNING

### ✅ Reproducibility
Complete experiment tracking:
- Seeded randomness
- Checkpoint saving/loading
- Detailed metrics export
- JSON snapshot serialization

## Performance Targets

For a fully developed system, we target:

| Metric | Target | Interpretation |
|--------|--------|-----------------|
| Prediction Accuracy | >85% | Can reliably predict transitions |
| Calibration Error | <0.1 | Confidence matches reality |
| Adaptation Time | <50 steps | Quick recovery from drift |
| Memory Retention | >70% | Doesn't forget old knowledge |
| Planning Success | >80% | Effective multi-step planning |
| Concept Drift Detection | <20 steps | Fast drift identification |
| Knowledge Coverage | >80% | Explores most of environment |

## Research Questions

TR-Core addresses fundamental research questions:

1. **Can agents autonomously build world models through interaction?**
2. **How can systems balance learning new knowledge with retaining old?**
3. **Can agents detect and adapt to environmental shifts?**
4. **How does curiosity drive efficient exploration?**
5. **Can agents model their own capabilities?**
6. **Is error-driven learning sufficient for development?**

## Development Roadmap

### Phase 1 (Complete)
- Core perception and learning loops
- Graph brain and world model
- Prediction engine
- Basic curiosity

### Phase 2 (In Progress)
- Memory consolidation
- Concept drift detection
- Adaptation mechanisms
- Benchmark suite

### Phase 3 (Planned)
- Advanced planning (MCTS)
- Transfer learning
- Skill discovery
- Hierarchical learning

### Phase 4 (Future)
- Multi-agent scenarios
- Hierarchical world models
- Option learning
- Genuine developmental emergence

## Installation

```bash
# Clone repository
git clone https://github.com/user/tabula_rasa_core.git
cd tabula_rasa_core

# No external dependencies needed (uses Python stdlib only)
# Optional: install dev tools
pip install pytest pytest-cov black flake8 mypy
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=core --cov=memory --cov=models tests/

# Run specific test
pytest tests/test_prediction.py::test_prediction_accuracy
```

## Citation

If you use TR-Core in research, please cite:

```bibtex
@misc{tabularasacore2024,
  title={TR-Core: Tabula Rasa Developmental Cognitive Architecture},
  author={[Your Name]},
  year={2024},
  url={https://github.com/user/tabula_rasa_core}
}
```

## License

MIT License - See LICENSE file

## References

Key papers informing this architecture:

1. **Continual Learning**: [Continual Lifelong Learning with Dynamic Synaptic Plasticity](https://arxiv.org/abs/2105.10919)
2. **World Models**: [World Models (Ha & Schmidhuber)](https://world-models.io/)
3. **Curious RL**: [Curiosity-Driven Exploration by Self-Supervised Prediction](https://arxiv.org/abs/1705.05363)
4. **Developmental Robotics**: [Developmental Robotics Overview](https://doi.org/10.1080/01691864.2023.2225232)
5. **Meta-Learning**: [Meta-Learning: A Survey](https://arxiv.org/abs/1810.03548)

## Contact & Contributions

- Questions? Open an issue
- Want to contribute? See CONTRIBUTING.md
- Have ideas? Start a discussion

---

**Status**: Research Prototype (v1.0)
**Last Updated**: 2024
**Maintained by**: [Your Team]
