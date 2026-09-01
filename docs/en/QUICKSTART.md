# TR-Core Quick Start Guide

## Installation

```bash
cd tabula_rasa_core
pip install -r requirements.txt
```

## Running Your First Episode

```python
from main import TabulaRasaCore
from environment.sandbox_sim import SandboxEnvironment

agent = TabulaRasaCore(seed=42, enable_metrics=True, verbose=True)
env = SandboxEnvironment(seed=42)

stats = agent.run_episode(env, max_steps=100)
agent.print_status()
```

## Understanding the Agent Loop

Each step executes:

1. **Perceive** - Process environmental observation
2. **Predict** - Generate prediction for next state
3. **Compare** - Evaluate prediction accuracy
4. **Adapt** - Update world model based on errors
5. **Memory** - Store episode and consolidate knowledge
6. **Act** - Select next action (explore or plan)

## Key Metrics

- **Prediction Accuracy** - How well the agent predicts next states
- **Calibration Error** - Confidence vs actual accuracy
- **Concept Drift Signal** - Environmental change detection
- **Exploration Rate** - Balance between exploration and exploitation

## Configuration

Modify agent initialization for different behaviors:

```python
agent = TabulaRasaCore(
    seed=42,
    enable_metrics=True,
    verbose=True
)
```

## Saving and Loading

```python
agent.save_checkpoint("my_checkpoint")
agent.load_checkpoint("my_checkpoint")
```

## Exporting Results

```python
metrics_file = agent.export_metrics("results.json")
```

## Next Steps

- Read ARCHITECTURE.md for system design
- Review RESEARCH_GUIDE.md for research applications
- Check test suite in tests/ for usage examples
