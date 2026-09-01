# TR-Core Research Guide

## Overview

TR-Core is a reproducible research platform for studying autonomous continual learning, developmental cognition, and metacognition without pretrained weights.

## Research Applications

### 1. Autonomous Learning

Study how agents learn state transitions and build world models through interaction.

**Key metrics:**
- World model accuracy
- Learning curve (error vs steps)
- Knowledge coverage

### 2. Concept Drift Detection

Investigate environmental change detection and adaptation strategies.

**Experimental design:**
- Introduce systematic changes in state transition probabilities
- Measure drift detection latency
- Track recovery after adaptation

### 3. Plasticity vs Stability

Balance between learning new information and retaining old knowledge.

**Measurements:**
- Plasticity: rate of belief updates
- Stability: retention of old knowledge
- Interference: conflicts between new and old patterns

### 4. Curiosity-Driven Exploration

Study how intrinsic motivation drives exploration.

**Components:**
- Novelty: visiting new states
- Uncertainty: low-confidence predictions
- Information gain: expected learning value

### 5. Memory Consolidation

Investigate episodic→semantic memory transfer.

**Analysis:**
- Consolidation rate
- Knowledge extraction quality
- Forgetting patterns

### 6. Metacognition

Study agent self-awareness and capability modeling.

**Metrics:**
- Self-model accuracy
- Uncertainty estimates
- Exploration/exploitation decisions

## Experimental Framework

### Setup

```python
from main import TabulaRasaCore
from experiments.runner import ExperimentRunner

runner = ExperimentRunner()
results = runner.run_comparative_experiments(
    conditions=["baseline", "high_drift", "curiosity_disabled"],
    episodes=10,
    steps_per_episode=500
)
```

### Metrics Collection

The MetricsCollector automatically tracks:

- Prediction accuracy and calibration
- Adaptation events and drift detections
- Memory statistics (episodic and semantic)
- Self-model metrics
- Curiosity scores
- Planning success rates

### Data Export

```python
agent.export_metrics("experiment_results.json")
```

## Publication Ideas

### Hypothesis 1: Autonomous World Model Formation

**Claim:** Agents can build accurate world models without supervision.

**Experiment:**
- Initialize agent in unknown environment
- Measure world model accuracy vs interaction steps
- Compare to random baseline

**Expected result:** Exponential improvement in prediction accuracy

### Hypothesis 2: Drift Detection without Retraining

**Claim:** Online adaptation enables drift detection without model retraining.

**Experiment:**
- Train agent until convergence
- Introduce concept drift at step N
- Measure time to detect and adaptation quality

**Expected result:** Detection within 5-10 steps, recovery within 20 steps

### Hypothesis 3: Curiosity Enables Exploration

**Claim:** Intrinsic motivation improves knowledge coverage vs random exploration.

**Experiment:**
- Run agent with curiosity enabled
- Run agent with curiosity disabled (random exploration)
- Compare knowledge coverage and world model completeness

**Expected result:** Curiosity group explores 30-50% more unique states

### Hypothesis 4: Metacognition Improves Decision-Making

**Claim:** Self-model awareness improves explore/exploit balance.

**Experiment:**
- Agent with self-model vs agent without
- Measure reward accumulation and prediction accuracy

**Expected result:** Self-model agent achieves 20-30% higher cumulative reward

## Reproducibility

All experiments use:
- Fixed random seed (42)
- Deterministic environment (SandboxEnvironment)
- Checkpoint system for replication
- Metrics export for analysis

## Custom Environments

Extend SandboxEnvironment for different experimental conditions:

```python
class CustomEnvironment(SandboxEnvironment):
    def step(self, action):
        # Custom logic here
        pass
```

## Analysis Tools

Use exported JSON metrics with:
- Pandas for data analysis
- Matplotlib for visualization
- Statistical tests (t-tests, ANOVA)

## Future Research Directions

1. **Hierarchical World Models** - Multi-level abstraction
2. **Transfer Learning** - Reuse world models in new tasks
3. **Social Learning** - Multiple agents exchanging knowledge
4. **Goal-Directed Learning** - Learning towards specific objectives
5. **Embodied Cognition** - Sensorimotor learning dynamics
