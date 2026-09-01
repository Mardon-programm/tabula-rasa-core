# TR-Core Architecture Overview

## System Philosophy

TR-Core is designed as a **reproducible research platform** for investigating:

- **How** agents autonomously build world models through interaction
- **How** systems maintain learning (plasticity) while preserving knowledge (stability)  
- **How** agents detect and adapt to environmental changes (concept drift)
- **How** curiosity drives efficient exploration
- **How** memory consolidation enables generalization
- **How** agents can model their own capabilities (metacognition)

NOT meant to be AGI, but rather a scientific instrument for these questions.

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN CONTROL LOOP                        │
│                     (main.py)                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
    PERCEPTION           EXPERIENCE
 (Substrate)           (Environment)
    │                     │
    │  ┌─────────────────┘
    │  │
    ▼  ▼
┌─────────────────────────────────────┐
│     COGNITIVE PROCESSING LOOP        │
│  (runs every step)                   │
└─────────────────────────────────────┘
    │
    ├─→ WORLD MODEL
    │   ├─ Entity tracking
    │   ├─ Relation discovery
    │   ├─ Transition statistics
    │   └─ Uncertainty maps
    │
    ├─→ PREDICTION
    │   ├─ Forecast next state
    │   ├─ Generate alternatives
    │   └─ Compute confidence
    │
    ├─→ ADAPTATION
    │   ├─ Compare prediction to outcome
    │   ├─ Update edge weights
    │   ├─ Detect concept drift
    │   └─ Adjust learning rate
    │
    ├─→ MEMORY
    │   ├─ Store episodic event
    │   ├─ Extract semantic patterns
    │   ├─ Consolidate knowledge
    │   └─ Selective forgetting
    │
    ├─→ SELF-MODEL
    │   ├─ Track capability confidence
    │   ├─ Estimate knowledge coverage
    │   └─ Assess uncertainty
    │
    ├─→ CURIOSITY
    │   ├─ Compute novelty
    │   ├─ Integrate uncertainty
    │   ├─ Calculate information gain
    │   └─ Generate intrinsic reward
    │
    ├─→ PLANNER
    │   ├─ Tree search simulation
    │   ├─ Evaluate trajectories
    │   └─ Select best action
    │
    └─→ ACTION
        │
        └─ Execute → ENVIRONMENT
```

---

## Data Flow (Single Step)

```
STEP N
├─ INPUT: env_observation, actual_next_state, reward
│
├─ PERCEIVE
│  └─ Substrate.add_observation()
│     └─ substrate.buffer += observation
│
├─ PREDICT  
│  └─ PredictionEngine.predict(current_state)
│     ├─ Get world_model transitions
│     ├─ Compute probabilities
│     └─ Return (expected_state, confidence, alternatives)
│
├─ EVALUATE
│  └─ PredictionEngine.evaluate(prediction, actual_state)
│     ├─ Compute magnitude = (expected != actual ? 1.0 : 0.0)
│     ├─ Compute surprise = 1 - P(actual)
│     └─ Return PredictionError
│
├─ WORLD MODEL UPDATE
│  └─ WorldModel.observe_transition(state, action, next_state)
│     ├─ Add/update edge weight
│     └─ Increment visit count
│
├─ ADAPT
│  └─ ModelAdapter.adapt_to_prediction_error(error, ...)
│     ├─ Adjust edge weights
│     ├─ Detect drift if error high
│     └─ Update uncertainty
│
├─ MEMORY
│  └─ Memory.store(state, action, next_state, reward, error)
│     ├─ EpisodicMemory.append()
│     └─ Check if should_consolidate()
│
├─ CONSOLIDATE (periodically)
│  └─ MemoryConsolidation.consolidate()
│     ├─ Extract patterns from episodic
│     ├─ Add to semantic memory
│     ├─ Forget low-confidence facts
│     └─ Update retention metrics
│
├─ SELF-MODEL
│  └─ SelfModel.update_metrics(...)
│     ├─ Track prediction_accuracy
│     ├─ Estimate knowledge
│     └─ Update exploration_rate
│
├─ DEVELOPMENT
│  └─ DevelopmentalController.record_stage_transition()
│     ├─ Update metrics
│     └─ Check advancement requirements
│
├─ ACTION SELECTION
│  └─ if self_model.should_explore()
│     ├─ action = CuriosityEngine.select_exploration_action()
│     └─ curiosity_reward = calculate_reward()
│
│  else
│     ├─ plan = Planner.plan(current_state, world_model)
│     └─ action = plan[0]
│
├─ METRICS (if enabled)
│  └─ MetricsCollector.record(...)
│     ├─ Store all performance indicators
│     └─ Compute trends
│
└─ OUTPUT: {step, state, action, reward, accuracy, drift_signal, ...}
```

---

## Memory Hierarchy

```
SENSORY INPUT
    ↓
WORKING MEMORY (current observation)
    ↓
EPISODIC MEMORY (recent experiences)
    ├─ Timestamped events
    ├─ Full state + action + outcome
    ├─ Limited capacity (FIFO)
    └─ Recent focus
    │
    ├─ CONSOLIDATION CYCLE (every N steps)
    │  └─ Pattern extraction
    │     ├─ Frequency analysis
    │     ├─ Transition probability
    │     └─ Causal relationships
    │
    ▼
SEMANTIC MEMORY (extracted knowledge)
    ├─ Abstracted facts
    ├─ Confidence scores
    ├─ Generalized patterns
    └─ Long-term retention
    │
    ├─ FORGETTING (decay low-confidence facts)
    │  └─ Confidence < threshold → remove
    │
    ▼
WORLD MODEL (integrated representation)
    ├─ Entity properties
    ├─ Relations
    ├─ Transition rules
    └─ Used for planning & prediction

PROCEDURAL MEMORY
    ├─ Action sequences (plans)
    ├─ Success statistics
    └─ Transfer learning
```

---

## Uncertainty Quantification

```
PREDICTION ENGINE
    ├─ Epistemic (what we don't know)
    │  └─ Reducible through more experience
    │
    ├─ Aleatoric (inherent randomness)
    │  └─ Cannot be reduced
    │
    ├─ Model Uncertainty
    │  └─ How good is our model?
    │
    └─ Prediction Uncertainty
       └─ Confidence in this specific forecast

USED FOR:
    ├─ Curiosity (explore high-uncertainty)
    ├─ Adaptation (high uncertainty → learn more)
    ├─ Planning (account for uncertainty in futures)
    └─ Self-Model (calibration assessment)
```

---

## Concept Drift Detection

```
STEADY STATE
├─ Predictions mostly correct
├─ Error distribution stable
└─ Learning rate normal

    ↓ ENVIRONMENT CHANGES

DRIFT DETECTION TRIGGERED
├─ Error spike detected
├─ Magnitude > threshold
├─ Surprise increases
├─ Multiple incorrect predictions

    ↓

ADAPTIVE RESPONSE
├─ Learning rate increases (higher plasticity)
├─ Confidence in old beliefs decreases
├─ Uncertainty increases (more exploration)
├─ Context-dependent marking of old knowledge

    ↓

RECOVERY
├─ New patterns learned
├─ Error decreases
├─ Confidence stabilizes
├─ Learning rate normalizes
```

---

## Developmental Stages

```
STAGE 0: BLANK
├─ No knowledge
├─ Random behavior
└─ Requirement: 100+ observations

    ↓

STAGE 1: PERCEPTION
├─ Sensory integration
├─ State recognition
└─ Requirement: ???

    ↓

STAGE 2: ASSOCIATION
├─ A predicts B
├─ Simple transitions
└─ Requirement: 500+ associations

    ↓

STAGE 3: PREDICTION
├─ Forecast consequences
├─ Multi-step reasoning
└─ Requirement: >70% accuracy

    ↓

STAGE 4: EXPLORATION  
├─ Curiosity-driven
├─ Strategic sampling
└─ Requirement: novelty > threshold

    ↓

STAGE 5: MEMORY
├─ Consolidation working
├─ Facts retained
└─ Requirement: retention > 70%

    ↓

STAGE 6: SELF-MODEL
├─ Metacognition active
├─ Knows capabilities
└─ Requirement: calibration > 70%

    ↓

STAGE 7: PLANNING
├─ Multi-step planning
├─ Tree search effective
└─ Requirement: planning_success > 80%

    ↓

STAGE 8: EMBODIMENT
├─ Full autonomy
├─ All systems integrated
└─ Final stage
```

---

## Key Parameters

### Learning
- `learning_rate` (0.001-1.0): How fast to update beliefs
- `stability_factor` (0-1): Weight on stability vs. plasticity
- `drift_threshold` (0-1): When to trigger adaptation

### Exploration
- `novelty_weight` (0-1): How much to value unexplored
- `uncertainty_weight` (0-1): How much to value uncertain
- `info_gain_weight` (0-1): How much to value informative

### Memory
- `episodic_limit` (buffer size)
- `consolidation_interval` (steps between consolidations)
- `forgetting_threshold` (minimum confidence to retain)

### Prediction
- `confidence_buckets` (calibration bins)
- `concept_drift_signal_threshold` (when to trigger drift response)

---

## Metrics Dashboard

### Real-time
- Current prediction accuracy
- Drift signal strength
- Episodic buffer utilization
- Semantic fact count

### Aggregated
- Learning curve (accuracy over time)
- Calibration error trend
- Concept drift detections
- Memory retention rate
- Recovery time after drift

### Diagnostic
- Plasticity vs. stability balance
- Novelty coverage
- Adaptation event rate
- Stage progression

---

## Integration Points

### Environment Interface
```python
class Environment:
    def observe() -> Observation
    def step(action) -> (state, reward, done)
```

### Custom Predictor
```python
class CustomPredictor:
    def predict(state, world_model) -> Prediction
    def evaluate(prediction, actual) -> Error
```

### Custom Adapter
```python
class CustomAdapter:
    def adapt(error, world_model) -> AdaptationEvent
    def handle_drift(old_state, new_state, world_model) -> None
```

### Custom Planner
```python
class CustomPlanner:
    def plan(state, world_model, goal) -> [actions]
```

---

## Reproducibility

All components seeded:
```python
import random
random.seed(42)

agent = TabulaRasaCore(seed=42)
env = SandboxEnvironment(seed=42)

# Multiple runs with same seed produce identical results
```

Checkpointing enables resume:
```python
agent.save_checkpoint("midway")
# later...
agent.load_checkpoint("midway")
agent.run_episode(env, max_steps=100)
```

---

## Assumptions & Limitations

### Assumptions
- Environment is Markovian (current state sufficient for prediction)
- Discrete state space (can be extended to continuous)
- Deterministic transitions (can handle stochastic with probability)
- Single agent (multi-agent future)

### Limitations
- No pretrained representations (learns from scratch)
- Limited to simple environments initially
- Memory is local (no distributed learning)
- Single-threaded execution
- No uncertainty quantification framework external to module

### Future Extensions
- Hierarchical world models
- Compositional representations
- Transfer learning between domains
- Multi-agent coordination
- Continuous state/action spaces

---

## Research Value

This architecture enables:

1. **Controlled Experiments**
   - Isolate individual components
   - Measure their contribution
   - Ablation studies

2. **Benchmarking**
   - Compare different learning algorithms
   - Measure against baselines
   - Track progress over time

3. **Reproducibility**
   - Full source code
   - Seeded randomness
   - Detailed logging

4. **Extensibility**
   - Modular design
   - Clear interfaces
   - Easy to add new components

5. **Understanding**
   - Trace decisions
   - Inspect internal states
   - Analyze learned models

---

## Implementation Quality

- ✅ Type hints throughout (Python 3.10+)
- ✅ Comprehensive docstrings
- ✅ Error handling
- ✅ Logging/debugging support
- ✅ Test coverage
- ✅ No external dependencies (stdlib only)
- ✅ Performance-conscious (caching, pruning)
- ✅ Memory-efficient (bounded buffers)

---

## Use Cases

### Academic
- Study continual learning in controlled settings
- Benchmark adaptation mechanisms
- Investigate world model formation

### Prototyping
- Test cognitive architecture ideas
- Validate algorithms before scaling
- Compare approaches quickly

### Education
- Teach cognitive science concepts
- Demonstrate autonomous learning
- Hands-on AI experiments

### Industry
- Baseline for autonomous systems
- Continual learning infrastructure
- Adaptation mechanism testing

---

**TR-Core: Research-grade implementation of autonomous developmental cognition**
