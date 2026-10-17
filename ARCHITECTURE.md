# TR-Core Architecture: Behavioral Program with Apriori Induction

**Version:** v0.3 (Phase 3/4) — *Behavioral Program over Fixed Substrate*

---

## System Philosophy

TR-Core is a **formal experimentation framework** for studying autonomous symbolic induction of compositional rules from experience. It implements a developmental cognitive architecture where a **behavioral program** (layer of changeable rules over a fixed substrate) learns via seven development operations.

**Key contribution:** An Apriori-style induction engine with binomial statistical filtering that discovers rare compositional rules (conjunctions of 3–4 literals) and generalizes to structurally held-out contexts.

---

## Core Architecture

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

---

## Behavioral Program (v0.2 Spec)

### Rule Structure

```python
Rule:
  condition: Condition      # Conjunction of Literals (AND)
  action: Action            # Primitive or Subprogram
  status: LATENT | ACTIVE | RETIRED | DELETED
  evidence: Evidence        # (n_success, n_failure)
  provenance: list[str]     # Operations that created/modified
```

### Literal Types

| Operator | Meaning | Example |
|----------|---------|---------|
| `=` | Equality | `role = adult` |
| `≠` | Inequality | `status ≠ stranger` |
| `∈` | Set membership | `season ∈ {spring, autumn}` |

### Seven Development Operations

| Operation | Description | Precondition |
|-----------|-------------|--------------|
| **CREATE** | New LATENT rule from successful interaction | No covering rule exists |
| **GENERALIZE** | Replace constants with variable (`∈` set) | Multiple rules share action, validated on train |
| **SPECIALIZE** | Add discriminant literal, split parent | Discriminating feature found (separation score) |
| **SPLIT** | Partition rule along attribute values | Multiple discriminant values |
| **COMPOSE** | Conjoin two rules into subprogram | Both rules active, compatible |
| **MODIFY** | Change condition or action | Direct edit |
| **DELETE** | Retire rule as last resort | No other option |

### Revision Policy (Spec v0.2 §4)

1. **One failure ≠ wrong rule** — evidence accumulated first
2. **Failure threshold exceeded** → context analysis finds discriminating feature → SPECIALIZE or SPLIT
3. **Otherwise** → GENERALIZE on train-validated hypothesis
4. **DELETE** only as last resort

---

## Apriori Induction Engine (Phase 3/4)

### Algorithm

```python
def develop(self) -> int:
    explore = [t for t in self.trials if t[3]]  # uniform-random trials only
    for action in actions:
        # Level k = 1: all single literals
        # Level k > 1: extend supported (k-1)-conjunctions (Apriori join)
        for k in 1..K:
            for candidate in candidates_k:
                cov = coverage(candidate, explore, action)
                if cov < min_coverage: continue  # Apriori pruning
                hits = success_count(candidate, explore, action)
                prec = hits / cov
                if hits >= min_hits and prec >= theta:
                    pval = binom_sf(hits-1, cov, base_rate)
                    if pval <= alpha:
                        materialize(candidate, action, op=COMPOSE if k>=2 else GENERALIZE)
```

### Anti-Monotonicity Pruning

**Theorem:** If `cov(C) < min_coverage`, then for all `C' ⊃ C`, `cov(C') < min_coverage`.

This induces a **phase transition**: with `min_coverage=4`, effective branching factor `b_eff(k) < 1` for `k ≥ 3`, making search tractable (`O((1+βL)^K)` expected vs `O(L^K)` worst-case).

### Binomial Statistical Filter

For candidate `C` with `cov=n`, `hits=h`, base rate `p_a`:

```
p-value = P(X ≥ h) where X ~ Binomial(n, p_a)
Accept iff: p-value ≤ α (1e-3) AND precision ≥ θ (0.95) AND hits ≥ h_min (5)
```

**Guarantees:** Exact per-candidate Type I error ≤ α. Controls false discovery from sparse conjunctions.

---

## Environments

### Phase 3: Rare Compositional Rules (V3)

| Property | Value |
|----------|-------|
| Attributes | 8 (role, friendliness, status, mood, context, season, location, time) |
| Contexts | 4608 |
| Actions | 5 (approach, avoid, ask, obey, offer) |
| Oracle | Priority list: 4 rare rules (arity 3–4) + 3 simple + fallback |
| Key property | Single literals ambiguous (precision 0.16–0.68); only conjunctions disambiguate |

### Phase 4: Structural-OOD Generalization (V4)

| Property | Value |
|----------|-------|
| Attributes | 6 (agent, emotion, task, terrain, weather, hour) |
| Contexts | 864 |
| Actions | 4 (approach, avoid, ask, obey) |
| Held-out | `calm ∧ guard ∧ night → approach` (zero train support) |
| Train rule | `calm ∧ guard → approach` (pair covers held-out triple) |
| Test | Generalization from pair to never-observed triple |

---

## Data Flow (Single Interaction)

```
STEP
├─ INPUT: context (dict), chosen_action, success (bool), explored (bool)
│
├─ OBSERVE
│  └─ trials.append((context, action, success, explored))
│
├─ MATCH
│  └─ active_rules = [r for r in rules if r.condition.matches(context)]
│
├─ ACT
│  └─ If no active rule: explore with probability explore_rate
│
├─ UPDATE EVIDENCE
│  └─ fired_rule.evidence.record(success)
│
├─ REVISE (if failure)
│  ├─ If failure threshold exceeded:
│  │  ├─ Find discriminating feature (separation score)
│  │  └─ SPECIALIZE or SPLIT
│  └─ Else: GENERALIZE or DELETE
│
└─ DEVELOP (periodically, every develop_interval steps)
   └─ Apriori induction on explore trials → materialize new rules
```

---

## Experimental Infrastructure

### Hydra Configuration

```
conf/
├── config.yaml                    # Main (defaults, output dirs)
├── experiment/phase3.yaml         # Phase 3 params
├── experiment/phase4.yaml         # Phase 4 params
├── agent/behavioral_program_v3.yaml  # Induction params
├── env/attribute_tasks_v3.yaml    # V3 schema
└── env/attribute_tasks_v4.yaml    # V4 schema
```

### DVC Pipeline (14 stages)

```
generate_data_v3 → run_phase3 → run_ablations_phase3 → run_null_phase3 → sweep_* → analyze_phase3 → generate_figures
generate_data_v4 → run_phase4 → run_ablations_phase4 → run_null_phase4 → sweep_* → analyze_phase4 → generate_figures
```

### Statistical Analysis

- **Paired permutation tests** (20k permutations)
- **Cohen's dz** effect sizes
- **Bootstrap 95% CI** (2000 resamples)
- **Mechanistic controls:** `null_bp` (random outcomes), `ab_no_compose` (COMPOSE disabled)

### Gymnasium Interface

```python
import gymnasium as gym
from environment.gymnasium_env import register_envs

register_envs()
env = gym.make("TRCore-AttributeTaskV4-v0", seed=42)
# Dict(Discrete) observation, Discrete action space
```

---

## Key Results Summary

### Phase 3 (Rare Compositional, 24 seeds × 2 replications)

| Model | Main Balanced Acc | Replication Balanced Acc |
|-------|-------------------|--------------------------|
| **full_bp** | **0.784** | **0.803** |
| ValueDListML | 0.813 | 0.766 |
| ValueTree | 0.752 | 0.763 |
| ValuekNN | 0.702 | 0.710 |
| ValueRF | 0.659 | 0.671 |
| **ab_no_compose / null_bp** | **0.200** | **0.200** |

- COMPOSE necessary (ablations → chance)
- Rule recovery: 68% exact/subset, behavioral FDR = 0.36

### Phase 4 (Structural-OOD, 24 seeds × 2 replications)

| Model | Main Heldout | Replication Heldout |
|-------|--------------|---------------------|
| **full_bp** | **0.958** | **1.000** |
| ValueTree | 0.977 | 0.985 |
| ValueDListML | 0.910 | 0.995 |
| ValuekNN | 0.253 | 0.311 |
| **ab_no_compose / null_bp** | **0.000** | **0.000** |

- Pair rule `calm∧guard→approach` recovered 24/24 seeds
- COMPOSE = causal mechanism for OOD generalization

---

## Formal Guarantees (Appendix)

| Property | Bound |
|----------|-------|
| Worst-case time | `O(N·L + L^K)` |
| Expected time (pruning) | `O((1+βL)^K)` |
| Space | `O(N + L + R·K)` |
| Per-candidate Type I error | `≤ α = 1e-3` |
| Ground-truth recovery | If coverage ≥ c_min, precision ≥ θ+δ, p-value ≤ α/2 |

Full proofs: `docs/apriori_formal_analysis.tex`

---

## Reproducibility

```bash
# Full pipeline
dvc repro

# Single experiment
python -m experiments.run --experiment phase4 --seeds 1-24 --summarize --paired-stats

# Parameter sweep
python -c "
from experiments.harness import BenchmarkHarness
h = BenchmarkHarness()
r = h.run_sweep('phase4', {'agent.max_arity': [2,3,4]}, seeds=range(1,6))
print(h.summarize(r))
"
```

All components seeded. Checkpointing via `persistence/`.

---

## Limitations & Future Work

| Limitation | Status |
|------------|--------|
| Binomial filter weak for low-sample rules | Known (warm/stranger/spring/dawn: 13/24 seeds) |
| Arity ≥ 5 and `≠` literals not supported | Planned |
| Neural-symbolic baselines (GNN, NTP) not included | Can be added to `compositional_baselines.py` |
| Only attribute-task environments | Transfer to POMDP/continuous next |

---

## Project Structure (Current v0.3)

```
tr-core/
├── behavioral/              # Behavioral Program (core)
│   ├── program.py          # Base (v0.2 spec)
│   ├── program_v3.py       # Apriori induction (Phase 3/4)
│   └── rules.py            # Rule, Condition, Literal, Evidence
├── environment/             # Task environments
│   ├── attribute_tasks_v3.py
│   ├── attribute_tasks_v4.py
│   └── gymnasium_env.py    # Gymnasium wrapper
├── experiments/             # Experimentation framework
│   ├── experiment_phase3.py
│   ├── experiment_phase4.py
│   ├── compositional_baselines.py
│   ├── harness.py          # BenchmarkHarness
│   ├── run.py              # CLI runner
│   ├── analyze_results.py  # Stats
│   └── generate_figures.py
├── conf/                    # Hydra configs
├── docs/
│   ├── apriori_formal_analysis.tex
│   ├── tr_core_paper.tex
│   └── references.bib
├── dvc.yaml                 # Pipeline
├── pyproject.toml           # Package
└── tests/
```

---

## Citation

```bibtex
@article{trcore2026,
  title={TR-Core: Autonomous Symbolic Induction of Compositional Rules via Apriori Search with Binomial Filtering},
  author={TR-Core Research Group},
  year={2026},
  note={arXiv preprint}
}
```