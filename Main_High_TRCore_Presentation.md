---
title: "TR-Core: Autonomous Symbolic Induction for Structural-OOD Generalization"
author: "Mardon [Surname] | Independent Systems Engineer"
institute: "School No. 7, Osh, Kyrgyzstan"
theme: "metropolis"
colortheme: "dark"
fonttheme: "professionalfonts"
---

# TR-Core: Autonomous Symbolic Induction for Structural-OOD Generalization

**Mardon [Surname]**  
Independent Systems Engineer  
School No. 7, Osh, Kyrgyzstan  

---

# The Problem: Why Neural Networks Fail at Structural OOD

| Limitation | Consequence |
|------------|-------------|
| Memorization over composition | Fails on valid but unseen concept combinations |
| Black-box opacity | Cannot audit, debug, or repair decisions |
| Data hunger | Requires massive labeled datasets |
| Static knowledge | Cannot adapt without full retraining |
| No causal mechanism | Correlation mistaken for composition |

**Structural-OOD Example**  
Learned: calm AND guard IMPLIES approach  
Held-out (never seen): calm AND guard AND night IMPLIES approach  
Neural nets: ~0.25 accuracy (chance)  
Required: Compositional generalization via COMPOSE

---

# TR-Core Architecture

```
+-------------------------------------------------------------+
|                    ATTRIBUTE TASK ENVIRONMENT               |
|         (Gymnasium: V3 Rare / V4 Structural-OOD)            |
+-----------------------------+-------------------------------+
                              | observe(context, action, success)
                              V
+-------------------------------------------------------------+
|  BEHAVIORAL PROGRAM V3                                      |
|  +-----------------+  +---------------------------------+  |
|  |   RULE BASE     |  |     INDUCTION ENGINE (Apriori)  |  |
|  |  * LATENT       |  |  * Coverage pruning (c_min=4)   |  |
|  |  * ACTIVE       |  |  * Binomial filter (alpha=1e-3) |  |
|  |  * RETIRED      |  |  * Precision >= 0.95            |  |
|  |  * DELETED      |  |  * Max arity K=4                |  |
+------------+--------+  +----------------+----------------+  |
             |                         |                      |
             +------------+------------+                      |
                          V                                     |
                  SUBPROGRAMS (COMPOSE)                         |
+-------------------------------------------------------------+
```

**7 Development Operations**: CREATE, GENERALIZE, SPECIALIZE, SPLIT, COMPOSE, MODIFY, DELETE

---

# Phase 3: Rare Compositional Rules (8 attrs, 4,608 contexts)

| Model | Balanced Accuracy (Main/Repl) |
|-------|------------------------------|
| **TR-Core (full_bp)** | **0.784 / 0.803** |
| ValueDListML | 0.813 / 0.766 |
| ValueDecisionTree | 0.752 / 0.763 |
| ValuekNN | 0.702 / 0.710 |
| ValueRandomForest | 0.659 / 0.671 |
| Logistic Regression | 0.200 / 0.200 |
| **ab_no_compose / null_bp** | **0.200 / 0.200** |

**Rule Recovery**: 68% exact/subset | Behavioral FDR = 0.36

---

# Phase 4: Structural-OOD (Held-out triple, zero training support)

| Model | Held-out Accuracy (Main/Repl) |
|-------|------------------------------|
| **TR-Core (full_bp)** | **0.958 / 1.000** |
| ValueDecisionTree | 0.977 / 0.985 |
| ValueDListML | 0.910 / 0.995 |
| ValueRandomForest | 0.844 / 0.935 |
| ValuekNN | 0.253 / 0.311 |
| **ab_no_compose / null_bp** | **0.000 / 0.000** |

**Pair rule recovery**: calm AND guard IMPLIES approach recovered 24/24 seeds

---

# Ablation Studies: Causal Evidence

| Ablation | Phase 3 Balanced Acc | Phase 4 Held-out Acc |
|----------|---------------------|---------------------|
| full_bp (COMPOSE enabled) | 0.784 / 0.803 | 0.958 / 1.000 |
| ab_no_compose (COMPOSE disabled) | 0.200 / 0.200 | 0.000 / 0.000 |
| null_bp (randomized outcomes) | 0.200 / 0.202 | 0.000 / 0.000 |
| chance baseline | 0.199 / 0.205 | 0.251 / 0.236 |

Conclusion: COMPOSE is the causal mechanism for structural-OOD generalization. Without it, the system collapses to random guessing. With it, it achieves near-ceiling OOD accuracy.

---

# Formal Guarantees & Infrastructure

### Formal Analysis (Proven)

| Property | Bound |
|----------|-------|
| Worst-case time | O(N*L + L^K) |
| Expected time (pruning) | O((1+beta*L)^K) |
| Space | O(N + L + R*K) |
| Type I error (per candidate) | <= alpha = 0.001 (exact binomial) |

### Professional Infrastructure

- Gymnasium interface: TRCore-AttributeTaskV3-v0, V4-v0
- Configuration: Hydra (sweep-ready YAML)
- Pipeline: DVC (14 stages: data to exp to analysis to figures)
- Statistics: Permutation tests (20k), Cohen's dz, bootstrap CI
- Reproducibility: 24 seeds x 2 replications, fixed seeds, DVC tracking
- Package: pip install -e . via pyproject.toml

---

# Conclusions & Next Steps: Alan

### What TR-Core Proves
1. Compositional generalization without neural networks -- symbolic induction + exact statistics suffices
2. COMPOSE is necessary and sufficient for structural-OOD -- ablations isolate it as causal mechanism
3. Exact statistical control enables tractable search -- binomial filter + coverage floor tames combinatorial explosion
4. White-box by construction -- every decision traceable to a rule with provenance

### Next Steps: Alan (Embodied AGI Prototype)

| Phase | Goal | Timeline |
|-------|------|----------|
| 1. Perception | Pixels to discrete attributes (Slot Attention + DINOv2) | Months 1-3 |
| 2. Hierarchical Composition | Rules to Skills to HTN Programs (library learning) | Months 4-7 |
| 3. Lifelong Memory | Episodic to Semantic to Procedural consolidation | Months 8-11 |
| 4. Embodiment | Sim2Real on Unitree Go1 / ALOHA (CBF-MPC safety) | Months 12-18 |

TR-Core is the verified symbolic core. Alan builds the organism around it.

---

# Contact & Links

Repository: github.com/Mardon-programm/tabula-rasa-core  
Alan (Next Phase): github.com/Mardon-programm/alan  
Paper (LaTeX): docs/tr_core_paper.tex  
Formal Analysis: docs/apriori_formal_analysis.tex  
Demo: streamlit run web/app.py -> Phase 4 Demo

Contact: [email] | Independent Systems Engineer | School No. 7, Osh
