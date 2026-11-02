# Papers with Code Submission Guide

## Submission URL
https://paperswithcode.com/paper/add

---

## Step 1: Paper Information

### Title
```
TR-Core: Autonomous Symbolic Induction of Compositional Rules via Apriori Search with Binomial Filtering
```

### Abstract
```
We present TR-Core, a formal experimentation framework for studying autonomous symbolic induction of compositional rules from experience. Unlike neural approaches requiring pretraining, TR-Core's behavioral program learns via seven development operations (CREATE, MODIFY, GENERALIZE, SPECIALIZE, COMPOSE, SPLIT, DELETE) over a fixed substrate. The core contribution is an Apriori-style induction engine with binomial statistical filtering that discovers rare compositional rules (conjunctions of 3-4 literals) and generalizes to structurally held-out contexts.

In Phase 3 (rare compositional rules, 8 attributes, 4608 contexts), TR-Core achieves balanced accuracy 0.78-0.80, significantly outperforming kNN (0.70), Random Forest (0.66), and Logistic Regression (0.20). Ablation studies isolate COMPOSE as the causal mechanism: disabling it collapses performance to chance (0.20). In Phase 4 (Structural-OOD generalization), TR-Core generalizes a learned pair rule (calm ∧ guard → approach) to a never-observed triple (calm ∧ guard ∧ night → approach) with 96% held-out accuracy, while decision lists (0.91) and decision trees (0.98) overfit to training conjunctions.

We provide formal complexity analysis: worst-case time O(L^K), expected time O((1+βL)^K) due to anti-monotonicity pruning, space O(N+L), with exact binomial test controlling per-candidate Type I error at α=10^-3. The combinatorial explosion is tamed by a coverage floor inducing a phase transition to subcritical branching.
```

### Authors
```
Mardon Iiminazhi Uulu
```

### Affiliation
```
Independent Researcher
```

### Paper URL
```
https://osf.io/ruvjg  (or arXiv link when available)
```

### Code URL
```
https://github.com/Mardon-programm/tabula-rasa-core
```

---

## Step 2: Tasks (add all applicable)

### Primary Tasks
1. **Compositional Generalization** — `paperswithcode.com/task/compositional-generalization`
2. **Symbolic Reasoning** — `paperswithcode.com/task/symbolic-reasoning`
3. **Rule Induction** — `paperswithcode.com/task/rule-induction`

### Secondary Tasks
4. **Out-of-Distribution Generalization** — `paperswithcode.com/task/out-of-distribution-generalization`
5. **Program Synthesis** — `paperswithcode.com/task/program-synthesis`

---

## Step 3: Methods

### Primary Methods
1. **Apriori Search** — level-wise candidate generation with anti-monotonicity pruning
2. **Binomial Filtering** — exact statistical test for rule significance
3. **Behavioral Program** — layer of changeable rules over fixed substrate

### Secondary Methods
4. **COMPOSE** — conjoining rules into subprograms
5. **GENERALIZE** — replacing constants with variables
6. **SPECIALIZE** — adding discriminant literals

---

## Step 4: Datasets

### Primary Dataset
- **Name**: Attribute Tasks V3/V4
- **Type**: Synthetic (controlled experimental)
- **Description**: Discrete attribute environments for testing compositional rule induction
- **URL**: https://www.kaggle.com/datasets/mardonprogramm/tr-core-compositional-generalization-experiments

---

## Step 5: Results Table

### Phase 3: Rare Compositional Rules (8 attrs, 4608 contexts)

| Model | Balanced Acc (Main) | Balanced Acc (Repl) |
|-------|---------------------|---------------------|
| TR-Core (full_bp) | 0.784 | 0.803 |
| ValueDListML | 0.813 | 0.766 |
| ValueDecisionTree | 0.752 | 0.763 |
| ValuekNN | 0.702 | 0.710 |
| ValueRandomForest | 0.659 | 0.671 |
| ValueCompoLift | 0.681 | 0.685 |
| ValueLogisticRegression | 0.200 | 0.200 |
| ValueDecisionList (arity-1) | 0.200 | 0.200 |
| null_bp | 0.200 | 0.202 |
| ab_no_compose | 0.200 | 0.200 |
| chance | 0.199 | 0.205 |
| ceiling | 1.000 | 1.000 |

### Phase 4: Structural-OOD Generalization

| Model | Held-out Acc (Main) | Held-out Acc (Repl) |
|-------|---------------------|---------------------|
| TR-Core (full_bp) | 0.958 | 1.000 |
| ValueDecisionTree | 0.977 | 0.985 |
| ValueDListML | 0.910 | 0.995 |
| ValueRandomForest | 0.844 | 0.935 |
| ValuekNN | 0.253 | 0.311 |
| ab_no_compose | 0.000 | 0.000 |
| null_bp | 0.000 | 0.000 |
| chance | 0.251 | 0.236 |
| ceiling | 1.000 | 1.000 |

---

## Step 6: Additional Metadata

### License
CC BY 4.0 (Creative Commons Attribution)

### Keywords
cognitive-architecture, neuro-symbolic, apriori, compositional-generalization, symbolic-induction, tabula-rasa, behavioral-program, binomial-filtering

### Code Repository
https://github.com/Mardon-programm/tabula-rasa-core

### Zenodo DOI
10.5281/zenodo.23619568

### OSF Preprint
https://osf.io/ruvjg

### Kaggle Dataset
https://www.kaggle.com/datasets/mardonprogramm/tr-core-compositional-generalization-experiments

---

## Step 7: After Submission

Once the paper is on Papers with Code, you will receive a link like:
`https://paperswithcode.com/paper/tr-core-autonomous-symbolic-induction-of`

**Send me that link** and I will:
1. Add a Papers with Code badge to README.md
2. Update docs/README.md with the link
3. Add the link to the paper's "Related Work" or "Resources" section

---

## README Badge (to add after submission)

```markdown
[![Papers with Code](https://img.shields.io/badge/Papers%20with%20Code-TR--Core-blue)](YOUR_PWC_LINK_HERE)
```

---

## Quick Copy-Paste Checklist

- [ ] Title copied
- [ ] Abstract copied
- [ ] Authors filled
- [ ] Paper URL filled (OSF/arXiv)
- [ ] Code URL filled (GitHub)
- [ ] Tasks selected (Compositional Generalization, Symbolic Reasoning, Rule Induction)
- [ ] Methods selected (Apriori Search, Binomial Filtering, Behavioral Program)
- [ ] Dataset linked (Kaggle)
- [ ] Results table added
- [ ] License selected (CC BY 4.0)
- [ ] Keywords added
- [ ] Submitted!
- [ ] Link sent to AI assistant for README update
