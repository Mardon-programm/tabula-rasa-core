"""
Compositional-v1 baselines: value-classifiers over (context, action).

Phase 1 honest baselines. Every learner receives the SAME experience as the
behavioral-program agent: a sequence of trials ``(context, action, success)``
where ``action`` is uniform-random (explore) and ``success`` comes from a
deterministic oracle. No learner ever sees the oracle's labels directly.

A value-classifier learns ``P(success | context, action)``; at test time it
predicts ``argmax_a P(success | context, a)``. Unlike the original
DecisionList baseline, nothing here has oracle access, so the comparison with
the agent is fair.

All models are numpy-only (sklearn is not available in this environment).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from environment.attribute_tasks import (
    ACTIONS,
    ATTRIBUTE_DOMAINS,
    ALL_ATTRIBUTES,
    Context,
    is_success,
)


# ---------------------------------------------------------------------------
# Feature encoding: context (one-hot over attribute values) + action (one-hot).
# ---------------------------------------------------------------------------

class Encoder:
    """One-hot encoder over a schema (attributes, domains, actions)."""

    def __init__(
        self,
        attributes=None,
        domains=None,
        actions=None,
    ) -> None:
        if attributes is None:
            attributes = ALL_ATTRIBUTES
        if domains is None:
            domains = ATTRIBUTE_DOMAINS
        if actions is None:
            actions = ACTIONS
        self.actions = list(actions)
        self.attributes = list(attributes)
        value_index = {}
        offsets = {}
        ctx_dim = 0
        for attribute in self.attributes:
            offsets[attribute] = ctx_dim
            value_index[attribute] = {
                value: ctx_dim + i for i, value in enumerate(domains[attribute])
            }
            ctx_dim += len(domains[attribute])
        self._value_index = value_index
        self._ctx_dim = ctx_dim
        self._action_dim = len(self.actions)
        self.ctx_dim = ctx_dim
        self.action_dim = self._action_dim

    def context_vector(self, context) -> np.ndarray:
        x = np.zeros(self._ctx_dim, dtype=float)
        for attribute, value in context.items():
            x[self._value_index[attribute][value]] = 1.0
        return x

    def action_vector(self, action: str) -> np.ndarray:
        v = np.zeros(self._action_dim, dtype=float)
        v[self.actions.index(action)] = 1.0
        return v

    def features(self, context, action: str) -> np.ndarray:
        return np.concatenate([self.context_vector(context), self.action_vector(action)])

    def design_matrix(self, trials) -> tuple[np.ndarray, np.ndarray]:
        X = np.array(
            [self.features(ctx, action) for ctx, action, *_ in trials], dtype=float
        )
        y = np.array([1.0 if t[2] else 0.0 for t in trials], dtype=float)
        return X, y


# Default v0.1 encoder (keeps Phase 1 semantics unchanged).
def _default_encoder():
    global _DEFAULT_ENCODER
    return _DEFAULT_ENCODER


_DEFAULT_ENCODER = Encoder()


# ---------------------------------------------------------------------------
# Value Logistic Regression (binary, gradient descent, L2).
# ---------------------------------------------------------------------------


class ValueLogisticRegression:
    """P(success | context, action) as sigmoid(w·[ctx; act]) + b, GD."""

    name = "ValueLogReg"

    def __init__(
        self,
        *,
        seed: int = 0,
        lr: float = 0.5,
        epochs: int = 400,
        l2: float = 1e-3,
        encoder: Encoder | None = None,
    ) -> None:
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.enc = encoder or _DEFAULT_ENCODER
        self.w: np.ndarray | None = None
        self.b = 0.0

    def fit(self, trials: list[tuple[Context, str, bool]]) -> None:
        X, y = self.enc.design_matrix(trials)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w = rng.normal(0.0, 0.01, size=d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))
            grad_w = X.T @ (p - y) / n + self.l2 * self.w
            grad_b = np.mean(p - y)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def _p(self, x: np.ndarray) -> float:
        return 1.0 / (1.0 + np.exp(-(x @ self.w + self.b)))

    def predict(self, context: Context) -> str:
        scores = [
            self._p(self.enc.features(context, a)) for a in self.enc.actions
        ]
        return self.enc.actions[int(np.argmax(scores))]


# ---------------------------------------------------------------------------
# Value k-Nearest-Neighbours.
# ---------------------------------------------------------------------------


class ValueKNN:
    """P(success | context, action) by smoothed kNN over the joint space."""

    name = "ValuekNN"

    def __init__(self, *, k: int = 31, encoder: Encoder | None = None) -> None:
        self.k = k
        self.enc = encoder or _DEFAULT_ENCODER
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, trials: list[tuple[Context, str, bool]]) -> None:
        self.X, self.y = self.enc.design_matrix(trials)

    def predict(self, context: Context) -> str:
        best_action: str | None = None
        best_prob = -1.0
        for a in self.enc.actions:
            q = self.enc.features(context, a)
            dists = np.linalg.norm(self.X - q, axis=1)
            k = min(self.k, len(dists))
            idx = np.argpartition(dists, k - 1)[:k]
            prob = float(np.mean(self.y[idx]))
            if prob > best_prob:
                best_prob = prob
                best_action = a
        return best_action


# ---------------------------------------------------------------------------
# Value Decision Tree (CART-style, Gini, unscrambling into binary).
# ---------------------------------------------------------------------------


class _Tree:
    __slots__ = ("feature", "threshold", "left", "right", "prob")

    def __init__(self, prob: float):
        self.feature: int | None = None
        self.threshold: float = 0.0
        self.left: _Tree | None = None
        self.right: _Tree | None = None
        self.prob = prob


class ValueDecisionTree:
    """Binary CART (Gini) on X; predicts P(success) per leaf."""

    name = "ValueTree"

    def __init__(
        self,
        *,
        max_depth: int = 8,
        min_samples: int = 8,
        encoder: Encoder | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.enc = encoder or _DEFAULT_ENCODER
        self.root: _Tree | None = None

    def fit(self, trials: list[tuple[Context, str, bool]]) -> None:
        X, y = self.enc.design_matrix(trials)
        self.root = self._build(X, y, depth=0)

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = np.mean(y)
        return 1.0 - p * p - (1.0 - p) * (1.0 - p)

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Tree:
        node = _Tree(float(np.mean(y)))
        if depth >= self.max_depth or len(y) < self.min_samples:
            return node
        if float(np.unique(y).size) < 2:
            return node

        best_gain = 0.0
        best_f = None
        best_t = 0.0
        parent_imp = self._gini(y)
        for f in range(X.shape[1]):
            vals = np.unique(X[:, f])
            if vals.size < 2:
                continue
            for thr in vals[:-1]:
                mask = X[:, f] <= thr
                if mask.sum() < 1 or (~mask).sum() < 1:
                    continue
                imp = (
                    mask.sum() / len(y) * self._gini(y[mask])
                    + (~mask).sum() / len(y) * self._gini(y[~mask])
                )
                gain = parent_imp - imp
                if gain > best_gain:
                    best_gain = gain
                    best_f = f
                    best_t = float(thr)

        if best_f is None or best_gain <= 1e-9:
            return node

        node.feature = best_f
        node.threshold = best_t
        mask = X[:, best_f] <= best_t
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _prob(self, node: _Tree, x: np.ndarray) -> float:
        while node.feature is not None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prob

    def predict(self, context: Context) -> str:
        best_action: str | None = None
        best_prob = -1.0
        for a in self.enc.actions:
            q = self.enc.features(context, a)
            prob = self._prob(self.root, q)
            if prob > best_prob:
                best_prob = prob
                best_action = a
        return best_action


# ---------------------------------------------------------------------------
# Honest Decision-List: greedy rules over (attr=value, action) → success.
# ---------------------------------------------------------------------------


class ValueDecisionList:
    """Greedy cover of successful (attr=value, action) regions.

    Learned model: ordered list of ((attr, value, action), success-rate).
    Uses only the explore trials; no oracle labels.
    """

    name = "DList"

    def __init__(
        self,
        *,
        min_coverage: int = 3,
        ratio: float = 0.7,
        encoder: Encoder | None = None,
    ) -> None:
        self.min_coverage = min_coverage
        self.ratio = ratio
        self.enc = encoder or _DEFAULT_ENCODER
        self._items: list[tuple[str, str, str, float]] = []
        self._fallback: str = self.enc.actions[0]

    def fit(self, trials: list[tuple[Context, str, bool]]) -> None:
        counts: dict[tuple[str, str, str], int] = {}
        totals: dict[tuple[str, str, str], int] = {}
        for trial in trials:
            ctx, action, success = trial[0], trial[1], trial[2]
            for attribute in self.enc.attributes:
                key = (attribute, ctx[attribute], action)
                totals[key] = totals.get(key, 0) + 1
                if success:
                    counts[key] = counts.get(key, 0) + 1

        candidates = []
        for key, total in totals.items():
            if total < self.min_coverage:
                continue
            rate = counts.get(key, 0) / total
            candidates.append((*key, rate, total))

        # Greedy cover: pick highest (rate × cover), drop covered positives.
        candidates.sort(key=lambda t: (-t[3], -t[4]))
        positive = sum(1 for t in trials if t[2])
        covered = set()
        self._items = []
        for attr, value, action, rate, total in candidates:
            if rate < self.ratio:
                break
            added = 0
            for i, t in enumerate(trials):
                if i in covered or not t[2]:
                    continue
                if t[1] == action and t[0][attr] == value:
                    covered.add(i)
                    added += 1
                    if added >= total:
                        break
            self._items.append((attr, value, action, rate))
            if len(covered) >= positive:
                break

        if trials:
            from collections import Counter

            self._fallback = Counter(t[1] for t in trials).most_common(1)[0][0]

    def predict(self, context: Context) -> str:
        for attr, value, action, _ in self._items:
            if context.get(attr) == value:
                return action
        return self._fallback


# ---------------------------------------------------------------------------
# Compositional baselines (Phase 3 extension): learners that also search
# equality conjunctions, isolating whether BehavioralProgramV3's advantage
# comes from the composition *search* per se or from its selection scoring.
# All numpy-only, all on the identical uniform-trials experience.
# ---------------------------------------------------------------------------


def apriori_conjunction_candidates(
    trials,
    *,
    encoder: Encoder,
    max_arity: int = 4,
    min_coverage: int = 8,
):
    """Equality-conjunction candidates over the SAME hypothesis space as
    BehavioralProgramV3: Apriori-pruned literal sets of arity 1..max_arity,
    per action, with (coverage, hits) counted over trials that chose that
    action. Returns ``(action, literals, cov, hits)`` tuples (literals sorted).
    The agent additionally applies a binomial filter and precision >= 0.95;
    this function returns raw candidates so other scorers can select rules."""
    atoms = [
        (attr, value)
        for attr in encoder.attributes
        for value in encoder._value_index[attr]
    ]
    atom_ix = {a: i for i, a in enumerate(atoms)}
    per_action: dict[str, list[list]] = {}
    for trial in trials:
        ctx, act, success = trial[0], trial[1], bool(trial[2])
        mask = 0
        for attr, value in ctx.items():
            pos = atom_ix.get((attr, value))
            if pos is not None:
                mask |= 1 << pos
        lst = per_action.setdefault(act, [[], []])
        lst[0].append(mask)
        lst[1].append(success)
    out = []
    for action, (masks_list, hits_list) in per_action.items():
        if not masks_list:
            continue
        masks = np.asarray(masks_list, dtype=np.uint32)
        hits = np.asarray(hits_list, dtype=np.bool_)

        def stats(cond_mask: int) -> tuple[int, int]:
            sel = (masks & cond_mask) == cond_mask
            cov = int(sel.sum())
            return cov, int(hits[sel].sum()) if cov else 0

        supported_prev: list[frozenset[tuple[str, str]]] = []
        for k in range(1, max_arity + 1):
            if k == 1:
                candidates = [frozenset({atom}) for atom in atoms]
            else:
                built = set()
                for base in supported_prev:
                    for atom in atoms:
                        if atom in base:
                            continue
                        built.add(frozenset(base | {atom}))
                candidates = sorted(built, key=lambda cs: sorted(cs))
            if not candidates:
                break
            supported = []
            for cs in candidates:
                cond_mask = 0
                for atom in cs:
                    cond_mask |= 1 << atom_ix[atom]
                cov, n_hits = stats(cond_mask)
                if cov >= min_coverage:
                    supported.append(cs)
                    out.append((action, tuple(sorted(cs)), cov, n_hits))
            supported_prev = supported
            if not supported:
                break
    return out


class ValueCompoRuleLearner:
    """Same Apriori conjunction search, but rule selection by LIFT.

    Control for the induction *scoring* in BehavioralProgramV3: no precision
    threshold, no binomial p-value filter, no demotion — candidates are ranked
    by lift (precision / base success rate of the action) and the top-K per
    action are retained (suppressing redundant specialisations). Prediction is
    specificity-first exactly like the agent (tie-break by lift instead of
    evidence). Isolates search-cost from scoring-cost.
    """

    name = "CompoLift"

    def __init__(
        self,
        *,
        max_arity: int = 4,
        min_coverage: int = 8,
        min_lift: float = 2.0,
        min_hits: int = 5,
        max_rules_per_action: int = 40,
        encoder: Encoder | None = None,
    ) -> None:
        self.max_arity = max_arity
        self.min_coverage = min_coverage
        self.min_lift = min_lift
        self.min_hits = min_hits
        self.max_rules_per_action = max_rules_per_action
        self.enc = encoder or _DEFAULT_ENCODER
        self._rules: list[tuple[frozenset, str, float]] = []
        self._fallback: str = self.enc.actions[0]

    def fit(self, trials: list[tuple[Context, str, bool]]) -> None:
        from collections import Counter

        totals: dict[str, list[int]] = {}
        for trial in trials:
            key = trial[1]
            slot = totals.setdefault(key, [0, 0])
            slot[0] += 1
            slot[1] += 1 if trial[2] else 0

        self._rules = []
        for action in self.enc.actions:
            base_total, base_hits = totals.get(action, (0, 0))
            if base_total < 1 or base_hits < 1:
                continue
            base = base_hits / base_total
            pool = []
            for act, lits, cov, n_hits in apriori_conjunction_candidates(
                trials,
                encoder=self.enc,
                max_arity=self.max_arity,
                min_coverage=self.min_coverage,
            ):
                if act != action or n_hits < self.min_hits:
                    continue
                precision = n_hits / cov
                lift = (precision / base) if base > 0 else 0.0
                if lift >= self.min_lift:
                    pool.append((lits, precision, cov, lift))
            pool.sort(key=lambda x: (-x[3], -x[2]))
            sel = []
            for lits, _precision, _cov, lift in pool:
                ls = frozenset(lits)
                if any(ls.issuperset(prior) for prior, _a, _l in sel):
                    continue
                sel.append((ls, action, lift))
                if len(sel) >= self.max_rules_per_action:
                    break
            self._rules.extend(sel)

        if trials:
            self._fallback = Counter(t[1] for t in trials).most_common(1)[0][0]

    def predict(self, context: Context) -> str:
        best: str | None = None
        best_key = None
        for ls, action, lift in self._rules:
            if all(context.get(k) == v for k, v in ls):
                key = (-len(ls), -lift)
                if best_key is None or key < best_key:
                    best = action
                    best_key = key
        return best if best is not None else self._fallback


class ValueMultiDecisionList:
    """Decision list over equality conjunctions (arity up to max_arity).

    State-of-the-art-style symbolic rule learner (greedy cover as in
    CN2/SLIPPER): candidates from the shared Apriori search are ranked by
    (precision desc, coverage desc), the list is grown greedily, picking at
    each step the rule that covers the most still-uncovered positive trials,
    and prediction is first-match.
    """

    name = "DListML"

    def __init__(
        self,
        *,
        max_arity: int = 4,
        min_coverage: int = 8,
        ratio: float = 0.85,
        encoder: Encoder | None = None,
    ) -> None:
        self.max_arity = max_arity
        self.min_coverage = min_coverage
        self.ratio = ratio
        self.enc = encoder or _DEFAULT_ENCODER
        self._items: list[tuple[tuple, str, float]] = []
        self._fallback: str = self.enc.actions[0]

    def fit(self, trials: list[tuple[Context, str, bool]]) -> None:
        from collections import Counter

        cands = []
        for act, lits, cov, n_hits in apriori_conjunction_candidates(
            trials,
            encoder=self.enc,
            max_arity=self.max_arity,
            min_coverage=self.min_coverage,
        ):
            rate = n_hits / cov
            if rate >= self.ratio:
                cands.append((act, lits, rate, cov))
        cands.sort(key=lambda x: (-x[2], -x[3]))

        remaining = set(i for i, t in enumerate(trials) if t[2])
        self._items = []
        for act, lits, rate, _cov in cands:
            add = []
            for i in remaining:
                t = trials[i]
                if t[1] == act and all(t[0].get(k) == v for k, v in lits):
                    add.append(i)
            if not add:
                continue
            for i in add:
                remaining.discard(i)
            self._items.append((lits, act, rate))

        if trials:
            self._fallback = Counter(t[1] for t in trials).most_common(1)[0][0]

    def predict(self, context: Context) -> str:
        for lits, action, _rate in self._items:
            if all(context.get(k) == v for k, v in lits):
                return action
        return self._fallback


def _build_rf_tree(
    X: np.ndarray,
    y: np.ndarray,
    depth: int,
    *,
    rng,
    max_depth: int,
    min_samples: int,
    max_features: int,
) -> _Tree:
    """CART (Gini) with per-split random feature subsampling (Random Forest)."""
    node = _Tree(float(np.mean(y)))
    if depth >= max_depth or len(y) < min_samples or float(np.unique(y).size) < 2:
        return node
    feats = rng.choice(X.shape[1], size=max_features, replace=False)
    parent_imp = 1.0 - np.mean(y) ** 2 - (1 - np.mean(y)) ** 2
    best_gain = 0.0
    best_f = None
    best_t = 0.0
    for f in feats:
        vals = np.unique(X[:, f])
        if vals.size < 2:
            continue
        for thr in vals[:-1]:
            mask = X[:, f] <= thr
            if mask.sum() < 1 or (~mask).sum() < 1:
                continue
            l, r = y[mask], y[~mask]
            imp = (
                mask.sum() / len(y) * (1 - l.mean() ** 2 - (1 - l.mean()) ** 2)
                + (~mask).sum() / len(y) * (1 - r.mean() ** 2 - (1 - r.mean()) ** 2)
            )
            gain = parent_imp - imp
            if gain > best_gain:
                best_gain = gain
                best_f = f
                best_t = float(thr)
    if best_f is None or best_gain <= 1e-9:
        return node
    node.feature = best_f
    node.threshold = best_t
    mask = X[:, best_f] <= best_t
    node.left = _build_rf_tree(
        X[mask], y[mask], depth + 1, rng=rng,
        max_depth=max_depth, min_samples=min_samples, max_features=max_features,
    )
    node.right = _build_rf_tree(
        X[~mask], y[~mask], depth + 1, rng=rng,
        max_depth=max_depth, min_samples=min_samples, max_features=max_features,
    )
    return node


class ValueRandomForest:
    """Bagged CART forest on value features (pure-python, numpy only).

    The strongest non-symbolic controller that *can* learn conjunctions
    implicitly through tree splits — but has no explicit compositional
    induction step, so it tests whether explicit composition search adds
    anything over a powerful statistical learner.
    """

    name = "ValueForest"

    def __init__(
        self,
        *,
        n_estimators: int = 25,
        max_depth: int = 8,
        min_samples: int = 8,
        max_features: int | None = None,
        seed: int = 0,
        encoder: Encoder | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.seed = seed
        self.enc = encoder or _DEFAULT_ENCODER
        self._trees: list[_Tree] = []
        self._d = 0

    def fit(self, trials: list[tuple[Context, str, bool]]) -> None:
        X, y = self.enc.design_matrix(trials)
        rng = np.random.default_rng(self.seed)
        self._d = X.shape[1]
        mf = self.max_features or max(1, int(np.sqrt(self._d)))
        n = X.shape[0]
        self._trees = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            self._trees.append(
                _build_rf_tree(
                    X[idx], y[idx], 0, rng=rng,
                    max_depth=self.max_depth,
                    min_samples=self.min_samples,
                    max_features=mf,
                )
            )

    def _prob(self, node: _Tree, x: np.ndarray) -> float:
        while node.feature is not None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prob

    def predict(self, context: Context) -> str:
        best_action: str | None = None
        best_prob = -1.0
        for a in self.enc.actions:
            q = self.enc.features(context, a)
            acc = 0.0
            for tree in self._trees:
                acc += self._prob(tree, q)
            prob = acc / len(self._trees)
            if prob > best_prob:
                best_prob = prob
                best_action = a
        return best_action


# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------

from collections import defaultdict

def balanced_accuracy(predictor, contexts: list[Context], oracle=None) -> float:
    """Per-class average recall — fair when oracle classes are imbalanced."""
    if oracle is None:
        from environment.attribute_tasks import oracle_action as oracle
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ctx in contexts:
        true_a = oracle(ctx)
        pred_a = predictor.predict(ctx)
        per_class[true_a][1] += 1
        if pred_a == true_a:
            per_class[true_a][0] += 1
    if not per_class:
        return 0.0
    return sum(v[0] / v[1] for v in per_class.values()) / len(per_class)


def evaluate_predictor(
    model: Any,
    contexts: list[Context],
    success=None,
    oracle=None,
) -> dict[str, float]:
    if success is None:
        from environment.attribute_tasks import is_success as success
    correct = sum(1 for ctx in contexts if success(ctx, model.predict(ctx)))
    return {
        "overall": correct / max(len(contexts), 1),
        "balanced": balanced_accuracy(model, contexts, oracle),
    }


# ---------------------------------------------------------------------------
# Unbiased trial builder shared by agent and baselines.
# ---------------------------------------------------------------------------


def build_experience(
    train: list[Context],
    seed: int,
    actions: list[str] | None = None,
    success=None,
) -> list[tuple[Context, str, bool]]:
    """Uniform-random action per train context, oracle outcome.

    This is the single source of experience for every learner: the agent
    runs induction over these trials, each baseline fits on them.
    """
    from random import Random

    if actions is None:
        from environment.attribute_tasks import ACTIONS as actions
    if success is None:
        from environment.attribute_tasks import is_success as success

    rng = Random(seed)
    trials = []
    for ctx in train:
        action = actions[rng.randrange(len(actions))]
        trials.append((ctx, action, success(ctx, action)))
    return trials