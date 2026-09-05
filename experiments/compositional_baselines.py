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

_VALUE_INDEX: dict[str, dict[str, int]] = {}
_OFFSETS: dict[str, int] = {}
_ctx_dim = 0
for attribute in ALL_ATTRIBUTES:
    _OFFSETS[attribute] = _ctx_dim
    _VALUE_INDEX[attribute] = {
        value: _ctx_dim + i for i, value in enumerate(ATTRIBUTE_DOMAINS[attribute])
    }
    _ctx_dim += len(ATTRIBUTE_DOMAINS[attribute])
ACTION_DIM = len(ACTIONS)


def context_vector(context: Context) -> np.ndarray:
    x = np.zeros(_ctx_dim, dtype=float)
    for attribute, value in context.items():
        x[_VALUE_INDEX[attribute][value]] = 1.0
    return x


def action_vector(action: str) -> np.ndarray:
    v = np.zeros(ACTION_DIM, dtype=float)
    v[ACTIONS.index(action)] = 1.0
    return v


def features(context: Context, action: str) -> np.ndarray:
    return np.concatenate([context_vector(context), action_vector(action)])


def design_matrix(trials: list[tuple[Context, str, bool]]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [features(ctx, action) for ctx, action, _ in trials], dtype=float
    )
    y = np.array([1.0 if s else 0.0 for _, _, s in trials], dtype=float)
    return X, y


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
    ) -> None:
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w: np.ndarray | None = None
        self.b = 0.0

    def fit(self, trials: list[tuple[Context, str, bool]]) -> None:
        X, y = design_matrix(trials)
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
            self._p(features(context, a)) for a in ACTIONS
        ]
        return ACTIONS[int(np.argmax(scores))]


# ---------------------------------------------------------------------------
# Value k-Nearest-Neighbours.
# ---------------------------------------------------------------------------


class ValueKNN:
    """P(success | context, action) by smoothed kNN over the joint space."""

    name = "ValuekNN"

    def __init__(self, *, k: int = 31) -> None:
        self.k = k
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, trials: list[tuple[Context, str, bool]]) -> None:
        self.X, self.y = design_matrix(trials)

    def predict(self, context: Context) -> str:
        best_action: str | None = None
        best_prob = -1.0
        for a in ACTIONS:
            q = features(context, a)
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

    def __init__(self, *, max_depth: int = 8, min_samples: int = 8) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root: _Tree | None = None

    def fit(self, trials: list[tuple[Context, str, bool]]) -> None:
        X, y = design_matrix(trials)
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
        for a in ACTIONS:
            q = features(context, a)
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

    def __init__(self, *, min_coverage: int = 3, ratio: float = 0.7) -> None:
        self.min_coverage = min_coverage
        self.ratio = ratio
        self._items: list[tuple[str, str, str, float]] = []
        self._fallback: str = ACTIONS[0]

    def fit(self, trials: list[tuple[Context, str, bool]]) -> None:
        counts: dict[tuple[str, str, str], int] = {}
        totals: dict[tuple[str, str, str], int] = {}
        for ctx, action, success in trials:
            for attribute in ALL_ATTRIBUTES:
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
        positive = sum(1 for _, _, s in trials if s)
        covered = set()
        self._items = []
        for attr, value, action, rate, total in candidates:
            if rate < self.ratio:
                break
            added = 0
            for i, (ctx, a, s) in enumerate(trials):
                if i in covered or not s:
                    continue
                if a == action and ctx[attr] == value:
                    covered.add(i)
                    added += 1
                    if added >= total:
                        break
            self._items.append((attr, value, action, rate))
            if len(covered) >= positive:
                break

        if trials:
            from collections import Counter

            self._fallback = Counter(a for _, a, _ in trials).most_common(1)[0][0]

    def predict(self, context: Context) -> str:
        for attr, value, action, _ in self._items:
            if context.get(attr) == value:
                return action
        return self._fallback


# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------

from collections import defaultdict

def balanced_accuracy(predictor, contexts: list[Context]) -> float:
    """Per-class average recall — fair when oracle classes are imbalanced."""
    from environment.attribute_tasks import oracle_action as _oracle
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ctx in contexts:
        true_a = _oracle(ctx)
        pred_a = predictor.predict(ctx)
        per_class[true_a][1] += 1
        if pred_a == true_a:
            per_class[true_a][0] += 1
    if not per_class:
        return 0.0
    return sum(v[0] / v[1] for v in per_class.values()) / len(per_class)


def evaluate_predictor(model: Any, contexts: list[Context]) -> dict[str, float]:
    correct = sum(1 for ctx in contexts if is_success(ctx, model.predict(ctx)))
    return {
        "overall": correct / max(len(contexts), 1),
        "balanced": balanced_accuracy(model, contexts),
    }


# ---------------------------------------------------------------------------
# Unbiased trial builder shared by agent and baselines.
# ---------------------------------------------------------------------------


def build_experience(
    train: list[Context],
    seed: int,
) -> list[tuple[Context, str, bool]]:
    """Uniform-random action per train context, oracle outcome.

    This is the single source of experience for every learner: the agent
    runs induction over these trials, each baseline fits on them.
    """
    from random import Random

    rng = Random(seed)
    trials = []
    for ctx in train:
        action = ACTIONS[rng.randrange(len(ACTIONS))]
        trials.append((ctx, action, is_success(ctx, action)))
    return trials