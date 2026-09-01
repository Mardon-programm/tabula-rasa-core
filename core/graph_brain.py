from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Synapse:
    source: str
    target: str
    weight: float = 0.1
    activations: int = 0
    age: int = 0


class DynamicSparseGraph:
    def __init__(
        self,
        *,
        learning_rate: float = 0.15,
        decay_rate: float = 0.01,
        pruning_threshold: float = 0.03,
    ) -> None:
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.pruning_threshold = pruning_threshold

        self._nodes: set[str] = set()
        self._edges: dict[tuple[str, str], Synapse] = {}

        self.steps = 0

    def add_node(self, node: str) -> None:
        self._nodes.add(node)

    def observe_transition(
        self,
        source: str,
        target: str,
    ) -> None:
        self.add_node(source)
        self.add_node(target)

        key = (source, target)

        if key not in self._edges:
            self._edges[key] = Synapse(
                source=source,
                target=target,
                weight=0.1,
            )

        synapse = self._edges[key]

        synapse.weight += self.learning_rate * (
            1.0 - synapse.weight
        )

        synapse.activations += 1
        synapse.age = 0

    def decay(self) -> None:
        self.steps += 1

        to_delete: list[tuple[str, str]] = []

        for key, synapse in self._edges.items():
            synapse.weight *= (1.0 - self.decay_rate)
            synapse.age += 1

            if synapse.weight < self.pruning_threshold:
                to_delete.append(key)

        for key in to_delete:
            del self._edges[key]

    def prune(self) -> int:
        to_delete = [
            key
            for key, synapse in self._edges.items()
            if synapse.weight < self.pruning_threshold
        ]

        for key in to_delete:
            del self._edges[key]

        return len(to_delete)

    def predict_next(
        self,
        source: str,
    ) -> str | None:
        candidates = [
            synapse
            for synapse in self._edges.values()
            if synapse.source == source
        ]

        if not candidates:
            return None

        candidates.sort(
            key=lambda synapse: synapse.weight,
            reverse=True,
        )

        return candidates[0].target

    def predict_chain(
        self,
        source: str,
        depth: int = 3,
    ) -> list[str]:
        chain = [source]
        current = source

        for _ in range(depth):
            prediction = self.predict_next(current)

            if prediction is None:
                break

            chain.append(prediction)
            current = prediction

        return chain

    def prediction_accuracy(
        self,
        source: str,
        actual_target: str,
    ) -> float:
        prediction = self.predict_next(source)

        if prediction is None:
            return 0.0

        return 1.0 if prediction == actual_target else 0.0

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def sparsity(self) -> float:
        possible = self.node_count * max(
            self.node_count - 1,
            1,
        )

        if possible == 0:
            return 1.0

        return 1.0 - (self.edge_count / possible)

    def edges(self) -> list[Synapse]:
        return list(self._edges.values())

    def snapshot(self) -> dict[str, Any]:
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "sparsity": self.sparsity,
            "steps": self.steps,
            "synapses": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": round(edge.weight, 4),
                    "activations": edge.activations,
                    "age": edge.age,
                }
                for edge in self._edges.values()
            ],
        }