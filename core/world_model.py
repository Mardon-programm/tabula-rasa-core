from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Entity:
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    capability: float = 0.5  
    confidence: float = 0.5  


@dataclass(slots=True)
class Relation:
    source: str
    target: str
    relation_type: str  
    strength: float = 0.5
    observations: int = 0


@dataclass(slots=True)
class Transition:
    from_state: str
    to_state: str
    action: str
    precondition: str | None = None
    consequence: str | None = None
    probability: float = 0.5
    observations: int = 0


@dataclass(slots=True)
class Uncertainty:
    epistemic: float = 0.5  
    aleatoric: float = 0.2  
    model_uncertainty: float = 0.3  
    prediction_uncertainty: float = 0.4
    action_uncertainty: float = 0.3


class WorldModel:

    def __init__(self, *, recency: float = 0.9) -> None:
        self.recency = recency
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._transitions: dict[tuple[str, str, str], Transition] = {}
        
        # Causal model
        self._causal_links: dict[str, set[str]] = {}
        
        # Uncertainty tracking
        self._uncertainty: dict[str, Uncertainty] = {}
        
        # Observation statistics
        self._state_visits: dict[str, int] = {}

        # Recency-weighted transition weights: keyed by (from, action, to).
        # Decays on each observation of the parent (from, action) so that
        # superseded transitions are gradually forgotten after concept drift.
        self._transition_weights: dict[tuple[str, str, str], float] = {}
        self._transition_counts: dict[tuple[str, str], int] = {}

    def add_entity(
        self,
        name: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        
        if name not in self._entities:
            self._entities[name] = Entity(
                name=name,
                properties=properties or {},
            )
        else:
            if properties:
                self._entities[name].properties.update(properties)

    def observe_transition(
        self,
        from_state: str,
        action: str,
        to_state: str,
    ) -> None:

        # Add states as entities
        self.add_entity(from_state)
        self.add_entity(to_state)
        
        self._state_visits[from_state] = self._state_visits.get(from_state, 0) + 1
        self._state_visits[to_state] = self._state_visits.get(to_state, 0) + 1
        
        key = (from_state, action, to_state)
        if key not in self._transitions:
            self._transitions[key] = Transition(
                from_state=from_state,
                to_state=to_state,
                action=action,
                probability=0.5,
            )
        
        transition = self._transitions[key]
        transition.observations += 1
        
        # Recency-weighted probabilities: decay competing transitions under the
        # same (from_state, action), then boost the observed target. This lets
        # the transition model forget superseded rules after concept drift.
        for (fs, ac, ts), weight in list(self._transition_weights.items()):
            if fs == from_state and ac == action:
                self._transition_weights[(fs, ac, ts)] = weight * self.recency
        self._transition_weights[key] = self._transition_weights.get(key, 0.0) + 1.0

        total_weight = sum(
            w
            for (fs, ac, _), w in self._transition_weights.items()
            if fs == from_state and ac == action
        ) + 1e-6

        for (fs, ac, ts), weight in self._transition_weights.items():
            if fs == from_state and ac == action and (fs, ac, ts) in self._transitions:
                self._transitions[(fs, ac, ts)].probability = weight / total_weight
        
        if from_state not in self._causal_links:
            self._causal_links[from_state] = set()
        self._causal_links[from_state].add(to_state)

    def predict_next_state(
        self,
        current_state: str,
        action: str,
    ) -> tuple[str, float] | None:

        best_match = None
        best_prob = 0.0
        
        for (from_s, act, to_s), transition in self._transitions.items():
            if from_s == current_state and act == action:
                if transition.probability > best_prob:
                    best_prob = transition.probability
                    best_match = to_s
        
        if best_match:
            return (best_match, best_prob)
        
        return None

    def get_possible_transitions(
        self,
        current_state: str,
    ) -> dict[str, dict[str, float]]:

        result: dict[str, dict[str, float]] = {}
        
        for (from_s, act, to_s), transition in self._transitions.items():
            if from_s == current_state:
                if act not in result:
                    result[act] = {}
                result[act][to_s] = transition.probability
        
        return result

    def get_uncertainty(self, state: str) -> Uncertainty:
        """Get uncertainty estimates for a state."""
        if state not in self._uncertainty:
            visits = self._state_visits.get(state, 0)
            
            epistemic = 1.0 / (1.0 + visits * 0.1)
            aleatoric = 0.2 
            model_uncertainty = 0.3 + 0.2 * epistemic
            
            self._uncertainty[state] = Uncertainty(
                epistemic=epistemic,
                aleatoric=aleatoric,
                model_uncertainty=model_uncertainty,
                prediction_uncertainty=epistemic + aleatoric,
                action_uncertainty=epistemic,
            )
        
        return self._uncertainty[state]

    def update_uncertainty(
        self,
        state: str,
        prediction_error: float,
    ) -> None:
        unc = self.get_uncertainty(state)
        
        unc.epistemic = min(1.0, unc.epistemic + 0.1 * prediction_error)
        unc.model_uncertainty = min(1.0, unc.model_uncertainty + 0.05 * prediction_error)

    def get_statistics(self) -> dict[str, Any]:
        return {
            "entities": len(self._entities),
            "relations": len(self._relations),
            "transitions": len(self._transitions),
            "state_visits": dict(self._state_visits),
            "causal_links": {k: len(v) for k, v in self._causal_links.items()},
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "entities": {
                name: {
                    "properties": entity.properties,
                    "capability": entity.capability,
                    "confidence": entity.confidence,
                }
                for name, entity in self._entities.items()
            },
            "transitions": {
                f"{k[0]}-{k[1]}-{k[2]}": {
                    "probability": t.probability,
                    "observations": t.observations,
                }
                for k, t in self._transitions.items()
            },
            "state_visits": self._state_visits,
        }
