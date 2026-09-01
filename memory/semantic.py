from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SemanticFact:

    subject: str
    relation: str
    object: str
    
    confidence: float = 0.5
    observations: int = 0
    first_seen: int = 0
    last_seen: int = 0
    
    stability: float = 0.5 
    uncertainty: float = 0.5  


class SemanticMemory:


    def __init__(self) -> None:
        self._facts: dict[tuple[str, str, str], SemanticFact] = {}
        self._step_count = 0

    def record_transition(
        self,
        subject: str,
        relation: str,
        object: str,
    ) -> None:
        key = (subject, relation, object)
        
        if key not in self._facts:
            self._facts[key] = SemanticFact(
                subject=subject,
                relation=relation,
                object=object,
                first_seen=self._step_count,
            )
        
        fact = self._facts[key]
        fact.observations += 1
        fact.last_seen = self._step_count
        
        fact.confidence = min(
            1.0,
            0.1 + (fact.observations * 0.1),
        )
        
        fact.stability = min(
            1.0,
            fact.observations / 10.0,
        )

    def query(
        self,
        subject: str,
        relation: str | None = None,
    ) -> list[SemanticFact]:
        results = []
        
        for (s, r, o), fact in self._facts.items():
            if s == subject:
                if relation is None or r == relation:
                    results.append(fact)
        
        return results

    def get_fact(
        self,
        subject: str,
        relation: str,
        object: str,
    ) -> SemanticFact | None:
        key = (subject, relation, object)
        return self._facts.get(key)

    def most_confident_facts(
        self,
        limit: int = 20,
    ) -> list[SemanticFact]:
        sorted_facts = sorted(
            self._facts.values(),
            key=lambda f: f.confidence,
            reverse=True,
        )
        return sorted_facts[:limit]

    def weakest_facts(
        self,
        limit: int = 20,
    ) -> list[SemanticFact]:
        sorted_facts = sorted(
            self._facts.values(),
            key=lambda f: f.confidence,
        )
        return sorted_facts[:limit]

    def consolidate_from_episodic(
        self,
        episodes: list[Any],
    ) -> int:
        newly_consolidated = 0
        
        for episode in episodes:
            key = (episode.state, episode.action, episode.next_state)
            
            if key not in self._facts:
                newly_consolidated += 1
            
            self.record_transition(
                episode.state,
                episode.action,
                episode.next_state,
            )
        
        return newly_consolidated

    def step(self) -> None:
        self._step_count += 1

    def statistics(self) -> dict[str, Any]:
        if not self._facts:
            return {
                "total_facts": 0,
                "avg_confidence": 0.0,
                "avg_stability": 0.0,
                "total_observations": 0,
            }
        
        facts = list(self._facts.values())
        
        return {
            "total_facts": len(facts),
            "avg_confidence": sum(f.confidence for f in facts) / len(facts),
            "avg_stability": sum(f.stability for f in facts) / len(facts),
            "total_observations": sum(f.observations for f in facts),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "facts": {
                f"{s}-{r}-{o}": {
                    "confidence": f.confidence,
                    "observations": f.observations,
                    "stability": f.stability,
                }
                for (s, r, o), f in self._facts.items()
            },
            "step_count": self._step_count,
        }
