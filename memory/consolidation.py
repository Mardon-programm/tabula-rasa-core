from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ConsolidationEvent:
    timestamp: float
    episodic_count: int
    semantic_count: int
    newly_generalized: int
    patterns_extracted: int


class MemoryConsolidation:

    def __init__(
        self,
        *,
        consolidation_interval: int = 50,
        forgetting_rate: float = 0.01,
    ) -> None:
        self.consolidation_interval = consolidation_interval
        self.forgetting_rate = forgetting_rate
        
        self._step_count = 0
        self._consolidation_events: list[ConsolidationEvent] = []

    def should_consolidate(self) -> bool:
        return self._step_count % self.consolidation_interval == 0

    def consolidate(
        self,
        episodic_memory: Any,
        semantic_memory: Any,
        timestamp: float,
    ) -> ConsolidationEvent:
        recent_episodes = episodic_memory.recent(
            n=min(50, self.consolidation_interval * 2)
        )
        
        newly_generalized = semantic_memory.consolidate_from_episodic(
            recent_episodes
        )
        
        patterns_extracted = self._extract_patterns(
            semantic_memory
        )
        
        event = ConsolidationEvent(
            timestamp=timestamp,
            episodic_count=len(episodic_memory.get_all()),
            semantic_count=len(semantic_memory._facts),
            newly_generalized=newly_generalized,
            patterns_extracted=patterns_extracted,
        )
        
        self._consolidation_events.append(event)
        
        return event

    def _extract_patterns(self, semantic_memory: Any) -> int:
        patterns = 0
        
        for fact in semantic_memory.most_confident_facts(limit=50):
            if fact.confidence > 0.7 and fact.observations > 3:
                patterns += 1
        
        return patterns

    def apply_forgetting(
        self,
        semantic_memory: Any,
    ) -> int:
        weakened = 0
        
        for fact in semantic_memory._facts.values():
            age = self._step_count - fact.last_seen
            
            if age > 100:
                fact.confidence *= (1.0 - self.forgetting_rate)
                fact.uncertainty *= (1.0 + self.forgetting_rate)
                weakened += 1
        
        return weakened

    def step(self) -> None:
        self._step_count += 1

    def retention_rate(self) -> float:
        if not self._consolidation_events:
            return 1.0
        
        last_event = self._consolidation_events[-1]
        
        if last_event.semantic_count == 0:
            return 1.0
        
        retention = 0.7  
        
        return min(1.0, retention)

    def consolidation_efficiency(self) -> float:
        if not self._consolidation_events:
            return 0.0
        
        total_generalized = sum(
            e.newly_generalized for e in self._consolidation_events
        )
        
        return total_generalized / len(self._consolidation_events)

    def statistics(self) -> dict[str, Any]:
        return {
            "consolidation_events": len(self._consolidation_events),
            "retention_rate": self.retention_rate(),
            "consolidation_efficiency": self.consolidation_efficiency(),
            "step_count": self._step_count,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "step_count": self._step_count,
            "consolidation_events": len(self._consolidation_events),
            "statistics": self.statistics(),
        }
