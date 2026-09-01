from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Episode:
    episode_id: int
    timestamp: float
    state: str
    action: str
    result: str
    reward: float


@dataclass(slots=True)
class SemanticFact:
    subject: str
    relation: str
    object: str
    confidence: float = 0.0
    observations: int = 0


class MemorySystem:
    def __init__(
        self,
        *,
        episodic_limit: int = 1_000,
    ) -> None:
        self.episodic_limit = episodic_limit

        self._episodes: list[Episode] = []
        self._semantic: dict[
            tuple[str, str, str],
            SemanticFact,
        ] = {}

        self._next_episode_id = 1
        self._consolidated_count = 0


    # Episodic memory

    def remember(
        self,
        *,
        timestamp: float,
        state: str,
        action: str,
        result: str,
        reward: float,
    ) -> Episode:
        episode = Episode(
            episode_id=self._next_episode_id,
            timestamp=timestamp,
            state=state,
            action=action,
            result=result,
            reward=reward,
        )

        self._next_episode_id += 1

        self._episodes.append(episode)

        if len(self._episodes) > self.episodic_limit:
            self._episodes.pop(0)

        return episode


    # Consolidation

    def consolidate(self) -> int:
        """
        Extract simple repeated state/action/result patterns.

        Returns number of newly processed episodes.
        """

        if not self._episodes:
            return 0

        processed = 0

        for episode in self._episodes:
            key = (
                episode.state,
                episode.action,
                episode.result,
            )

            if key not in self._semantic:
                self._semantic[key] = SemanticFact(
                    subject=episode.state,
                    relation=episode.action,
                    object=episode.result,
                    confidence=0.1,
                    observations=1,
                )
            else:
                fact = self._semantic[key]

                fact.observations += 1
                fact.confidence += (
                    0.1 * (1.0 - fact.confidence)
                )

            processed += 1

        self._consolidated_count += processed

        return processed

    def semantic_lookup(
        self,
        subject: str,
    ) -> list[SemanticFact]:
        return [
            fact
            for fact in self._semantic.values()
            if fact.subject == subject
        ]

    def forget_episodes(
        self,
        *,
        keep_recent: int = 100,
    ) -> int:
        """
        Simulated infantile forgetting.

        Semantic memory remains intact.
        """

        if keep_recent < 0:
            raise ValueError("keep_recent must be >= 0")

        if len(self._episodes) <= keep_recent:
            return 0

        forgotten = len(self._episodes) - keep_recent

        self._episodes = self._episodes[-keep_recent:]

        return forgotten


    # Metrics

    @property
    def episodic_count(self) -> int:
        return len(self._episodes)

    @property
    def semantic_count(self) -> int:
        return len(self._semantic)

    @property
    def consolidation_ratio(self) -> float:
        if self._consolidated_count == 0:
            return 0.0

        return min(
            1.0,
            self.semantic_count
            / self._consolidated_count,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "episodic_memory": self.episodic_count,
            "semantic_memory": self.semantic_count,
            "consolidated_episodes": self._consolidated_count,
            "consolidation_ratio": self.consolidation_ratio,
            "semantic_facts": [
                {
                    "subject": fact.subject,
                    "relation": fact.relation,
                    "object": fact.object,
                    "confidence": round(
                        fact.confidence,
                        4,
                    ),
                    "observations": fact.observations,
                }
                for fact in self._semantic.values()
            ],
        }