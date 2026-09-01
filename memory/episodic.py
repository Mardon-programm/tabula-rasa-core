from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EpisodilMemory:
    timestamp: float
    state: str
    action: str
    next_state: str
    reward: float
    prediction_error: float = 0.0
    surprise: float = 0.0


class EpisodilMemoryBuffer:

    def __init__(self, limit: int = 500) -> None:
        self.limit = limit
        self._buffer: list[EpisodilMemory] = []

    def store(
        self,
        timestamp: float,
        state: str,
        action: str,
        next_state: str,
        reward: float,
        prediction_error: float = 0.0,
        surprise: float = 0.0,
    ) -> None:
        memory = EpisodilMemory(
            timestamp=timestamp,
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            prediction_error=prediction_error,
            surprise=surprise,
        )
        
        self._buffer.append(memory)
        
        if len(self._buffer) > self.limit:
            self._buffer.pop(0)

    def recent(self, n: int = 10) -> list[EpisodilMemory]:
        return self._buffer[-n:]

    def replay_sample(
        self,
        n: int = 5,
        prioritize_errors: bool = True,
    ) -> list[EpisodilMemory]:
        if not self._buffer:
            return []
        
        if prioritize_errors:
            sorted_mem = sorted(
                self._buffer,
                key=lambda m: m.prediction_error,
                reverse=True,
            )
            return sorted_mem[:n]
        else:
            start = max(0, len(self._buffer) - 100)
            return self._buffer[start:start + n]

    def get_all(self) -> list[EpisodilMemory]:
        return self._buffer.copy()

    def clear(self) -> None:
        self._buffer.clear()

    def statistics(self) -> dict[str, Any]:
        if not self._buffer:
            return {
                "size": 0,
                "capacity": self.limit,
                "avg_error": 0.0,
                "avg_surprise": 0.0,
            }
        
        avg_error = sum(m.prediction_error for m in self._buffer) / len(self._buffer)
        avg_surprise = sum(m.surprise for m in self._buffer) / len(self._buffer)
        
        return {
            "size": len(self._buffer),
            "capacity": self.limit,
            "usage_percent": len(self._buffer) / self.limit * 100,
            "avg_error": avg_error,
            "avg_surprise": avg_surprise,
        }
