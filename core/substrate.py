from __future__ import annotations

from dataclasses import dataclass, field
from math import log2
from typing import Any, Iterable


@dataclass(slots=True, frozen=True)
class Observation:
    timestamp: float
    value: Any
    modality: str = "unknown"


@dataclass(slots=True)
class Segment:
    observations: list[Observation] = field(default_factory=list)

    information_density: float = 0.0
    entropy: float = 0.0

    @property
    def start_time(self) -> float | None:
        return self.observations[0].timestamp if self.observations else None

    @property
    def end_time(self) -> float | None:
        return self.observations[-1].timestamp if self.observations else None

    @property
    def duration(self) -> float:
        if len(self.observations) < 2:
            return 0.0

        return self.end_time - self.start_time

    @property
    def size(self) -> int:
        return len(self.observations)


class EntropyCalculator:
    @staticmethod
    def calculate(values: Iterable[Any]) -> float:
        values = list(values)

        if not values:
            return 0.0

        frequencies: dict[Any, int] = {}

        for value in values:
            try:
                frequencies[value] = frequencies.get(value, 0) + 1
            except TypeError:
                key = repr(value)
                frequencies[key] = frequencies.get(key, 0) + 1

        total = len(values)

        entropy = 0.0

        for count in frequencies.values():
            probability = count / total
            entropy -= probability * log2(probability)

        return entropy


class EntropySegmenter:
    def __init__(
        self,
        *,
        window_size: int = 8,
        threshold: float = 0.5,
        minimum_segment_size: int = 1,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")

        if threshold < 0:
            raise ValueError("threshold must be >= 0")

        if minimum_segment_size < 1:
            raise ValueError("minimum_segment_size must be >= 1")

        self.window_size = window_size
        self.threshold = threshold
        self.minimum_segment_size = minimum_segment_size

    def segment(
        self,
        observations: list[Observation],
    ) -> list[Segment]:
        if not observations:
            return []

        if len(observations) <= self.window_size:
            return [self._build_segment(observations)]

        segments: list[Segment] = []
        current: list[Observation] = []

        previous_entropy = 0.0

        for observation in observations:
            current.append(observation)

            if len(current) < self.window_size:
                continue

            window = current[-self.window_size:]

            entropy = EntropyCalculator.calculate(
                item.value for item in window
            )

            delta = abs(entropy - previous_entropy)

            should_split = (
                delta >= self.threshold
                and len(current) >= self.minimum_segment_size
            )

            if should_split and len(current) > self.window_size:
                split_point = len(current) - self.window_size

                segment_data = current[:split_point]

                if segment_data:
                    segments.append(
                        self._build_segment(segment_data)
                    )

                current = current[split_point:]

            previous_entropy = entropy

        if current:
            segments.append(self._build_segment(current))

        return segments

    @staticmethod
    def _build_segment(
        observations: list[Observation],
    ) -> Segment:
        entropy = EntropyCalculator.calculate(
            item.value for item in observations
        )

        density = (
            entropy / len(observations)
            if observations
            else 0.0
        )

        return Segment(
            observations=list(observations),
            information_density=density,
            entropy=entropy,
        )


class SensorySubstrate:
    def __init__(
        self,
        *,
        window_size: int = 8,
        entropy_threshold: float = 0.5,
        buffer_limit: int = 10_000,
    ) -> None:
        if buffer_limit < window_size:
            raise ValueError(
                "buffer_limit must be >= window_size"
            )

        self.segmenter = EntropySegmenter(
            window_size=window_size,
            threshold=entropy_threshold,
        )

        self.buffer_limit = buffer_limit
        self._buffer: list[Observation] = []

        self.total_observations = 0
        self.total_segments = 0

    def ingest(self, observation: Observation) -> None:
        self._buffer.append(observation)
        self.total_observations += 1

        if len(self._buffer) > self.buffer_limit:
            overflow = len(self._buffer) - self.buffer_limit
            del self._buffer[:overflow]

    def ingest_many(
        self,
        observations: Iterable[Observation],
    ) -> None:
        for observation in observations:
            self.ingest(observation)

    def perceive(self) -> list[Segment]:
        segments = self.segmenter.segment(self._buffer)
        self.total_segments += len(segments)
        return segments

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def average_entropy(self) -> float:
        if not self._buffer:
            return 0.0

        return EntropyCalculator.calculate(
            item.value for item in self._buffer
        )

    def clear(self) -> None:
        self._buffer.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "buffer_size": self.buffer_size,
            "total_observations": self.total_observations,
            "total_segments": self.total_segments,
            "average_entropy": self.average_entropy,
            "segmenter": {
                "window_size": self.segmenter.window_size,
                "threshold": self.segmenter.threshold,
                "minimum_segment_size": (
                    self.segmenter.minimum_segment_size
                ),
            },
        }

    def add_observation(self, observation: Observation) -> None:
        self.ingest(observation)