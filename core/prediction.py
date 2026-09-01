from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Prediction:
    expected_state: str
    confidence: float
    alternatives: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass(slots=True)
class PredictionError:

    predicted_state: str
    actual_state: str
    magnitude: float  
    surprise: float  
    corrected: bool = False


class PredictionEngine:

    def __init__(self) -> None:
        self.total_predictions = 0
        self.correct_predictions = 0
        
        self._prediction_history: list[tuple[Prediction, str]] = []
        self._error_history: list[PredictionError] = []
        
        self._confidence_buckets: dict[int, list[bool]] = {}

    def predict(
        self,
        current_state: str,
        world_model: Any,
        available_transitions: dict[str, dict[str, float]] | None = None,
    ) -> Prediction:

        alternatives: dict[str, float] = {}
        
        if available_transitions:
            for action, next_states in available_transitions.items():
                for next_state, prob in next_states.items():
                    if next_state not in alternatives:
                        alternatives[next_state] = 0.0
                    alternatives[next_state] += prob
        
        if alternatives:
            total = sum(alternatives.values())
            alternatives = {
                state: prob / total
                for state, prob in alternatives.items()
            }
        else:
            alternatives = {}
        
        if alternatives:
            expected_state = max(
                alternatives.items(),
                key=lambda x: x[1],
            )[0]
            confidence = alternatives[expected_state]
        else:
            expected_state = current_state
            confidence = 0.0
        
        prediction = Prediction(
            expected_state=expected_state,
            confidence=confidence,
            alternatives=alternatives.copy(),
        )
        
        self.total_predictions += 1
        
        return prediction

    def evaluate(
        self,
        prediction: Prediction,
        actual_state: str,
    ) -> PredictionError:

        magnitude = 0.0 if prediction.expected_state == actual_state else 1.0
        
        actual_prob = prediction.alternatives.get(actual_state, 0.0)
        surprise = 1.0 - actual_prob
        
        error = PredictionError(
            predicted_state=prediction.expected_state,
            actual_state=actual_state,
            magnitude=magnitude,
            surprise=surprise,
        )
        
        if magnitude == 0.0:
            self.correct_predictions += 1
        
        self._prediction_history.append((prediction, actual_state))
        self._error_history.append(error)
        
        confidence_bucket = int(prediction.confidence * 10)
        if confidence_bucket not in self._confidence_buckets:
            self._confidence_buckets[confidence_bucket] = []
        self._confidence_buckets[confidence_bucket].append(magnitude == 0.0)
        
        return error

    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def calibration_error(self) -> float:
        if not self._confidence_buckets:
            return 0.0
        
        total_error = 0.0
        total_count = 0
        
        for confidence_bucket, results in self._confidence_buckets.items():
            if not results:
                continue
            
            predicted_confidence = (confidence_bucket + 0.5) / 10.0
            actual_accuracy = sum(results) / len(results)
            
            error = abs(predicted_confidence - actual_accuracy)
            total_error += error * len(results)
            total_count += len(results)
        
        if total_count == 0:
            return 0.0
        
        return total_error / total_count

    def recent_accuracy(self, n: int = 100) -> float:
        if not self._error_history:
            return 0.0
        
        recent = self._error_history[-n:]
        correct = sum(1 for e in recent if e.magnitude == 0.0)
        
        return correct / len(recent)

    def concept_drift_signal(self) -> float:
        if len(self._error_history) < 10:
            return 0.0
        
        older_errors = [
            e.magnitude for e in self._error_history[-20:-10]
        ]
        recent_errors = [
            e.magnitude for e in self._error_history[-10:]
        ]
        
        older_avg = sum(older_errors) / len(older_errors) if older_errors else 0.0
        recent_avg = sum(recent_errors) / len(recent_errors)
        
        drift = max(0.0, recent_avg - older_avg)
        
        return min(1.0, drift)

    def reset(self) -> None:
        self.total_predictions = 0
        self.correct_predictions = 0
        self._prediction_history.clear()
        self._error_history.clear()
        self._confidence_buckets.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "accuracy": self.accuracy(),
            "calibration_error": self.calibration_error(),
            "concept_drift_signal": self.concept_drift_signal(),
            "recent_accuracy": self.recent_accuracy(),
        }
