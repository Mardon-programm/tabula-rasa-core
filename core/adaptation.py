from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AdaptationEvent:
    timestamp: float
    error_magnitude: float
    belief_updated: bool
    confidence_delta: float
    concept_drift_detected: bool


class AdaptationEngine:

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        drift_threshold: float = 0.3,
        stability_factor: float = 0.8,
    ) -> None:
        self.learning_rate = learning_rate
        self.drift_threshold = drift_threshold
        self.stability_factor = stability_factor
        
        self._events: list[AdaptationEvent] = []
        self._total_updates = 0
        self._drift_detections = 0

    def adapt_to_error(
        self,
        prediction_error: Any,  
        world_model: Any,
        concept_drift_signal: float,
    ) -> AdaptationEvent:

        error_mag = prediction_error.magnitude
        
        drift_detected = concept_drift_signal > self.drift_threshold
        
        lr = self.learning_rate
        if drift_detected:
            lr *= (1.0 + concept_drift_signal)  
        
        lr *= self.stability_factor
        
        world_model.update_uncertainty(
            prediction_error.predicted_state,
            error_mag,
        )
        
        if prediction_error.surprise > 0.5:
            pass
        
        confidence_delta = -lr * error_mag
        
        event = AdaptationEvent(
            timestamp=0.0,  
            error_magnitude=error_mag,
            belief_updated=error_mag > 0.0,
            confidence_delta=confidence_delta,
            concept_drift_detected=drift_detected,
        )
        
        self._events.append(event)
        
        if event.belief_updated:
            self._total_updates += 1
        
        if drift_detected:
            self._drift_detections += 1
        
        return event

    def handle_concept_drift(
        self,
        old_state: str,
        new_state: str,
        world_model: Any,
    ) -> None:
        
        unc = world_model.get_uncertainty(old_state)
        
        unc.epistemic = min(1.0, unc.epistemic + 0.2)
        unc.model_uncertainty = min(1.0, unc.model_uncertainty + 0.15)

    def plasticity_vs_stability(self) -> dict[str, float]:

        if not self._events:
            return {
                "plasticity": 0.5,
                "stability": 0.5,
                "adaptation_rate": 0.0,
                "drift_detection_rate": 0.0,
            }
        
        recent_events = self._events[-50:]
        recent_updates = sum(
            1 for e in recent_events if e.belief_updated
        )
        plasticity = recent_updates / len(recent_events)
        
        stability = 1.0 - plasticity
        
        adaptation_rate = sum(
            abs(e.confidence_delta) for e in recent_events
        ) / len(recent_events)
        
        return {
            "plasticity": plasticity,
            "stability": stability,
            "adaptation_rate": adaptation_rate,
            "drift_detection_rate": min(
                1.0,
                self._drift_detections / max(1, len(self._events)) * 10,
            ),
        }

    def recovery_after_drift(self) -> float:

        if len(self._events) < 10:
            return 0.0
        
        last_drift_idx = -1
        for i in range(len(self._events) - 1, -1, -1):
            if self._events[i].concept_drift_detected:
                last_drift_idx = i
                break
        
        if last_drift_idx < 0:
            return 1.0  
        
        post_drift_events = self._events[last_drift_idx + 1:]
        
        if len(post_drift_events) < 5:
            return 0.5  
        
        recent_errors = [
            e.error_magnitude for e in post_drift_events[:5]
        ]
        recent_avg = sum(recent_errors) / len(recent_errors)
        
        recovery = max(0.0, 1.0 - recent_avg)
        
        return recovery


class ModelAdapter:
    
    def __init__(self, world_model: Any):
        self.world_model = world_model
        self.update_count = 0
    
    def adapt_to_prediction_error(
        self,
        error: Any,
        current_state: str,
        actual_state: str,
        action: str,
        timestamp: float
    ) -> None:
        
        if hasattr(self.world_model, 'update_edge_weight'):
            old_weight = getattr(self.world_model, 'get_edge_weight', lambda *a: 0.5)(
                current_state, actual_state, action
            )
            new_weight = min(1.0, old_weight + 0.1 * error.magnitude)
            self.world_model.update_edge_weight(current_state, actual_state, action, new_weight)
        
        self.update_count += 1


    def statistics(self) -> dict[str, Any]:

        pv_s = self.plasticity_vs_stability()
        
        return {
            "total_events": len(self._events),
            "total_updates": self._total_updates,
            "drift_detections": self._drift_detections,
            "plasticity": pv_s["plasticity"],
            "stability": pv_s["stability"],
            "adaptation_rate": pv_s["adaptation_rate"],
            "recovery_score": self.recovery_after_drift(),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_updates": self._total_updates,
            "drift_detections": self._drift_detections,
            "statistics": self.statistics(),
        }
