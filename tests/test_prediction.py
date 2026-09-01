import pytest
from core.prediction import PredictionEngine, Prediction, PredictionError
from core.world_model import WorldModel


def test_prediction_creation():
    pred = Prediction(
        expected_state="B",
        confidence=0.85,
        alternatives={"B": 0.85, "C": 0.15},
    )
    
    assert pred.expected_state == "B"
    assert pred.confidence == 0.85
    assert len(pred.alternatives) == 2


def test_prediction_engine_accuracy():
    engine = PredictionEngine()
    
    assert engine.accuracy() == 0.0
    
    pred1 = Prediction("B", 0.9, {"B": 0.9, "C": 0.1})
    error1 = engine.evaluate(pred1, "B")  
    assert error1.magnitude == 0.0
    
    pred2 = Prediction("A", 0.7, {"A": 0.7})
    error2 = engine.evaluate(pred2, "C")  
    assert error2.magnitude == 1.0
    
    assert engine.accuracy() == 0.5 


def test_prediction_engine_calibration():
    """Test calibration error."""
    engine = PredictionEngine()
    
    for _ in range(10):
        pred = Prediction("A", 0.9, {"A": 0.9})
        engine.evaluate(pred, "A") 
    
    cal_error = engine.calibration_error()
    assert cal_error < 0.2  


def test_concept_drift_signal():
    engine = PredictionEngine()
    
    assert engine.concept_drift_signal() == 0.0
    
    for i in range(15):
        pred = Prediction("A", 0.95, {"A": 0.95})
        engine.evaluate(pred, "A")
    
    assert engine.concept_drift_signal() < 0.1
    
    for i in range(10):
        pred = Prediction("A", 0.95, {"A": 0.95})
        engine.evaluate(pred, "B")  
    
    drift = engine.concept_drift_signal()
    assert drift > 0.3


def test_prediction_with_world_model():
    world_model = WorldModel()
    engine = PredictionEngine()
    
    world_model.observe_transition("A", "X", "B")
    world_model.observe_transition("A", "X", "B")
    world_model.observe_transition("B", "X", "C")
    
    transitions = world_model.get_possible_transitions("A")
    
    pred = engine.predict("A", world_model, available_transitions=transitions)
    
    assert pred.expected_state == "B"
    assert pred.confidence > 0.5


def test_prediction_error():
    error = PredictionError(
        predicted_state="B",
        actual_state="C",
        magnitude=1.0,
        surprise=0.8,
    )
    
    assert error.predicted_state == "B"
    assert error.actual_state == "C"
    assert error.magnitude == 1.0
    assert error.surprise == 0.8


def test_prediction_reset():
    engine = PredictionEngine()
    
    for _ in range(10):
        pred = Prediction("A", 0.8, {"A": 0.8})
        engine.evaluate(pred, "A")
    
    assert engine.total_predictions > 0
    
    engine.reset()
    
    assert engine.total_predictions == 0
    assert engine.correct_predictions == 0
    assert engine.accuracy() == 0.0
