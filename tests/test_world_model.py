import pytest
from core.world_model import WorldModel, Entity, Relation, Transition


def test_world_model_entity_creation():
    wm = WorldModel()
    
    wm.add_entity("A", properties={"type": "state"})
    wm.add_entity("B", properties={"type": "state"})
    
    assert len(wm._entities) == 2
    assert "A" in wm._entities
    assert "B" in wm._entities


def test_world_model_transition_observation():
    wm = WorldModel()
    
    wm.observe_transition("A", "X", "B")
    wm.observe_transition("A", "X", "B")
    wm.observe_transition("A", "Y", "C")
    
    assert len(wm._transitions) == 2
    
    assert wm._state_visits["A"] >= 2
    assert wm._state_visits["B"] >= 1


def test_world_model_prediction():
    wm = WorldModel()
    
    wm.observe_transition("A", "X", "B")
    wm.observe_transition("A", "X", "B")
    wm.observe_transition("A", "Y", "C")
    
    result = wm.predict_next_state("A", "X")
    assert result is not None
    assert result[0] == "B"
    assert result[1] > 0.5  

def test_world_model_transitions_map():
    wm = WorldModel()
    
    wm.observe_transition("A", "X", "B")
    wm.observe_transition("A", "Y", "C")
    wm.observe_transition("A", "Z", "A")
    
    transitions = wm.get_possible_transitions("A")
    
    assert "X" in transitions
    assert "Y" in transitions
    assert "Z" in transitions
    assert "B" in transitions["X"]


def test_world_model_uncertainty():
    wm = WorldModel()
    
    unc = wm.get_uncertainty("A")
    assert unc.epistemic > 0.5
    
    for _ in range(10):
        wm.observe_transition("A", "X", "B")
    
    unc2 = wm.get_uncertainty("A")
    assert unc2.epistemic < unc.epistemic


def test_world_model_causal_links():
    wm = WorldModel()
    
    wm.observe_transition("A", "action", "B")
    wm.observe_transition("B", "action", "C")

    assert "A" in wm._causal_links
    assert "B" in wm._causal_links["A"]


def test_world_model_statistics():
    wm = WorldModel()
    
    wm.observe_transition("A", "X", "B")
    wm.observe_transition("B", "X", "C")
    
    stats = wm.get_statistics()
    
    assert stats["entities"] >= 3
    assert stats["transitions"] >= 2
    assert len(stats["state_visits"]) >= 3


def test_world_model_snapshot():
    wm = WorldModel()
    
    wm.observe_transition("A", "X", "B")
    wm.observe_transition("B", "X", "C")
    
    snapshot = wm.snapshot()
    
    assert "entities" in snapshot
    assert "transitions" in snapshot
    assert "state_visits" in snapshot
