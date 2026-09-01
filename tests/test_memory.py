import pytest
from memory.episodic import EpisodilMemoryBuffer
from memory.semantic import SemanticMemory
from memory.consolidation import MemoryConsolidation


def test_episodic_memory_storage():
    buffer = EpisodilMemoryBuffer(limit=10)
    
    for i in range(5):
        buffer.store(
            timestamp=float(i),
            state=f"S{i}",
            action="X",
            next_state=f"S{i+1}",
            reward=float(i),
        )
    
    assert len(buffer.get_all()) == 5


def test_episodic_memory_overflow():
    buffer = EpisodilMemoryBuffer(limit=5)
    
    for i in range(10):
        buffer.store(
            timestamp=float(i),
            state="A",
            action="X",
            next_state="B",
            reward=1.0,
        )
    
    assert len(buffer.get_all()) == 5


def test_episodic_memory_recent():
    buffer = EpisodilMemoryBuffer(limit=100)
    
    for i in range(20):
        buffer.store(float(i), "A", "X", "B", 1.0)
    
    recent = buffer.recent(5)
    assert len(recent) == 5


def test_semantic_memory_facts():
    sem_mem = SemanticMemory()
    
    sem_mem.record_transition("A", "X", "B")
    sem_mem.record_transition("A", "X", "B")
    sem_mem.record_transition("B", "X", "C")
    
    assert len(sem_mem._facts) >= 2


def test_semantic_memory_confidence():
    sem_mem = SemanticMemory()
    
    sem_mem.record_transition("A", "X", "B")
    fact1 = sem_mem.get_fact("A", "X", "B")
    conf1 = fact1.confidence if fact1 else 0.0
    
    for _ in range(10):
        sem_mem.record_transition("A", "X", "B")
    
    fact2 = sem_mem.get_fact("A", "X", "B")
    conf2 = fact2.confidence if fact2 else 0.0
    
    assert conf2 > conf1


def test_semantic_memory_query():
    sem_mem = SemanticMemory()
    
    sem_mem.record_transition("A", "X", "B")
    sem_mem.record_transition("A", "Y", "C")
    sem_mem.record_transition("B", "X", "C")
    
    facts_a = sem_mem.query("A")
    assert len(facts_a) >= 2
    
    facts_ax = sem_mem.query("A", "X")
    assert len(facts_ax) >= 1


def test_memory_consolidation_cycle():
    episodic = EpisodilMemoryBuffer(limit=100)
    semantic = SemanticMemory()
    consolidation = MemoryConsolidation(consolidation_interval=5)
    
    for i in range(10):
        episodic.store(float(i), f"S{i}", "X", f"S{i+1}", 1.0)
    
    assert consolidation.should_consolidate() == False
    
    consolidation.step()
    consolidation.step()
    consolidation.step()
    consolidation.step()
    consolidation.step()
    
    assert consolidation.should_consolidate()
    
    event = consolidation.consolidate(episodic, semantic, 0.0)
    assert event.newly_generalized >= 0


def test_memory_retention():
    consolidation = MemoryConsolidation()
    
    retention = consolidation.retention_rate()
    assert retention >= 0.0
    assert retention <= 1.0


def test_episodic_statistics():
    buffer = EpisodilMemoryBuffer(limit=10)
    
    stats = buffer.statistics()
    assert stats["size"] == 0
    
    for i in range(5):
        buffer.store(float(i), "A", "X", "B", 1.0)
    
    stats = buffer.statistics()
    assert stats["size"] == 5
    assert stats["capacity"] == 10
