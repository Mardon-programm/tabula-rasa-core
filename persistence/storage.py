from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


class PersistenceManager:

    def __init__(self, storage_dir: str | Path = "checkpoints") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def save_agent(
        self,
        agent: Any,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        checkpoint = {
            "agent_state": self._serialize_agent(agent),
            "metadata": metadata or {},
        }
        
        filepath = self.storage_dir / f"{name}.pkl"
        
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f)
        
        return filepath

    def load_agent(
        self,
        name: str,
        agent_class: type,
    ) -> tuple[Any, dict[str, Any]]:
        filepath = self.storage_dir / f"{name}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        with open(filepath, "rb") as f:
            checkpoint = pickle.load(f)
        
        agent_state = checkpoint["agent_state"]
        metadata = checkpoint["metadata"]
        
        agent = self._deserialize_agent(agent_state, agent_class)
        
        return agent, metadata

    def save_snapshot(
        self,
        agent: Any,
        name: str,
    ) -> Path:
        
        snapshot = {
            "world_model": agent.world_model.snapshot() if hasattr(agent, "world_model") else {},
            "self_model": agent.self_model.statistics() if hasattr(agent, "self_model") else {},
            "prediction_engine": agent.prediction_engine.snapshot() if hasattr(agent, "prediction_engine") else {},
            "adaptation": agent.adaptation.statistics() if hasattr(agent, "adaptation") else {},
            "memory": {
                "episodic": len(agent.memory_episodic.get_all()) if hasattr(agent, "memory_episodic") else 0,
            },
        }
        
        filepath = self.storage_dir / f"{name}_snapshot.json"
        
        with open(filepath, "w") as f:
            json.dump(snapshot, f, indent=2)
        
        return filepath

    def list_checkpoints(self) -> list[str]:
        files = list(self.storage_dir.glob("*.pkl"))
        return [f.stem for f in files]

    def _serialize_agent(self, agent: Any) -> dict[str, Any]:
        state = {}

        if hasattr(agent, "world_model"):
            state["world_model"] = agent.world_model.snapshot()
        
        if hasattr(agent, "brain"):
            state["brain"] = {
                "nodes": len(agent.brain._nodes),
                "edges": len(agent.brain._edges),
            }
        
        if hasattr(agent, "self_model"):
            state["self_model"] = agent.self_model.snapshot()
        
        if hasattr(agent, "memory_episodic"):
            state["memory_episodic_size"] = len(agent.memory_episodic.get_all())
        
        if hasattr(agent, "step_count"):
            state["step_count"] = agent.step_count
        
        return state

    def _deserialize_agent(
        self,
        state: dict[str, Any],
        agent_class: type,
    ) -> Any:
        agent = agent_class()
        return agent

    def cleanup_old_checkpoints(
        self,
        keep_count: int = 5,
    ) -> int:
        files = sorted(
            self.storage_dir.glob("*.pkl"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        
        to_delete = files[keep_count:]
        
        for f in to_delete:
            f.unlink()
        
        return len(to_delete)

    def export_timeline(
        self,
        checkpoints: list[str],
        output_file: str = "timeline.json",
    ) -> Path:
        timeline = {
            "checkpoints": [],
        }
        
        for name in checkpoints:
            filepath = self.storage_dir / f"{name}.pkl"
            if filepath.exists():
                timeline["checkpoints"].append({
                    "name": name,
                    "timestamp": filepath.stat().st_mtime,
                })
        
        output_path = self.storage_dir / output_file
        
        with open(output_path, "w") as f:
            json.dump(timeline, f, indent=2)
        
        return output_path
