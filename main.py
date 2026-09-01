"""
TR-Core — Tabula Rasa Core Developmental Cognitive Agent

Autonomous learning system with world model, prediction, planning, and self-model.
Designed as reproducible R&D research platform demonstrating:

- Autonomous continual learning
- World model formation through interaction  
- Prediction with calibration tracking
- Concept drift detection and adaptation
- Curiosity-driven exploration
- Memory consolidation
- Metacognition and self-modeling
- Goal-directed planning
- Developmental stage progression
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

from core.axiom_matrix import AxiomMatrix, Axiom
from core.substrate import SensorySubstrate, Observation
from core.graph_brain import DynamicSparseGraph
from core.world_model import WorldModel
from core.prediction import PredictionEngine, PredictionError
from core.curiosity_rl import CuriosityEngine
from core.adaptation import AdaptationEngine, ModelAdapter
from core.developmental_stage import DevelopmentalController
from memory.episodic import EpisodilMemoryBuffer
from memory.semantic import SemanticMemory
from memory.consolidation import MemoryConsolidation
from models.state import StateManager
from models.self_model import SelfModel
from planner.tree_search import TreeSearchPlanner
from experiments.metrics import MetricsCollector
from experiments.runner import ExperimentRunner
from persistence.storage import PersistenceManager
from environment.sandbox_sim import SandboxEnvironment


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


class TabulaRasaCore:
    """
    Complete developmental cognitive agent.
    
    Implements:
    - Perception & substrate (raw experience quantization)
    - World modeling (entity, property, relation, transition learning)
    - Prediction engine (state forecasting + confidence tracking)
    - Adaptation system (error-driven learning + drift detection)
    - Curiosity-driven exploration
    - Memory systems (episodic, semantic, consolidation)
    - Self-model (metacognition)
    - Planning (tree search)
    - Developmental stages (progressive capability emergence)
    """

    def __init__(self, seed: int = 42, enable_metrics: bool = True, verbose: bool = False) -> None:
        """
        Initialize TR-Core agent.
        
        Args:
            seed: Random seed for reproducibility
            enable_metrics: Whether to collect metrics
            verbose: Print detailed logs
        """
        self.seed = seed
        self.enable_metrics = enable_metrics
        self.verbose = verbose
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        
        self.development = DevelopmentalController()
        self.substrate = SensorySubstrate(window_size=4, entropy_threshold=0.4)
        self.brain = DynamicSparseGraph(learning_rate=0.2, decay_rate=0.005, pruning_threshold=0.02)
        self.world_model = WorldModel()
        self.prediction_engine = PredictionEngine()
        self.model_adapter = ModelAdapter(self.world_model)
        self.adaptation_engine = AdaptationEngine(learning_rate=0.1, drift_threshold=0.3)
        
        self.memory_episodic = EpisodilMemoryBuffer(limit=500)
        self.memory_semantic = SemanticMemory()
        self.consolidation = MemoryConsolidation(consolidation_interval=50)
        
        self.state_manager = StateManager()
        self.self_model = SelfModel()
        
        self.curiosity_engine = CuriosityEngine()
        self.planner = TreeSearchPlanner(max_depth=3, max_children=5)
        
        self.axioms = AxiomMatrix([
            Axiom(name="no-invalid-action", description="Agent may only execute known actions."),
        ])
        
        self.metrics = MetricsCollector() if enable_metrics else None
        self.persistence = PersistenceManager(storage_dir=CHECKPOINT_DIR)
        self.experiment_runner = None
        
        self.current_state: str = "START"
        self.previous_state: Optional[str] = None
        self.previous_action: Optional[str] = None
        
        self._prediction_errors: List[float] = []
        self._adaptation_events: int = 0
        self._drift_detections: int = 0
        
        if self.verbose:
            print("[TR-Core] Initialized")

    def perceive(self, env_state: str) -> None:
        """Process environmental observation."""
        observation = Observation(
            timestamp=time.time(),
            value=env_state,
            modality="environment"
        )
        self.substrate.add_observation(observation)
        self.previous_state = self.current_state
        self.current_state = env_state
        
        if self.verbose and self.step_count % 100 == 0:
            print(f"[Perception] state={env_state}")

    def predict(self, current_state: Optional[str] = None) -> Tuple[str, float]:
        """
        Generate prediction for next state.
        
        Returns:
            (predicted_state, confidence)
        """
        if current_state is None:
            current_state = self.current_state
        
        transitions = self.world_model.get_possible_transitions(current_state)
        prediction = self.prediction_engine.predict(
            current_state,
            self.world_model,
            available_transitions=transitions
        )
        
        return (prediction.expected_state, prediction.confidence)

    def evaluate_prediction(
        self,
        prediction,
        actual_state: str,
        timestamp: float
    ) -> PredictionError:
        """Evaluate prediction and compute error."""
        error = self.prediction_engine.evaluate(prediction, actual_state)
        
        # Track for statistics
        self._prediction_errors.append(error.magnitude)
        
        return error

    def adapt(
        self,
        prediction_error: Optional[PredictionError],
        concept_drift_signal: float
    ) -> None:
        """
        Adapt beliefs based on prediction error.
        
        Triggers model updates, drift detection, and self-model adjustments.
        """
        if prediction_error is None:
            return
        
        self._adaptation_events += 1
        
        # Update world model
        adjustment = min(0.3, prediction_error.magnitude * 0.5)
        self.model_adapter.adapt_to_prediction_error(
            prediction_error,
            self.previous_state or self.current_state,
            prediction_error.actual_state,
            self.previous_action or "X",
            time.time()
        )
        
        # Detect concept drift
        if self.prediction_engine.concept_drift_signal() > 0.3:
            self._drift_detections += 1
            if self.verbose:
                print(f"[Concept Drift] Detected at step {self.step_count}")
            self.self_model.uncertainty_level = min(
                1.0,
                self.self_model.uncertainty_level + 0.15
            )

    def explore(self) -> str:
        """
        Select exploratory action based on curiosity.
        
        Balances exploration (novel states) and exploitation (known states).
        """
        curiosity_reward = self.curiosity_engine.calculate_reward(
            state=self.current_state,
            prediction_confidence=self.prediction_engine.accuracy(),
            prediction_error=max(self._prediction_errors[-5:]) if self._prediction_errors else 0.0,
        )
        
        # Simple action space for sandbox
        available_actions = ["X", "Y", "Z"]
        
        # High curiosity → explore; Low curiosity → exploit
        if curiosity_reward > 0.6:
            return available_actions[self.step_count % len(available_actions)]
        else:
            return available_actions[0]

    def consolidate_memory(self) -> None:
        """
        Run memory consolidation cycle.
        
        Episodic memory → Pattern extraction → Semantic memory
        """
        if self.consolidation.should_consolidate():
            self.consolidation.consolidate(
                self.memory_episodic,
                self.memory_semantic,
                timestamp=time.time()
            )
            self.consolidation.apply_forgetting(self.memory_semantic)
            
            if self.verbose and self.step_count % 200 == 0:
                print(f"[Memory] Consolidated at step {self.step_count}")

    def plan(self, goal_state: Optional[str] = None) -> List[str]:
        """
        Generate action plan using tree search.
        
        Args:
            goal_state: Optional goal state. If None, uses current best trajectory.
            
        Returns:
            List of actions to reach goal
        """
        return self.planner.plan(
            self.current_state,
            self.world_model,
            goal_state=goal_state
        )

    def step(
        self,
        env_observation: str,
        actual_next_state: str,
        reward: float = 0.0
    ) -> Dict[str, Any]:
        """
        Execute one complete step of agent loop.
        
        Implements: Perceive → Predict → Compare → Adapt → Consolidate → Act
        
        Args:
            env_observation: Current environment observation
            actual_next_state: What state actually resulted
            reward: Reward signal from environment
            
        Returns:
            Step result dictionary with metrics
        """
        step_start = time.time()
        self.step_count += 1
        self.total_reward += reward
        
        self.perceive(env_observation)
        
        predicted_state, prediction_confidence = self.predict()
        prediction_obj = self.prediction_engine.predict(
            env_observation,
            self.world_model
        )
        
        is_correct = predicted_state == actual_next_state
        prediction_error = None
        
        if not is_correct:
            prediction_error = PredictionError(
                predicted_state=predicted_state,
                actual_state=actual_next_state,
                magnitude=1.0,
                surprise=1.0 - prediction_confidence,
            )
            self.evaluate_prediction(prediction_obj, actual_next_state, step_start)
        
        if self.previous_action:
            self.world_model.observe_transition(
                env_observation,
                self.previous_action,
                actual_next_state
            )
            self.brain.observe_transition(env_observation, actual_next_state)
            self.brain.decay()
        
        concept_drift_signal = self.prediction_engine.concept_drift_signal()
        if prediction_error:
            self.adapt(prediction_error, concept_drift_signal)
        
        prediction_error_mag = prediction_error.magnitude if prediction_error else 0.0
        prediction_surprise = prediction_error.surprise if prediction_error else 0.0
        
        self.memory_episodic.store(
            timestamp=step_start,
            state=env_observation,
            action=self.previous_action or "START",
            next_state=actual_next_state,
            reward=reward,
            prediction_error=prediction_error_mag,
            surprise=prediction_surprise,
        )
        
        self.consolidation.step()
        self.consolidate_memory()
        
        self.self_model.update_metrics(
            prediction_accuracy=self.prediction_engine.accuracy(),
            graph_size=len(self.brain._nodes),
            uncertainty=concept_drift_signal,
            exploration_rate=self.curiosity_engine.novelty(self.current_state),
            planning_success=0.7,
        )
        
        self.development.record_stage_transition(
            self.current_state,
            self.self_model.estimated_knowledge()
        )
        
        if self.self_model.should_explore():
            self.previous_action = self.explore()
        else:
            plan = self.plan()
            self.previous_action = plan[0] if plan else "X"
        
        if self.metrics:
            pv_s = self.adaptation_engine.plasticity_vs_stability()
            self.metrics.record(
                timestamp=step_start,
                prediction_accuracy=self.prediction_engine.accuracy(),
                calibration_error=self.prediction_engine.calibration_error(),
                adaptation_rate=pv_s["adaptation_rate"],
                plasticity=pv_s["plasticity"],
                stability=pv_s["stability"],
                drift_detection_rate=pv_s["drift_detection_rate"],
                recovery_score=self.adaptation_engine.recovery_after_drift(),
                episodic_size=len(self.memory_episodic.get_all()),
                semantic_size=len(self.memory_semantic._facts),
                retention_rate=self.consolidation.retention_rate(),
                curiosity_score=self.curiosity_engine.calculate_reward(
                    state=self.current_state,
                    prediction_confidence=prediction_confidence,
                    prediction_error=prediction_error_mag,
                ),
                exploration_rate=self.self_model.exploration_rate,
                planning_success_rate=0.7,
                estimated_knowledge=self.self_model.estimated_knowledge(),
                uncertainty_level=concept_drift_signal,
                total_reward=self.total_reward,
            )
        
        return {
            "step": self.step_count,
            "state": self.current_state,
            "action": self.previous_action,
            "predicted_state": predicted_state,
            "actual_state": actual_next_state,
            "prediction_accuracy": self.prediction_engine.accuracy(),
            "drift_signal": concept_drift_signal,
            "reward": reward,
            "total_reward": self.total_reward,
            "step_duration": time.time() - step_start,
        }

    def run_episode(
        self,
        environment: SandboxEnvironment,
        max_steps: int = 100,
        checkpoint_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete episode.
        
        Args:
            environment: Sandbox environment
            max_steps: Maximum steps per episode
            checkpoint_name: Optional name to save checkpoint
            
        Returns:
            Episode statistics
        """
        self.episode_count += 1
        print(f"\n[Episode {self.episode_count}] Starting (max {max_steps} steps)")
        
        obs = environment.observe()
        self.perceive(obs.state)
        
        episode_rewards = []
        episode_accuracy = []
        
        for step_idx in range(max_steps):
            env_result = environment.step(self.previous_action or "X")
            result = self.step(
                env_observation=self.current_state,
                actual_next_state=env_result.state,
                reward=env_result.reward
            )
            episode_rewards.append(env_result.reward)
            episode_accuracy.append(self.prediction_engine.accuracy())
            
            if step_idx % 20 == 0:
                print(
                    f"  Step {step_idx:3d}: "
                    f"state={env_result.state} "
                    f"accuracy={self.prediction_engine.accuracy():.1%} "
                    f"drift={'Y' if self.prediction_engine.concept_drift_signal() > 0.3 else 'N'}"
                )
        
        stats = {
            "episode": self.episode_count,
            "episode_length": step_idx + 1,
            "total_reward": sum(episode_rewards),
            "avg_reward": sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0,
            "prediction_accuracy": self.prediction_engine.accuracy(),
            "avg_accuracy": sum(episode_accuracy) / len(episode_accuracy) if episode_accuracy else 0.0,
            "memory_stats": {
                "episodic": len(self.memory_episodic.get_all()),
                "semantic": len(self.memory_semantic._facts),
            },
            "drift_detections": self._drift_detections,
            "adaptation_events": self._adaptation_events,
        }
        
        print(f"[Episode {self.episode_count}] Complete: {stats['total_reward']:.1f} reward, {stats['prediction_accuracy']:.1%} accuracy")
        
        if checkpoint_name:
            self.save_checkpoint(checkpoint_name)
        
        return stats

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status snapshot."""
        return {
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "current_state": self.current_state,
            "total_reward": self.total_reward,
            "prediction_accuracy": self.prediction_engine.accuracy(),
            "calibration": self.prediction_engine.calibration_error(),
            "concept_drift": self.prediction_engine.concept_drift_signal(),
            "memory": {
                "episodic": len(self.memory_episodic.get_all()),
                "semantic": len(self.memory_semantic._facts),
            },
            "self_model": self.self_model.statistics(),
            "development_stage": self.development.current_stage,
            "statistics": {
                "adaptation_events": self._adaptation_events,
                "drift_detections": self._drift_detections,
                "prediction_errors": len(self._prediction_errors),
            }
        }

    def print_status(self) -> None:
        """Print human-readable status report."""
        status = self.get_status()
        print("\n" + "=" * 70)
        print(f"TR-CORE STATUS (Step {status['step_count']}, Episode {status['episode_count']})")
        print("=" * 70)
        print(f"\nPerformance:")
        print(f"  Prediction Accuracy:     {status['prediction_accuracy']:.1%}")
        print(f"  Calibration Error:       {status['calibration']:.3f}")
        print(f"  Concept Drift Signal:    {status['concept_drift']:.2f}")
        
        print(f"\nMemory:")
        print(f"  Episodic Buffer:         {status['memory']['episodic']} events")
        print(f"  Semantic Memory:         {status['memory']['semantic']} facts")
        
        print(f"\nSelf-Model:")
        sm = status['self_model']
        print(f"  Estimated Knowledge:     {sm.get('estimated_knowledge', 0.0):.1%}")
        print(f"  Uncertainty Level:       {sm.get('uncertainty_level', 0.0):.2f}")
        print(f"  Exploration Rate:        {sm.get('exploration_rate', 0.0):.1%}")
        
        print(f"\nDevelopment:")
        print(f"  Current Stage:           {status['development_stage']}")
        print(f"  Total Reward:            {status['total_reward']:.1f}")
        
        print(f"\nAdaptation:")
        print(f"  Adaptation Events:       {status['statistics']['adaptation_events']}")
        print(f"  Drift Detections:        {status['statistics']['drift_detections']}")
        print("=" * 70 + "\n")

    def save_checkpoint(self, name: str) -> Path:
        """Save agent checkpoint."""
        return self.persistence.save_agent(self, name)

    def load_checkpoint(self, name: str) -> bool:
        """Load agent from checkpoint."""
        loaded = self.persistence.load_agent(name)
        if loaded:
            self.__dict__.update(loaded.__dict__)
            return True
        return False

    def export_metrics(self, filename: str = "metrics.json") -> Path:
        """Export metrics to JSON."""
        if not self.metrics:
            return None
        
        summary = self.metrics.summary()
        filepath = RESULTS_DIR / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return filepath


def main() -> None:
    """Main entry point for TR-Core research platform."""
    print("\n" + "=" * 70)
    print("TR-CORE v1.0 — Developmental Cognitive Agent Research Platform")
    print("=" * 70 + "\n")
    
    # Initialize agent
    print("[INIT] Creating TR-Core instance...")
    agent = TabulaRasaCore(seed=42, enable_metrics=True, verbose=True)
    env = SandboxEnvironment(seed=42)
    
    # Run baseline episode
    print("\n[EPISODE] Running baseline episode...")
    episode_stats = agent.run_episode(env, max_steps=100, checkpoint_name="baseline_ep1")
    agent.print_status()
    
    # Export metrics
    print("\n[METRICS] Exporting results...")
    metrics_file = agent.export_metrics("tr_core_metrics.json")
    print(f"  Metrics saved to: {metrics_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Episodes run:        {agent.episode_count}")
    print(f"Total steps:         {agent.step_count}")
    print(f"Final accuracy:      {agent.prediction_engine.accuracy():.1%}")
    print(f"Concept drifts:      {agent._drift_detections}")
    print(f"Adaptation events:   {agent._adaptation_events}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

