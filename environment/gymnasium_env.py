#!/usr/bin/env python3
"""
Gymnasium-Compatible Environment Wrapper for TR-Core Attribute Tasks.

This module wraps the attribute task environments (V3, V4) in a standard
Gymnasium interface, enabling use with standard RL baselines and benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gymnasium import Env, spaces


@dataclass
class AttributeTaskConfig:
    """Configuration for attribute task environment."""
    attributes: List[str]
    domains: Dict[str, List[str]]
    actions: List[str]
    oracle_rules: List[Dict[str, Any]]
    max_steps: int = 1
    reward_success: float = 1.0
    reward_failure: float = 0.0
    seed: Optional[int] = None


class AttributeTaskEnv(Env):
    """
    Gymnasium environment for TR-Core attribute tasks.
    
    The environment presents a context (combination of attribute values)
    and the agent must choose an action. Success is determined by the
    oracle rules (priority-based matching).
    
    Observation space: Dict of categorical attributes
    Action space: Discrete actions
    Reward: 1.0 for success, 0.0 for failure (configurable)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        config: AttributeTaskConfig,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        
        # Build attribute to index mappings
        self.attr_to_idx = {attr: i for i, attr in enumerate(config.attributes)}
        self.attr_domains = config.domains
        self.attr_sizes = [len(config.domains[attr]) for attr in config.attributes]
        self.n_actions = len(config.actions)
        self.action_to_idx = {a: i for i, a in enumerate(config.actions)}
        self.idx_to_action = config.actions
        
        # Oracle rules: list of (condition_dict, action_idx)
        self.oracle_rules = []
        for rule in config.oracle_rules:
            cond = rule["condition"]
            action = rule["action"]
            self.oracle_rules.append((cond, self.action_to_idx[action]))
        
        # Observation space: Dict of Discrete spaces
        self.observation_space = spaces.Dict({
            attr: spaces.Discrete(len(config.domains[attr]))
            for attr in config.attributes
        })
        
        # Action space
        self.action_space = spaces.Discrete(self.n_actions)
        
        # State
        self.current_context: Optional[Dict[str, int]] = None
        self.step_count = 0
        self.rng = np.random.default_rng(config.seed)
        
        # Precompute all possible contexts for sampling
        self._all_contexts = self._generate_all_contexts()
        
    def _generate_all_contexts(self) -> List[Dict[str, int]]:
        """Generate all possible context combinations."""
        import itertools
        contexts = []
        domain_lists = [list(range(len(self.attr_domains[attr]))) for attr in self.config.attributes]
        for combo in itertools.product(*domain_lists):
            contexts.append({attr: combo[i] for i, attr in enumerate(self.config.attributes)})
        return contexts
    
    def _sample_context(self) -> Dict[str, int]:
        """Sample a random context."""
        return self.rng.choice(self._all_contexts)
    
    def _evaluate_oracle(self, context: Dict[str, int], action: int) -> bool:
        """Evaluate oracle rules for given context and action."""
        for condition, rule_action in self.oracle_rules:
            matches = all(context.get(attr) == self.attr_domains[attr].index(val) 
                         for attr, val in condition.items())
            if matches:
                return action == rule_action
        # Default: no rule matches, failure
        return False
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.current_context = self._sample_context()
        self.step_count = 0
        
        obs = self._context_to_obs(self.current_context)
        info = {"context": self.current_context.copy()}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        if self.current_context is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Evaluate oracle
        success = self._evaluate_oracle(self.current_context, action)
        
        # Reward
        reward = self.config.reward_success if success else self.config.reward_failure
        
        # Episode ends after one step (contextual bandit)
        self.step_count += 1
        terminated = self.step_count >= self.config.max_steps
        truncated = False
        
        # Next context (for multi-step, would sample new; for bandit, same)
        next_context = self._sample_context() if not terminated else self.current_context
        self.current_context = next_context
        
        obs = self._context_to_obs(self.current_context)
        info = {
            "context": self.current_context.copy(),
            "success": success,
            "oracle_action": self._get_oracle_action(self.current_context),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_oracle_action(self, context: Dict[str, int]) -> int:
        """Get the oracle's action for a context."""
        for condition, rule_action in self.oracle_rules:
            matches = all(context.get(attr) == self.attr_domains[attr].index(val) 
                         for attr, val in condition.items())
            if matches:
                return rule_action
        return 0  # Default
    
    def _context_to_obs(self, context: Dict[str, int]) -> Dict[str, np.int64]:
        """Convert context dict to observation dict."""
        return {k: np.int64(v) for k, v in context.items()}
    
    def render(self) -> Optional[np.ndarray]:
        """Render environment."""
        if self.render_mode == "human":
            ctx_str = ", ".join(f"{k}={self.attr_domains[k][v]}" for k, v in self.current_context.items())
            print(f"Context: {{{ctx_str}}}")
        return None
    
    def close(self) -> None:
        """Close environment."""
        pass
    
    def get_oracle_policy(self) -> Dict[Tuple[int, ...], int]:
        """Get full oracle policy mapping contexts to actions."""
        policy = {}
        for ctx in self._all_contexts:
            ctx_tuple = tuple(ctx[attr] for attr in self.config.attributes)
            policy[ctx_tuple] = self._get_oracle_action(ctx)
        return policy


def make_attribute_task_v3(**kwargs) -> AttributeTaskEnv:
    """Factory for V3 environment."""
    from environment.attribute_tasks_v3 import (
        V3_ATTRIBUTES, ATTRIBUTE_DOMAINS_V3, ACTIONS_V3, ORACLE_RULES_V3
    )
    config = AttributeTaskConfig(
        attributes=list(V3_ATTRIBUTES),
        domains=ATTRIBUTE_DOMAINS_V3,
        actions=ACTIONS_V3,
        oracle_rules=ORACLE_RULES_V3,
        **kwargs
    )
    return AttributeTaskEnv(config)


def make_attribute_task_v4(**kwargs) -> AttributeTaskEnv:
    """Factory for V4 environment."""
    from environment.attribute_tasks_v4 import (
        V4_ATTRIBUTES, ATTRIBUTE_DOMAINS_V4, ACTIONS_V4, ORACLE_RULES_V4
    )
    config = AttributeTaskConfig(
        attributes=list(V4_ATTRIBUTES),
        domains=ATTRIBUTE_DOMAINS_V4,
        actions=ACTIONS_V4,
        oracle_rules=ORACLE_RULES_V4,
        **kwargs
    )
    return AttributeTaskEnv(config)


# Registration for gymnasium.make()
def register_envs() -> None:
    """Register environments with Gymnasium."""
    from gymnasium.envs.registration import register
    
    register(
        id="TRCore-AttributeTaskV3-v0",
        entry_point="environment.gymnasium_env:make_attribute_task_v3",
        max_episode_steps=1,
        kwargs={},
    )
    
    register(
        id="TRCore-AttributeTaskV4-v0",
        entry_point="environment.gymnasium_env:make_attribute_task_v4",
        max_episode_steps=1,
        kwargs={},
    )


if __name__ == "__main__":
    # Test the environment
    env = make_attribute_task_v3(seed=42)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of contexts: {len(env._all_contexts)}")
    
    obs, info = env.reset()
    print(f"Initial obs: {obs}")
    print(f"Initial info: {info}")
    
    for action in range(env.n_actions):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action {action} ({env.idx_to_action[action]}): reward={reward}, success={info['success']}")
    
    env.close()