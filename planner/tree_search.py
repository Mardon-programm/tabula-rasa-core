from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TreeNode:
    state: str
    action: str | None = None
    parent: TreeNode | None = None
    children: list[TreeNode] = None
    
    predicted_state: str | None = None
    probability: float = 0.0
    reward: float = 0.0
    uncertainty: float = 0.5
    visits: int = 0
    value: float = 0.0
    
    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []


class TreeSearchPlanner:

    def __init__(
        self,
        *,
        max_depth: int = 3,
        max_children: int = 5,
    ) -> None:
        self.max_depth = max_depth
        self.max_children = max_children
        self._search_count = 0

    def plan(
        self,
        current_state: str,
        world_model: Any,
        goal_state: str | None = None,
        available_actions: list[str] | None = None,
    ) -> list[str]:
        if available_actions is None:
            available_actions = ["X", "Y", "Z"]
        
        root = TreeNode(
            state=current_state,
            action=None,
        )
        
        self._build_tree(
            root,
            world_model,
            depth=0,
            available_actions=available_actions,
        )
        
        best_path = self._find_best_path(
            root,
            goal_state=goal_state,
        )
        
        self._search_count += 1
        
        return best_path

    def _build_tree(
        self,
        node: TreeNode,
        world_model: Any,
        depth: int,
        available_actions: list[str],
    ) -> None:
        if depth >= self.max_depth:
            return
        
        if not available_actions:
            return
        
        actions_to_try = available_actions[:self.max_children]
        
        for action in actions_to_try:
            prediction = world_model.predict_next_state(
                node.state,
                action,
            )
            
            if prediction is None:
                next_state = f"{node.state}?"
                confidence = 0.0
            else:
                next_state, confidence = prediction
            
            child = TreeNode(
                state=next_state,
                action=action,
                parent=node,
                predicted_state=next_state,
                probability=confidence,
                uncertainty=1.0 - confidence,
            )
            
            node.children.append(child)
            
            self._build_tree(
                child,
                world_model,
                depth + 1,
                available_actions,
            )

    def _find_best_path(
        self,
        root: TreeNode,
        goal_state: str | None = None,
    ) -> list[str]:
        best_path: list[str] = []
        best_score = -float("inf")
        
        def dfs(
            node: TreeNode,
            path: list[str],
            score: float,
        ) -> None:
            nonlocal best_path, best_score
            
            node_score = score + node.probability - node.uncertainty
            
            if goal_state and node.state == goal_state:
                node_score += 10.0
            
            if node_score > best_score:
                best_score = node_score
                best_path = path.copy()
            
            for child in node.children:
                if child.action:
                    dfs(
                        child,
                        path + [child.action],
                        node_score,
                    )
        
        dfs(root, [], 0.0)
        
        return best_path

    def evaluate_plan(
        self,
        plan: list[str],
        current_state: str,
        world_model: Any,
    ) -> dict[str, Any]:
        state = current_state
        total_prob = 1.0
        total_uncertainty = 0.0
        
        for action in plan:
            prediction = world_model.predict_next_state(state, action)
            
            if prediction:
                state, prob = prediction
                total_prob *= prob
                total_uncertainty += (1.0 - prob)
            else:
                total_prob = 0.0
                total_uncertainty = 1.0
                break
        
        return {
            "plan": plan,
            "predicted_final_state": state,
            "success_probability": total_prob,
            "total_uncertainty": total_uncertainty,
            "length": len(plan),
        }

    def statistics(self) -> dict[str, Any]:
        return {
            "search_count": self._search_count,
            "max_depth": self.max_depth,
            "max_children": self.max_children,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "search_count": self._search_count,
            "statistics": self.statistics(),
        }
