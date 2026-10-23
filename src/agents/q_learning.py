"""
q_learning.py
-------------
Tabular Q-learning agent for database join-order optimisation.

Algorithm
---------
Standard tabular Q-learning with:
  Q(s, a) <- Q(s, a) + alpha * (TD_target - Q(s, a))

  where:
    TD_target = r                                  (if s' is terminal)
    TD_target = r + gamma * max_{a' in valid} Q(s', a')  (otherwise)

Key Features
------------
1. Valid-action masking: Invalid join actions (disconnected components) are
   strictly masked out during both exploration and exploitation. The policy
   never selects an illegal join.
2. State & Action Abstraction: Supports both:
   - "feature" (default): Query-agnostic topological and cardinality features
     from `state_representation.py`. Enables policy transfer to unseen queries.
   - "raw": Exact table-frozenset component keys.
3. Terminal handling: No bootstrapping on terminal states.
4. Epsilon-greedy exploration with configurable multiplicative or linear decay.
5. Model serialization: Save and load complete Q-table, configuration, and training state
   in human-readable, portable JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.environment.join_environment import JoinEnvironment


# ---------------------------------------------------------------------------
# Helpers for JSON serialization of nested tuple keys
# ---------------------------------------------------------------------------

def _to_json_friendly(obj: Any) -> Any:
    """Recursively convert tuples/sets to lists for JSON serialization."""
    if isinstance(obj, (tuple, list, set, frozenset)):
        return [_to_json_friendly(x) for x in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    return obj


def _from_json_friendly(obj: Any) -> Any:
    """Recursively convert lists back to tuples for hashable Q-table keys."""
    if isinstance(obj, list):
        return tuple(_from_from_list(x) for x in obj)
    return obj


def _from_from_list(x: Any) -> Any:
    if isinstance(x, list):
        return tuple(_from_from_list(y) for y in x)
    return x


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QLearningConfig:
    """Hyperparameters for tabular Q-learning."""

    alpha: float = 0.1
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    decay_schedule: str = "multiplicative"  # "multiplicative" or "linear"
    decay_episodes: int = 2500
    state_mode: str = "feature"     # "feature" or "raw"
    seed: Optional[int] = None
    # When all valid actions are unseen (Q=0), fall back to cheapest one-step
    # join cost instead of random pick. Fixes the optimism-in-noise problem.
    greedy_fallback: bool = False


# ---------------------------------------------------------------------------
# QLearningAgent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-learning agent with valid-action masking for query join ordering.
    """

    def __init__(self, config: Optional[QLearningConfig] = None):
        self.config = config or QLearningConfig()
        if self.config.state_mode not in ("raw", "feature"):
            raise ValueError(
                f"state_mode must be 'raw' or 'feature', got {self.config.state_mode!r}"
            )

        self.rng = np.random.default_rng(self.config.seed)
        self.epsilon = self.config.epsilon_start
        self.episodes_trained = 0

        # Q-table: maps (state_key, action_key) -> float
        self.q_table: Dict[Tuple[tuple, tuple], float] = {}
        # Track which (state_key, action_key) pairs have been trained on
        self._seen_pairs: set = set()

    # ------------------------------------------------------------------
    # Key extraction helpers
    # ------------------------------------------------------------------

    def get_state_key(self, env: JoinEnvironment) -> tuple:
        """Extract the state key matching agent's state_mode."""
        if self.config.state_mode == "feature":
            return env.get_feature_state_key()
        return env.current_state_key

    def get_action_key(self, env: JoinEnvironment, action: Tuple[int, int]) -> tuple:
        """Extract the action key matching agent's state_mode."""
        if self.config.state_mode == "feature":
            return env.get_action_feature(action)
        return action

    # ------------------------------------------------------------------
    # Q-value access
    # ------------------------------------------------------------------

    def get_q(self, state_key: tuple, action_key: tuple) -> float:
        """Return Q(s, a); defaults to 0.0 if unvisited."""
        return self.q_table.get((state_key, action_key), 0.0)

    def set_q(self, state_key: tuple, action_key: tuple, value: float) -> None:
        """Set Q(s, a)."""
        self.q_table[(state_key, action_key)] = float(value)

    # ------------------------------------------------------------------
    # Action selection with valid-action masking
    # ------------------------------------------------------------------

    def select_action(
        self,
        env: JoinEnvironment,
        explore: bool = True,
    ) -> Tuple[int, int]:
        """
        Epsilon-greedy action selection with valid-action masking.

        If greedy_fallback=True and ALL valid actions are unseen (Q=0),
        we pick the cheapest one-step join instead of a random tie — this
        avoids the optimism trap where Q=0 beats all trained negatives.
        """
        valid_actions = env.valid_actions
        if not valid_actions:
            raise RuntimeError("Cannot select action: no valid join actions available.")

        if explore and self.rng.random() < self.epsilon:
            return valid_actions[int(self.rng.integers(len(valid_actions)))]

        state_key = self.get_state_key(env)
        best_q = -float("inf")
        best_actions = []

        for action in valid_actions:
            action_key = self.get_action_key(env, action)
            q_val = self.get_q(state_key, action_key)
            if q_val > best_q:
                best_q = q_val
                best_actions = [action]
            elif q_val == best_q:
                best_actions.append(action)

        # If all Q-values are 0 (all unseen) and fallback is enabled,
        # break ties by cheapest one-step join cost instead of random.
        if self.config.greedy_fallback and best_q == 0.0 and len(best_actions) > 1:
            return self._greedy_fallback_action(env, best_actions)

        if len(best_actions) == 1:
            return best_actions[0]
        return best_actions[int(self.rng.integers(len(best_actions)))]

    def _greedy_fallback_action(self, env: JoinEnvironment, candidates: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Among candidate actions, return the one with lowest immediate join cost."""
        from src.environment.cardinality import get_connecting_selectivities, estimate_join_cardinality
        from src.environment.cost_model import join_cost
        components = env.components
        best_action, best_cost = candidates[0], float("inf")
        for act in candidates:
            i, j = act
            sels = get_connecting_selectivities(
                components[i].tables, components[j].tables,
                env.query.selectivities, env.query.edges,
            )
            rows_r = estimate_join_cardinality(components[i].rows, components[j].rows, sels)
            c = join_cost(components[i].rows, components[j].rows, rows_r)
            if c < best_cost:
                best_cost = c
                best_action = act
        return best_action

    # ------------------------------------------------------------------
    # Q-Learning update
    # ------------------------------------------------------------------

    def update(
        self,
        state_key: tuple,
        action_key: tuple,
        reward: float,
        next_env: Optional[JoinEnvironment],
        terminated: bool,
    ) -> float:
        """
        Perform one tabular Q-learning update step.

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_{a'} Q(s', a') - Q(s, a)]
        If terminated: max_{a'} Q(s', a') is strictly 0 (no bootstrap).

        Returns
        -------
        td_error : float
        """
        current_q = self.get_q(state_key, action_key)

        if terminated or next_env is None:
            td_target = reward
        else:
            next_state_key = self.get_state_key(next_env)
            next_valid = next_env.valid_actions
            if next_valid:
                max_next_q = max(
                    self.get_q(next_state_key, self.get_action_key(next_env, a))
                    for a in next_valid
                )
            else:
                max_next_q = 0.0
            td_target = reward + self.config.gamma * max_next_q

        td_error = td_target - current_q
        new_q = current_q + self.config.alpha * td_error
        self.set_q(state_key, action_key, new_q)
        self._seen_pairs.add((state_key, action_key))
        return td_error

    # ------------------------------------------------------------------
    # Episode execution & Training
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> float:
        """Apply one decay step to epsilon, bounded by epsilon_min."""
        if self.config.decay_schedule == "linear" and self.config.decay_episodes > 0:
            step_delta = (
                self.config.epsilon_start - self.config.epsilon_min
            ) / float(self.config.decay_episodes)
            self.epsilon = max(self.config.epsilon_min, self.epsilon - step_delta)
        else:
            self.epsilon = max(
                self.config.epsilon_min,
                self.epsilon * self.config.epsilon_decay,
            )
        return self.epsilon

    def train_episode(self, env: JoinEnvironment) -> Dict[str, Any]:
        """
        Run one complete training episode in `env`.

        Updates the Q-table at each step and decays epsilon at episode end.

        Returns
        -------
        dict with episode metrics:
          - return: cumulative reward
          - total_cost: modeled query execution cost
          - steps: number of join steps taken
          - epsilon: exploration rate after episode decay
          - mean_td_error: average TD error over all steps
        """
        # Configure env state mode to match agent configuration
        env.state_mode = self.config.state_mode
        env.reset()

        total_reward = 0.0
        td_errors = []
        steps = 0

        while not env.terminated:
            state_key = self.get_state_key(env)
            action = self.select_action(env, explore=True)
            action_key = self.get_action_key(env, action)

            _, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            steps += 1

            td_err = self.update(
                state_key=state_key,
                action_key=action_key,
                reward=reward,
                next_env=None if terminated else env,
                terminated=terminated,
            )
            td_errors.append(abs(td_err))

        self.episodes_trained += 1
        eps = self.decay_epsilon()

        return {
            "episode": self.episodes_trained,
            "return": total_reward,
            "total_cost": env.total_cost,
            "steps": steps,
            "epsilon": eps,
            "mean_td_error": float(np.mean(td_errors)) if td_errors else 0.0,
        }

    def evaluate_episode(self, env: JoinEnvironment) -> Dict[str, Any]:
        """
        Run one evaluation episode with purely greedy action selection (explore=False).
        Does not update the Q-table or decay epsilon.

        Returns
        -------
        dict with total_cost, return, steps, and plan.
        """
        env.state_mode = self.config.state_mode
        env.reset()

        total_reward = 0.0
        plan = []

        while not env.terminated:
            action = self.select_action(env, explore=False)
            plan.append(action)
            _, reward, _, _, _ = env.step(action)
            total_reward += reward

        return {
            "total_cost": env.total_cost,
            "return": total_reward,
            "steps": len(plan),
            "plan": plan,
        }

    def evaluate_episode_with_info(
        self, env: JoinEnvironment
    ) -> Dict[str, Any]:
        """
        Like evaluate_episode but also records:
          - peak_intermediate_cardinality: max intermediate row count seen
          - unseen_action_count: greedy actions chosen from unseen (s,a) pairs
          - action_q_values: Q-values queried during the episode

        Does not modify the Q-table or epsilon.
        """
        env.state_mode = self.config.state_mode
        env.reset()

        total_reward = 0.0
        plan = []
        peak_rows = 0
        unseen_count = 0
        q_values_used: List[float] = []

        while not env.terminated:
            state_key = self.get_state_key(env)

            # Track Q-values and whether this is an unseen pair
            action = self.select_action(env, explore=False)
            action_key = self.get_action_key(env, action)
            q_val = self.get_q(state_key, action_key)
            q_values_used.append(q_val)
            if (state_key, action_key) not in self._seen_pairs:
                unseen_count += 1

            plan.append(action)
            _, reward, _, _, info = env.step(action)
            total_reward += reward

            # Track peak intermediate cardinality
            peak_rows = max(peak_rows, info.get("rows_result", 0))

        return {
            "total_cost": env.total_cost,
            "return": total_reward,
            "steps": len(plan),
            "plan": plan,
            "peak_intermediate_cardinality": peak_rows,
            "unseen_action_count": unseen_count,
            "q_values_used": q_values_used,
        }

    @property
    def seen_pairs(self) -> set:
        """Read-only view of (state_key, action_key) pairs updated during training."""
        return self._seen_pairs

    # ------------------------------------------------------------------
    # Model Save / Load (JSON)
    # ------------------------------------------------------------------

    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save agent Q-table, hyperparameters, and training state to a JSON file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert entries to JSON-safe lists
        entries = [
            [_to_json_friendly(s_k), _to_json_friendly(a_k), q_val]
            for (s_k, a_k), q_val in self.q_table.items()
        ]

        payload = {
            "version": 1,
            "config": asdict(self.config),
            "episodes_trained": self.episodes_trained,
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "entries": entries,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> QLearningAgent:
        """
        Load an agent from a saved JSON file.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Agent checkpoint not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        config_dict = payload.get("config", {})
        config = QLearningConfig(**config_dict)
        agent = cls(config=config)

        agent.episodes_trained = payload.get("episodes_trained", 0)
        agent.epsilon = payload.get("epsilon", config.epsilon_min)

        # Reconstruct Q-table with hashable tuple keys
        agent.q_table = {}
        for entry in payload.get("entries", []):
            s_k_list, a_k_list, q_val = entry
            s_k = _from_json_friendly(s_k_list)
            a_k = _from_json_friendly(a_k_list)
            agent.q_table[(s_k, a_k)] = float(q_val)

        return agent
