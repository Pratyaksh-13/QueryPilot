"""
tests/test_q_learning.py
------------------------
Unit and controlled-experiment tests for tabular Q-learning (Day 4).

Verifies:
  1. Valid-action masking: Agent strictly selects valid actions.
  2. Q-value changes due to rewards: Q(s, a) shifts appropriately with TD updates.
  3. Terminal handling: Terminal steps do NOT bootstrap from future Q-values.
  4. Non-terminal bootstrapping: Non-terminal steps include gamma * max Q(s', a').
  5. Epsilon decay: Epsilon decreases per episode and floors at epsilon_min.
  6. Model persistence: Save / load round-trip preserves Q-table and greedy policy.
  7. Controlled experiment: A short training run updates Q-values and learns.
"""

import pytest
import numpy as np

from src.environment.query_generator import QueryInstance, generate_query
from src.environment.join_environment import JoinEnvironment
from src.agents.q_learning import QLearningAgent, QLearningConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_query() -> QueryInstance:
    """Deterministic 3-table chain: T0 - T1 - T2."""
    return QueryInstance(
        n_tables=3,
        cardinalities=(100, 500, 1000),
        selectivities={(0, 1): 0.05, (1, 2): 0.20},
        edges=frozenset({(0, 1), (1, 2)}),
    )


@pytest.fixture
def env(small_query) -> JoinEnvironment:
    return JoinEnvironment(small_query, state_mode="feature")


# ---------------------------------------------------------------------------
# Action Selection & Masking
# ---------------------------------------------------------------------------

class TestActionSelection:
    def test_valid_action_masking_always_respected(self, env):
        agent = QLearningAgent(QLearningConfig(epsilon_start=1.0, seed=42))
        env.reset()

        # Run 50 action selections; all must be in env.valid_actions
        for _ in range(50):
            action = agent.select_action(env, explore=True)
            assert action in env.valid_actions
            # Disconnected pair (0, 2) must never be chosen
            assert action != (0, 2)

    def test_greedy_selection_picks_highest_q(self, env):
        agent = QLearningAgent(QLearningConfig(state_mode="feature"))
        env.reset()

        state_key = agent.get_state_key(env)
        action_a = (0, 1)
        action_b = (1, 2)
        key_a = agent.get_action_key(env, action_a)
        key_b = agent.get_action_key(env, action_b)

        # Set distinct Q-values
        agent.set_q(state_key, key_a, -10.0)
        agent.set_q(state_key, key_b, -2.0)

        # Purely greedy selection must pick action_b (Q = -2 > -10)
        chosen = agent.select_action(env, explore=False)
        assert chosen == action_b


# ---------------------------------------------------------------------------
# Q-Learning Updates
# ---------------------------------------------------------------------------

class TestQLearningUpdates:
    def test_q_values_change_due_to_rewards(self, env):
        """Verify Q-values change from default 0.0 upon receiving rewards."""
        agent = QLearningAgent(QLearningConfig(alpha=0.5, gamma=1.0, state_mode="feature"))
        env.reset()

        state_key = agent.get_state_key(env)
        action = (0, 1)
        action_key = agent.get_action_key(env, action)

        # Before update, default Q is 0.0
        assert agent.get_q(state_key, action_key) == 0.0

        # Simulate update with reward -0.8 on terminal step
        td_err = agent.update(
            state_key=state_key,
            action_key=action_key,
            reward=-0.8,
            next_env=None,
            terminated=True,
        )

        # Expected: Q <- 0 + 0.5 * (-0.8 - 0) = -0.4
        new_q = agent.get_q(state_key, action_key)
        assert pytest.approx(new_q, rel=1e-5) == -0.4
        assert pytest.approx(td_err, rel=1e-5) == -0.8

    def test_terminal_update_no_bootstrap(self, env):
        """When terminated=True, target must be r even if next_env has high Q."""
        agent = QLearningAgent(QLearningConfig(alpha=1.0, gamma=1.0, state_mode="feature"))
        env.reset()

        state_key = agent.get_state_key(env)
        action_key = agent.get_action_key(env, (0, 1))

        # Update with terminated=True
        agent.update(
            state_key=state_key,
            action_key=action_key,
            reward=-0.25,
            next_env=env,  # even if passed, terminated=True takes precedence
            terminated=True,
        )

        assert pytest.approx(agent.get_q(state_key, action_key), rel=1e-5) == -0.25

    def test_non_terminal_update_includes_max_next_q(self, env):
        agent = QLearningAgent(QLearningConfig(alpha=1.0, gamma=0.9, state_mode="feature"))
        env.reset()

        s0 = agent.get_state_key(env)
        a0 = (0, 1)
        a0_key = agent.get_action_key(env, a0)

        # Step environment once
        env.step(a0)
        s1 = agent.get_state_key(env)

        # Set known Q-values in next state s1
        next_actions = env.valid_actions
        assert len(next_actions) > 0
        best_next_a = next_actions[0]
        best_next_k = agent.get_action_key(env, best_next_a)
        agent.set_q(s1, best_next_k, -0.5)

        # Update s0: target = r + gamma * max_next_q = -0.1 + 0.9 * (-0.5) = -0.55
        agent.update(
            state_key=s0,
            action_key=a0_key,
            reward=-0.1,
            next_env=env,
            terminated=False,
        )

        assert pytest.approx(agent.get_q(s0, a0_key), rel=1e-5) == -0.55


# ---------------------------------------------------------------------------
# Epsilon Decay
# ---------------------------------------------------------------------------

class TestEpsilonDecay:
    def test_epsilon_decays_and_floors(self):
        config = QLearningConfig(
            epsilon_start=1.0,
            epsilon_decay=0.5,
            epsilon_min=0.1,
        )
        agent = QLearningAgent(config)
        assert agent.epsilon == 1.0

        eps1 = agent.decay_epsilon()
        assert pytest.approx(eps1) == 0.5

        eps2 = agent.decay_epsilon()
        assert pytest.approx(eps2) == 0.25

        eps3 = agent.decay_epsilon()
        assert pytest.approx(eps3) == 0.125

        eps4 = agent.decay_epsilon()
        assert pytest.approx(eps4) == 0.1  # Floored at epsilon_min

        eps5 = agent.decay_epsilon()
        assert pytest.approx(eps5) == 0.1

    def test_linear_epsilon_decay(self):
        config = QLearningConfig(
            epsilon_start=1.0,
            epsilon_min=0.05,
            decay_schedule="linear",
            decay_episodes=10,
        )
        agent = QLearningAgent(config)
        assert agent.epsilon == 1.0

        for _ in range(5):
            agent.decay_epsilon()
        # Halfway through 10 episodes: 1.0 - 5 * (0.95/10) = 0.525
        assert pytest.approx(agent.epsilon) == 0.525

        for _ in range(5):
            agent.decay_epsilon()
        # At episode 10: reached epsilon_min 0.05
        assert pytest.approx(agent.epsilon) == 0.05

        # Further steps remain floored at epsilon_min
        agent.decay_epsilon()
        assert pytest.approx(agent.epsilon) == 0.05


# ---------------------------------------------------------------------------
# Model Serialization (Save / Load)
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_load_roundtrip(self, env, tmp_path):
        save_file = tmp_path / "q_agent.json"

        config = QLearningConfig(
            alpha=0.2,
            gamma=0.95,
            epsilon_start=0.8,
            epsilon_min=0.05,
            state_mode="feature",
            seed=123,
        )
        agent = QLearningAgent(config)

        # Run 5 training episodes to populate Q-table
        for _ in range(5):
            agent.train_episode(env)

        assert len(agent.q_table) > 0
        assert agent.episodes_trained == 5

        # Save to disk
        agent.save(save_file)
        assert save_file.is_file()

        # Load back
        loaded = QLearningAgent.load(save_file)
        assert loaded.config.alpha == 0.2
        assert loaded.config.gamma == 0.95
        assert loaded.config.state_mode == "feature"
        assert loaded.episodes_trained == 5
        assert pytest.approx(loaded.epsilon) == agent.epsilon
        assert len(loaded.q_table) == len(agent.q_table)

        # Compare all Q-values
        for key, val in agent.q_table.items():
            assert pytest.approx(loaded.get_q(key[0], key[1])) == val

        # Evaluate both agents: greedy decisions must match identically
        eval1 = agent.evaluate_episode(env)
        eval2 = loaded.evaluate_episode(env)
        assert eval1["plan"] == eval2["plan"]
        assert eval1["total_cost"] == eval2["total_cost"]


# ---------------------------------------------------------------------------
# Controlled Training Experiment
# ---------------------------------------------------------------------------

class TestControlledTrainingExperiment:
    def test_controlled_training_updates_q_table_and_runs(self, env):
        """
        Run a short training experiment (30 episodes) on a query.
        Verify that:
          1. Q-table grows and stores non-zero entries.
          2. Episode returns and costs are logged properly.
          3. Greedy evaluation produces a valid plan and positive cost.
        """
        config = QLearningConfig(
            alpha=0.1,
            gamma=1.0,
            epsilon_start=1.0,
            epsilon_decay=0.90,
            epsilon_min=0.05,
            state_mode="feature",
            seed=42,
        )
        agent = QLearningAgent(config)

        returns = []
        costs = []
        for _ in range(30):
            metrics = agent.train_episode(env)
            returns.append(metrics["return"])
            costs.append(metrics["total_cost"])

        assert agent.episodes_trained == 30
        assert len(agent.q_table) > 0
        assert agent.epsilon < 0.1  # Epsilon decayed

        # Evaluate greedy policy
        eval_metrics = agent.evaluate_episode(env)
        assert eval_metrics["total_cost"] > 0
        assert len(eval_metrics["plan"]) == env.query.n_tables - 1
