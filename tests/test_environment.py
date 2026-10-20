"""
tests/test_environment.py
--------------------------
Tests for JoinEnvironment: valid/invalid actions, state transitions,
cardinality calculations, episode completion, deterministic seeds, and reward.

All tests use fixed seeds and small synthetic queries for repeatability.
"""

import pytest
import numpy as np

from src.environment.query_generator import generate_query, QueryInstance
from src.environment.join_environment import (
    JoinEnvironment,
    Component,
    make_state_key,
    get_valid_actions,
    _discretise_rows,
)
from src.environment.cardinality import (
    estimate_join_cardinality,
    get_connecting_selectivities,
)
from src.environment.cost_model import join_cost, step_reward


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_3table_query(seed: int = 0) -> QueryInstance:
    """Return a deterministic 3-table query for fast unit tests."""
    rng = np.random.default_rng(seed)
    return generate_query(
        rng,
        min_tables=3,
        max_tables=3,
        min_cardinality=100,
        max_cardinality=1000,
    )


def _run_random_episode(env: JoinEnvironment, rng: np.random.Generator) -> int:
    """Run one complete random episode; return total cost."""
    state_key, valid = env.reset()
    while not env.terminated:
        action = valid[rng.integers(len(valid))]
        state_key, reward, terminated, valid, info = env.step(action)
    return env.total_cost


# --------------------------------------------------------------------------
# Cardinality unit tests
# --------------------------------------------------------------------------

class TestCardinality:
    def test_basic_estimate(self):
        result = estimate_join_cardinality(1000, 500, [0.1])
        assert result == max(1, min(10_000_000, int(1000 * 500 * 0.1)))

    def test_clamped_to_max(self):
        result = estimate_join_cardinality(10_000_000, 10_000_000, [1.0])
        assert result == 10_000_000  # clipped to MAX_CARD

    def test_minimum_one_row(self):
        result = estimate_join_cardinality(1, 1, [0.0001])
        assert result >= 1

    def test_multiple_selectivities_applied(self):
        """Multiple edges → selectivities multiplied, result ≤ single-edge."""
        result_two = estimate_join_cardinality(1000, 500, [0.5, 0.5])
        result_one = estimate_join_cardinality(1000, 500, [0.5])
        assert result_two <= result_one

    def test_empty_selectivities_raises(self):
        with pytest.raises(ValueError):
            estimate_join_cardinality(100, 200, [])

    def test_get_connecting_selectivities_connected(self):
        q = _make_3table_query(0)
        # Pick any edge from the query's join graph (spanning tree guarantees at least one).
        u, v = next(iter(q.edges))
        sels = get_connecting_selectivities(
            frozenset({u}), frozenset({v}),
            q.selectivities, q.edges
        )
        assert len(sels) >= 1
        assert all(0 < s <= 1 for s in sels)

    def test_get_connecting_selectivities_disconnected(self):
        """If two sets of tables share no edge, result is empty."""
        q = _make_3table_query(0)
        # Create two artificial disjoint sets with no edge between them
        sels = get_connecting_selectivities(
            frozenset({0}), frozenset({1}),
            {},   # empty selectivities dict → no edges
            frozenset(),
        )
        assert sels == []


# --------------------------------------------------------------------------
# Cost model unit tests
# --------------------------------------------------------------------------

class TestCostModel:
    def test_join_cost_formula(self):
        assert join_cost(100, 200, 50) == 350

    def test_step_reward_non_positive(self):
        r = step_reward(1000, 500, 200)
        assert r <= 0

    def test_step_reward_scale(self):
        r = step_reward(1_000_000, 0, 0, cost_scale=1_000_000)
        assert r == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# State key
# --------------------------------------------------------------------------

class TestStateKey:
    def test_state_key_hashable(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        key, _ = env.reset()
        # Must be usable as a dict key
        d = {key: 0.0}
        assert d[key] == 0.0

    def test_state_key_order_invariant(self):
        """Two lists of components in different order must give the same key."""
        c1 = Component(frozenset({0}), 100)
        c2 = Component(frozenset({1}), 200)
        key_ab = make_state_key([c1, c2])
        key_ba = make_state_key([c2, c1])
        assert key_ab == key_ba

    def test_discretise_rows_buckets(self):
        assert _discretise_rows(5) == 0       # < 10
        assert _discretise_rows(50) == 1      # < 100
        assert _discretise_rows(500) == 2     # < 1K
        assert _discretise_rows(5000) == 3    # < 10K
        assert _discretise_rows(50_000) == 4  # < 100K
        assert _discretise_rows(500_000) == 5 # < 1M
        assert _discretise_rows(5_000_000) == 6  # ≥ 1M


# --------------------------------------------------------------------------
# JoinEnvironment: reset and valid actions
# --------------------------------------------------------------------------

class TestEnvironmentReset:
    def test_reset_returns_tuple(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        result = env.reset()
        assert len(result) == 2  # (state_key, valid_actions)

    def test_initial_components_count(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        env.reset()
        assert env.n_components == q.n_tables

    def test_initial_component_rows_match_cardinalities(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        env.reset()
        rows = {frozenset({i}): q.cardinalities[i] for i in range(q.n_tables)}
        for comp in env.components:
            assert comp.rows == rows[comp.tables]

    def test_valid_actions_non_empty_after_reset(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        _, valid = env.reset()
        assert len(valid) > 0

    def test_all_valid_actions_have_i_lt_j(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        _, valid = env.reset()
        for i, j in valid:
            assert i < j

    def test_not_terminated_after_reset(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        env.reset()
        assert not env.terminated

    def test_total_cost_zero_after_reset(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        env.reset()
        assert env.total_cost == 0

    def test_reset_restores_state(self):
        """After completing one episode, reset must restore initial state."""
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        rng = np.random.default_rng(1)
        _run_random_episode(env, rng)
        assert env.terminated
        env.reset()
        assert not env.terminated
        assert env.n_components == q.n_tables
        assert env.total_cost == 0


# --------------------------------------------------------------------------
# JoinEnvironment: stepping
# --------------------------------------------------------------------------

class TestEnvironmentStep:
    def test_step_reduces_component_count(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        _, valid = env.reset()
        action = valid[0]
        _, _, _, _, info = env.step(action)
        assert env.n_components == q.n_tables - 1

    def test_step_returns_nonpositive_reward(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        _, valid = env.reset()
        _, reward, _, _, _ = env.step(valid[0])
        assert reward <= 0

    def test_step_info_contains_expected_keys(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        _, valid = env.reset()
        _, _, _, _, info = env.step(valid[0])
        for key in ("join_cost", "rows_r", "rows_s", "rows_result", "tables_joined"):
            assert key in info, f"Missing key: {key}"

    def test_step_join_cost_matches_formula(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        _, valid = env.reset()
        _, _, _, _, info = env.step(valid[0])
        expected = info["rows_r"] + info["rows_s"] + info["rows_result"]
        assert info["join_cost"] == expected

    def test_episode_terminates_correctly(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        rng = np.random.default_rng(5)
        _run_random_episode(env, rng)
        assert env.terminated
        assert env.n_components == 1

    def test_step_after_termination_raises(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        rng = np.random.default_rng(6)
        _run_random_episode(env, rng)
        with pytest.raises(RuntimeError):
            env.step((0, 1))

    def test_total_cost_accumulates(self):
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        rng = np.random.default_rng(7)
        env.reset()
        cumulative = 0
        while not env.terminated:
            valid = get_valid_actions(env.components, env.query)
            action = valid[0]
            _, _, _, _, info = env.step(action)
            cumulative += info["join_cost"]
        assert env.total_cost == cumulative

    def test_invalid_action_raises(self):
        """Trying to join two disconnected components must raise ValueError."""
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        env.reset()
        # Find an invalid action (pair with no edge)
        components = env.components
        n = len(components)
        invalid = None
        for i in range(n):
            for j in range(i + 1, n):
                sels = get_connecting_selectivities(
                    components[i].tables, components[j].tables,
                    q.selectivities, q.edges
                )
                if not sels:
                    invalid = (i, j)
                    break
            if invalid:
                break
        if invalid is not None:
            with pytest.raises(ValueError):
                env.step(invalid)

    def test_deterministic_episodes_same_seed(self):
        """Same query + same action sequence → same total cost."""
        q = _make_3table_query(0)
        rng1 = np.random.default_rng(10)
        rng2 = np.random.default_rng(10)
        env1 = JoinEnvironment(q)
        env2 = JoinEnvironment(q)
        cost1 = _run_random_episode(env1, rng1)
        cost2 = _run_random_episode(env2, rng2)
        assert cost1 == cost2

    def test_valid_actions_from_next_state_all_connected(self):
        """All returned valid actions must be between connected components."""
        q = _make_3table_query(0)
        env = JoinEnvironment(q)
        _, valid = env.reset()
        _, _, _, next_valid, _ = env.step(valid[0])
        for i, j in next_valid:
            sels = get_connecting_selectivities(
                env.components[i].tables, env.components[j].tables,
                q.selectivities, q.edges
            )
            assert len(sels) > 0, f"Valid action ({i},{j}) has no connecting edge"


# --------------------------------------------------------------------------
# Multi-table episode tests (4, 5, 6 tables)
# --------------------------------------------------------------------------

class TestMultiTableEpisodes:
    @pytest.mark.parametrize("n_tables", [4, 5, 6])
    def test_episode_completes_for_n_tables(self, n_tables):
        rng = np.random.default_rng(n_tables * 7)
        q = generate_query(
            rng, min_tables=n_tables, max_tables=n_tables
        )
        env = JoinEnvironment(q)
        rng2 = np.random.default_rng(n_tables)
        cost = _run_random_episode(env, rng2)
        assert env.terminated
        assert env.n_components == 1
        assert cost > 0

    @pytest.mark.parametrize("n_tables", [3, 4, 5, 6])
    def test_n_steps_equals_n_tables_minus_one(self, n_tables):
        rng = np.random.default_rng(n_tables * 13)
        q = generate_query(
            rng, min_tables=n_tables, max_tables=n_tables
        )
        env = JoinEnvironment(q)
        _, valid = env.reset()
        steps = 0
        rng2 = np.random.default_rng(42)
        while not env.terminated:
            action = valid[rng2.integers(len(valid))]
            _, _, _, valid, _ = env.step(action)
            steps += 1
        assert steps == n_tables - 1
