"""
tests/test_state_representation.py
----------------------------------
Tests for the feature-based state and action representations.

Verifies:
  1. Hashability: state and action features can be dictionary / Q-table keys.
  2. Order invariance: component ordering in list does not alter state key.
  3. Graph structure awareness: degrees reflect actual join graph topology.
  4. Query generalisation: isomorphic queries with different table indices produce
     identical state and action feature representations.
  5. Action feature symmetry: joining R with S produces the identical action feature
     as joining S with R.
  6. Environment integration: JoinEnvironment(state_mode="feature") runs full episodes.
"""

import pytest
import numpy as np

from src.environment.query_generator import QueryInstance, generate_query
from src.environment.join_environment import (
    JoinEnvironment,
    Component,
    _discretise_rows,
)
from src.environment.state_representation import (
    compute_component_degree,
    discretise_selectivity,
    make_feature_state_key,
    compute_action_feature,
)


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

def _make_chain_3table_query() -> QueryInstance:
    """Chain graph: T0 - T1 - T2."""
    return QueryInstance(
        n_tables=3,
        cardinalities=(100, 500, 1000),
        selectivities={(0, 1): 0.1, (1, 2): 0.2},
        edges=frozenset({(0, 1), (1, 2)}),
    )


def _make_isomorphic_shifted_query() -> QueryInstance:
    """
    Same chain graph structure and cardinalities, but tables shifted:
    T1 is now at end, T0 in middle: T2 (100) - T0 (500) - T1 (1000).
    Under raw indices this would differ completely, but under feature
    abstraction it should produce identical initial state features!
    """
    return QueryInstance(
        n_tables=3,
        cardinalities=(500, 1000, 100),  # T0=500, T1=1000, T2=100
        selectivities={(0, 2): 0.1, (0, 1): 0.2},
        edges=frozenset({(0, 2), (0, 1)}),
    )


# ---------------------------------------------------------------------------
# Selectivity and Degree tests
# ---------------------------------------------------------------------------

class TestSelectivityAndDegree:
    def test_discretise_selectivity_bins(self):
        assert discretise_selectivity(0.01) == 0  # < 0.05
        assert discretise_selectivity(0.049) == 0
        assert discretise_selectivity(0.05) == 1   # 0.05 <= sel < 0.15
        assert discretise_selectivity(0.14) == 1
        assert discretise_selectivity(0.15) == 2   # 0.15 <= sel < 0.30
        assert discretise_selectivity(0.29) == 2
        assert discretise_selectivity(0.30) == 3   # >= 0.30
        assert discretise_selectivity(0.80) == 3

    def test_component_degrees_in_chain(self):
        q = _make_chain_3table_query()
        components = [
            Component(frozenset({0}), 100),
            Component(frozenset({1}), 500),
            Component(frozenset({2}), 1000),
        ]
        # T0 is connected to T1 (degree 1)
        # T1 is connected to T0 and T2 (degree 2)
        # T2 is connected to T1 (degree 1)
        assert compute_component_degree(0, components, q) == 1
        assert compute_component_degree(1, components, q) == 2
        assert compute_component_degree(2, components, q) == 1


# ---------------------------------------------------------------------------
# Feature State Key tests
# ---------------------------------------------------------------------------

class TestFeatureStateKey:
    def test_hashable(self):
        q = _make_chain_3table_query()
        components = [
            Component(frozenset({0}), 100),
            Component(frozenset({1}), 500),
            Component(frozenset({2}), 1000),
        ]
        key = make_feature_state_key(components, q, _discretise_rows)
        d = {key: 42.0}
        assert d[key] == 42.0

    def test_list_order_invariance(self):
        q = _make_chain_3table_query()
        c0 = Component(frozenset({0}), 100)
        c1 = Component(frozenset({1}), 500)
        c2 = Component(frozenset({2}), 1000)

        key1 = make_feature_state_key([c0, c1, c2], q, _discretise_rows)
        key2 = make_feature_state_key([c2, c0, c1], q, _discretise_rows)
        assert key1 == key2

    def test_cross_query_generalisation(self):
        """Isomorphic queries with different table indices yield identical state keys."""
        q1 = _make_chain_3table_query()
        q2 = _make_isomorphic_shifted_query()

        env1 = JoinEnvironment(q1, state_mode="feature")
        env2 = JoinEnvironment(q2, state_mode="feature")

        k1, _ = env1.reset()
        k2, _ = env2.reset()

        assert k1 == k2, "Feature abstraction should match across isomorphic query graphs"


# ---------------------------------------------------------------------------
# Action Feature tests
# ---------------------------------------------------------------------------

class TestActionFeature:
    def test_action_feature_hashable(self):
        q = _make_chain_3table_query()
        env = JoinEnvironment(q)
        env.reset()
        feat = env.get_action_feature((0, 1))
        d = {feat: -1.5}
        assert d[feat] == -1.5

    def test_action_feature_canonical_symmetry(self):
        """Action feature does not depend on whether component order is swapped."""
        q = _make_chain_3table_query()
        c0 = Component(frozenset({0}), 100)
        c1 = Component(frozenset({1}), 500)
        c2 = Component(frozenset({2}), 1000)

        # In list [c0, c1, c2], joining (0, 1) joins T0 and T1
        # In list [c1, c0, c2], joining (0, 1) joins T1 and T0
        f1 = compute_action_feature((0, 1), [c0, c1, c2], q, _discretise_rows)
        f2 = compute_action_feature((0, 1), [c1, c0, c2], q, _discretise_rows)
        assert f1 == f2

    def test_disconnected_action_raises(self):
        q = _make_chain_3table_query()
        c0 = Component(frozenset({0}), 100)
        c1 = Component(frozenset({1}), 500)
        c2 = Component(frozenset({2}), 1000)
        # T0 and T2 are disconnected in this chain graph
        with pytest.raises(ValueError, match="Cannot compute action feature for disconnected pair"):
            compute_action_feature((0, 2), [c0, c1, c2], q, _discretise_rows)

    def test_valid_action_features_helper(self):
        q = _make_chain_3table_query()
        env = JoinEnvironment(q)
        _, valid = env.reset()
        action_feats = env.get_valid_action_features()
        assert len(action_feats) == len(valid)
        for (i, j), feat in action_feats:
            assert i < j
            assert isinstance(feat, tuple)


# ---------------------------------------------------------------------------
# Environment Integration with state_mode="feature"
# ---------------------------------------------------------------------------

class TestEnvironmentFeatureMode:
    def test_invalid_mode_raises(self):
        q = _make_chain_3table_query()
        with pytest.raises(ValueError, match="state_mode must be 'raw' or 'feature'"):
            JoinEnvironment(q, state_mode="unknown")

    def test_feature_mode_full_episode(self):
        rng = np.random.default_rng(42)
        q = generate_query(rng=rng, min_tables=4, max_tables=4)
        env = JoinEnvironment(q, state_mode="feature")

        state, valid = env.reset()
        assert isinstance(state, tuple)
        # Format: (n_components, n_edges, tuple_of_comp_feats)
        assert len(state) == 3
        assert state[0] == 4  # 4 components

        steps = 0
        while not env.terminated:
            action = valid[0]
            next_state, reward, term, valid, info = env.step(action)
            assert isinstance(next_state, tuple)
            steps += 1

        assert steps == 3
        assert env.terminated
        assert env.total_cost > 0
