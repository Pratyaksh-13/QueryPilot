"""
tests/test_query_generator.py
-----------------------------
Tests for the synthetic query generator.

All tests use fixed seeds for reproducibility.
"""

import numpy as np
import pytest

from src.environment.query_generator import (
    QueryInstance,
    generate_query,
    make_query_batch,
    _random_spanning_tree,
)


# --------------------------------------------------------------------------
# Helper: check graph connectivity via BFS
# --------------------------------------------------------------------------

def _is_connected(n: int, edges) -> bool:
    from collections import deque
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    visited = set()
    queue = deque([0])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adj[node] - visited)
    return visited == set(range(n))


# --------------------------------------------------------------------------
# Spanning-tree tests
# --------------------------------------------------------------------------

class TestSpanningTree:
    def test_spanning_tree_is_connected(self):
        for n in range(2, 8):
            rng = np.random.default_rng(0)
            edges = _random_spanning_tree(n, rng)
            assert _is_connected(n, edges), f"Spanning tree not connected for n={n}"

    def test_spanning_tree_has_n_minus_one_edges(self):
        for n in range(2, 8):
            rng = np.random.default_rng(1)
            edges = _random_spanning_tree(n, rng)
            assert len(edges) == n - 1

    def test_spanning_tree_edges_in_range(self):
        n = 5
        rng = np.random.default_rng(2)
        edges = _random_spanning_tree(n, rng)
        for u, v in edges:
            assert 0 <= u < n and 0 <= v < n
            assert u < v  # canonical key convention


# --------------------------------------------------------------------------
# QueryInstance generation
# --------------------------------------------------------------------------

class TestGenerateQuery:
    def test_returns_query_instance(self):
        rng = np.random.default_rng(42)
        q = generate_query(rng)
        assert isinstance(q, QueryInstance)

    def test_table_count_in_range(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            q = generate_query(rng, min_tables=3, max_tables=6)
            assert 3 <= q.n_tables <= 6

    def test_cardinalities_in_range(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            q = generate_query(rng, min_cardinality=100, max_cardinality=10_000)
            for c in q.cardinalities:
                assert 100 <= c <= 10_000

    def test_selectivities_in_range(self):
        rng = np.random.default_rng(8)
        for _ in range(20):
            q = generate_query(rng, min_selectivity=0.01, max_selectivity=0.5)
            for sel in q.selectivities.values():
                assert 0.01 <= sel <= 0.5

    def test_join_graph_is_connected(self):
        rng = np.random.default_rng(99)
        for _ in range(30):
            q = generate_query(rng)
            assert _is_connected(q.n_tables, q.edges), "Join graph must be connected"

    def test_edge_keys_canonical(self):
        """Edge keys must have i < j."""
        rng = np.random.default_rng(55)
        for _ in range(20):
            q = generate_query(rng)
            for u, v in q.edges:
                assert u < v, f"Edge key ({u},{v}) not canonical"

    def test_deterministic_with_same_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        q1 = generate_query(rng1)
        q2 = generate_query(rng2)
        assert q1.n_tables == q2.n_tables
        assert q1.cardinalities == q2.cardinalities
        assert q1.selectivities == q2.selectivities
        assert q1.edges == q2.edges

    def test_different_seeds_differ(self):
        q1 = generate_query(np.random.default_rng(0))
        q2 = generate_query(np.random.default_rng(1))
        # Not guaranteed to differ on every field but almost certainly differ
        assert q1.cardinalities != q2.cardinalities or q1.n_tables != q2.n_tables

    def test_table_names_length(self):
        rng = np.random.default_rng(3)
        q = generate_query(rng)
        assert len(q.table_names) == q.n_tables
        assert q.table_names[0] == "T0"

    def test_has_edge_symmetry(self):
        rng = np.random.default_rng(10)
        q = generate_query(rng)
        for u, v in q.edges:
            assert q.has_edge(u, v)
            assert q.has_edge(v, u)


class TestMakeQueryBatch:
    def test_batch_length(self):
        batch = make_query_batch(seed=42, n_queries=10)
        assert len(batch) == 10

    def test_batch_reproducible(self):
        b1 = make_query_batch(seed=42, n_queries=5)
        b2 = make_query_batch(seed=42, n_queries=5)
        for q1, q2 in zip(b1, b2):
            assert q1.n_tables == q2.n_tables
            assert q1.cardinalities == q2.cardinalities

    def test_train_test_seeds_differ(self):
        train = make_query_batch(seed=42, n_queries=10)
        test = make_query_batch(seed=99, n_queries=10)
        # Extremely unlikely to be identical; confirms seed separation
        cards_train = [q.cardinalities for q in train]
        cards_test = [q.cardinalities for q in test]
        assert cards_train != cards_test
