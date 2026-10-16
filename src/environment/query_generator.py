"""
query_generator.py
------------------
Generates reproducible synthetic query instances for QueryPilot.

Each query is represented as:
  - n_tables  : int, number of base tables (3–6 for MVP)
  - cardinalities : list[int], row counts for each base table
  - selectivities : dict[(i,j) -> float], join selectivity for each edge
                    keys are always (min(i,j), max(i,j)) so i < j
  - edges      : set of (i,j) pairs that form the join graph

The join graph is guaranteed to be connected (generated via random spanning
tree + optional extra edges) so no disconnected cross-joins arise by accident.

Design note — state encoding
----------------------------
A query instance carries all the information needed to reproduce any
environment episode deterministically from the same seed.  The generator
itself is stateless; pass an rng (numpy.random.Generator) for reproducibility.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Set, Tuple


@dataclass
class QueryInstance:
    """Immutable description of one join-order optimisation problem."""

    n_tables: int
    cardinalities: Tuple[int, ...]          # indexed 0 … n_tables-1
    selectivities: Dict[Tuple[int, int], float]  # key: (i,j) with i<j
    edges: FrozenSet[Tuple[int, int]]       # same key convention

    # human-readable alias: table names T0, T1, …
    @property
    def table_names(self) -> Tuple[str, ...]:
        return tuple(f"T{i}" for i in range(self.n_tables))

    def selectivity(self, i: int, j: int) -> float:
        """Return selectivity for edge (i,j); order-independent."""
        key = (min(i, j), max(i, j))
        return self.selectivities[key]

    def has_edge(self, i: int, j: int) -> bool:
        key = (min(i, j), max(i, j))
        return key in self.edges


def _random_spanning_tree(
    n: int, rng: np.random.Generator
) -> Set[Tuple[int, int]]:
    """
    Prim's algorithm with random edge selection to produce a random spanning
    tree on n nodes.  Guarantees connectivity.
    """
    visited = {0}
    edges: Set[Tuple[int, int]] = set()
    while len(visited) < n:
        candidates = [
            (u, v)
            for u in visited
            for v in range(n)
            if v not in visited
        ]
        u, v = candidates[rng.integers(len(candidates))]
        edges.add((min(u, v), max(u, v)))
        visited.add(v)
    return edges


def generate_query(
    rng: np.random.Generator,
    min_tables: int = 3,
    max_tables: int = 6,
    min_cardinality: int = 100,
    max_cardinality: int = 10_000,
    min_selectivity: float = 0.01,
    max_selectivity: float = 0.5,
    extra_edge_prob: float = 0.3,
) -> QueryInstance:
    """
    Generate one random connected-query instance.

    Parameters
    ----------
    rng               : numpy random Generator (caller controls seed)
    min/max_tables    : range of table counts
    min/max_cardinality: row-count range per table
    min/max_selectivity: per-edge selectivity range
    extra_edge_prob   : probability of adding each non-tree edge (makes
                        denser join graphs; kept low to stay tractable)

    Returns
    -------
    QueryInstance with a connected join graph.
    """
    n = int(rng.integers(min_tables, max_tables + 1))

    # Base-table cardinalities
    cards = tuple(
        int(rng.integers(min_cardinality, max_cardinality + 1))
        for _ in range(n)
    )

    # Build a random spanning tree to guarantee connectivity
    tree_edges = _random_spanning_tree(n, rng)

    # Optionally add extra edges for denser graphs
    all_possible = {
        (i, j) for i in range(n) for j in range(i + 1, n)
    }
    extra_edges = {
        e for e in (all_possible - tree_edges)
        if rng.random() < extra_edge_prob
    }
    edges = tree_edges | extra_edges

    # Assign a selectivity to every edge
    sels: Dict[Tuple[int, int], float] = {
        e: float(rng.uniform(min_selectivity, max_selectivity))
        for e in edges
    }

    return QueryInstance(
        n_tables=n,
        cardinalities=cards,
        selectivities=sels,
        edges=frozenset(edges),
    )


def make_query_batch(
    seed: int,
    n_queries: int,
    **kwargs,
) -> list[QueryInstance]:
    """
    Generate a reproducible batch of query instances.

    Use separate seeds for training and test batches (e.g. seed=42 for
    training, seed=99 for test) to avoid data leakage.
    """
    rng = np.random.default_rng(seed)
    return [generate_query(rng, **kwargs) for _ in range(n_queries)]
