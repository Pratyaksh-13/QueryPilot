"""
query_generator.py
------------------
Generates reproducible synthetic query instances for QueryPilot.

A QueryInstance holds:
  - n_tables       : number of base tables (3–6 for MVP)
  - cardinalities  : row count per base table
  - selectivities  : join selectivity per edge, keyed as (i, j) with i < j
  - edges          : frozenset of all join edges (same key convention)

The join graph is always connected — built from a random spanning tree plus
optional extra edges — so accidental cross-joins cannot occur.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, FrozenSet, Set, Tuple


@dataclass
class QueryInstance:
    """Immutable description of one join-order optimisation problem."""

    n_tables: int
    cardinalities: Tuple[int, ...]           # row counts indexed 0 … n_tables-1
    selectivities: Dict[Tuple[int, int], float]  # key: (i, j) with i < j
    edges: FrozenSet[Tuple[int, int]]        # same key convention

    @property
    def table_names(self) -> Tuple[str, ...]:
        """Human-readable table aliases: T0, T1, …"""
        return tuple(f"T{i}" for i in range(self.n_tables))

    def selectivity(self, i: int, j: int) -> float:
        """Return selectivity for edge (i, j); order-independent."""
        return self.selectivities[(min(i, j), max(i, j))]

    def has_edge(self, i: int, j: int) -> bool:
        return (min(i, j), max(i, j)) in self.edges


def _random_spanning_tree(n: int, rng: np.random.Generator) -> Set[Tuple[int, int]]:
    """
    Build a random spanning tree on n nodes using Prim's algorithm with
    randomised edge selection. Guarantees the result is connected.
    """
    visited = {0}
    edges: Set[Tuple[int, int]] = set()
    while len(visited) < n:
        candidates = [(u, v) for u in visited for v in range(n) if v not in visited]
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
    rng             : numpy Generator — caller controls the seed
    min/max_tables  : inclusive range for table count
    min/max_cardinality : row-count range per table
    min/max_selectivity : per-edge selectivity range
    extra_edge_prob : probability of adding each non-tree edge (creates
                      denser join graphs; kept low to stay tractable)
    """
    n = int(rng.integers(min_tables, max_tables + 1))

    cardinalities = tuple(
        int(rng.integers(min_cardinality, max_cardinality + 1))
        for _ in range(n)
    )

    # Spanning tree guarantees connectivity; extra edges make denser graphs.
    tree_edges = _random_spanning_tree(n, rng)
    all_possible = {(i, j) for i in range(n) for j in range(i + 1, n)}
    extra_edges = {e for e in (all_possible - tree_edges) if rng.random() < extra_edge_prob}
    edges = tree_edges | extra_edges

    selectivities: Dict[Tuple[int, int], float] = {
        e: float(rng.uniform(min_selectivity, max_selectivity))
        for e in edges
    }

    return QueryInstance(
        n_tables=n,
        cardinalities=cardinalities,
        selectivities=selectivities,
        edges=frozenset(edges),
    )


def make_query_batch(seed: int, n_queries: int, **kwargs) -> list[QueryInstance]:
    """
    Generate a reproducible batch of query instances from a single seed.

    Use different seeds for training and test splits to prevent data leakage
    (e.g. seed=42 for training, seed=99 for evaluation).
    """
    rng = np.random.default_rng(seed)
    return [generate_query(rng, **kwargs) for _ in range(n_queries)]
