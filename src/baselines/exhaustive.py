"""
exhaustive.py
-------------
Exhaustive join-order enumeration for small queries (evaluation only).

WARNING: This module must NEVER be used during RL training.
         It exists solely as an evaluation oracle to compute the minimum
         possible total cost under the same cost model, so we can measure
         how close each method gets to optimal.

Algorithm
---------
We enumerate all legal join orderings via recursive branch-and-bound.
At each step we try every valid (i, j) pair from the current component list,
recurse with the merged result, and keep track of the global minimum.

Complexity: O(n! * n^2) in the worst case, which is only tractable for
n ≤ 6 tables (≤ 720 orderings for 6 tables, manageable in milliseconds).
Do not call this on queries with > 8 tables without adding a cutoff.

Cost model
----------
Uses the same join_cost and estimate_join_cardinality functions as the
environment and all other baselines to guarantee fair comparison.
"""

from __future__ import annotations

from typing import FrozenSet, List, Tuple, Dict
from dataclasses import dataclass

from src.environment.cardinality import (
    estimate_join_cardinality,
    get_connecting_selectivities,
)
from src.environment.cost_model import join_cost
from src.environment.query_generator import QueryInstance


@dataclass
class _Component:
    """Lightweight component for exhaustive search (no environment dependency)."""
    tables: FrozenSet[int]
    rows: int


def _enumerate(
    components: List[_Component],
    query: QueryInstance,
    accumulated_cost: int,
    best_so_far: list,          # mutable box so recursion can update it
) -> None:
    """
    Recursively enumerate all legal join orderings.

    Parameters
    ----------
    components       : current list of components (base tables or intermediate)
    query            : the query being optimised
    accumulated_cost : total cost from joins executed so far in this branch
    best_so_far      : single-element list holding the current best total cost
                       (we use a list so nested calls can mutate it)
    """
    if len(components) == 1:
        # Leaf: complete plan — update global best if this branch is cheaper.
        if accumulated_cost < best_so_far[0]:
            best_so_far[0] = accumulated_cost
        return

    # Pruning: if this branch already exceeds the best known cost, cut it.
    if accumulated_cost >= best_so_far[0]:
        return

    k = len(components)
    for i in range(k):
        for j in range(i + 1, k):
            edge_sels = get_connecting_selectivities(
                components[i].tables, components[j].tables,
                query.selectivities, query.edges,
            )
            if not edge_sels:
                continue  # disconnected pair — skip

            rows_result = estimate_join_cardinality(
                components[i].rows, components[j].rows, edge_sels
            )
            cost = join_cost(components[i].rows, components[j].rows, rows_result)
            merged = _Component(
                tables=components[i].tables | components[j].tables,
                rows=rows_result,
            )
            remaining = [c for idx, c in enumerate(components) if idx not in (i, j)]
            remaining.append(merged)

            _enumerate(remaining, query, accumulated_cost + cost, best_so_far)


def exhaustive_min_cost(query: QueryInstance) -> int:
    """
    Find the minimum total modelled join cost over all legal join orderings.

    Parameters
    ----------
    query : the query to optimise (must have ≤ 8 tables for tractability)

    Returns
    -------
    Minimum total cost achievable under the same cost model used everywhere.

    Raises
    ------
    ValueError if the query has no tables.
    """
    if query.n_tables == 0:
        raise ValueError("Query must have at least one table.")
    if query.n_tables == 1:
        return 0  # nothing to join

    components = [
        _Component(tables=frozenset({i}), rows=query.cardinalities[i])
        for i in range(query.n_tables)
    ]

    best_so_far = [float("inf")]
    _enumerate(components, query, 0, best_so_far)
    return int(best_so_far[0])


def plan_cost(
    query: QueryInstance,
    join_order: List[Tuple[FrozenSet[int], FrozenSet[int]]],
) -> int:
    """
    Compute the total cost of a specific join plan.

    Parameters
    ----------
    query      : the query instance
    join_order : list of (tables_r, tables_s) pairs in execution order,
                 where each frozenset identifies a component by the base
                 tables it contains.

    Returns
    -------
    Total modelled cost of executing this exact plan.

    This is used in tests to cross-check that an environment episode
    produces the same cost as independently computing it from the plan.
    """
    # Build a live component map: frozenset_of_tables → current row count
    row_map: Dict[FrozenSet[int], int] = {
        frozenset({i}): query.cardinalities[i]
        for i in range(query.n_tables)
    }

    total = 0
    for tables_r, tables_s in join_order:
        rows_r = row_map[tables_r]
        rows_s = row_map[tables_s]
        edge_sels = get_connecting_selectivities(
            tables_r, tables_s, query.selectivities, query.edges
        )
        rows_result = estimate_join_cardinality(rows_r, rows_s, edge_sels)
        total += join_cost(rows_r, rows_s, rows_result)

        # Merge: remove the two inputs, add the combined result.
        del row_map[tables_r]
        del row_map[tables_s]
        row_map[tables_r | tables_s] = rows_result

    return total
