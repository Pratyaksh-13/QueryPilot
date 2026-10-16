"""
join_environment.py
-------------------
Gym-style environment for join-order optimisation (no gym dependency).

State representation
--------------------
The episode state is a tuple of *Component* objects.  Each component tracks:
  - tables   : FrozenSet[int]  — which base tables are included
  - rows     : int             — current estimated row count

A hashable discrete *state key* is derived for Q-table lookup:
  state_key = tuple of (frozenset_of_tables, discretised_rows) sorted by
              frozenset, where rows are discretised into log-10 buckets
              (powers of 10: <10, <100, <1K, <10K, <100K, <1M, ≥1M).

This discretisation is explicit and documented: states with similar row
counts map to the same bucket.  States that differ only in row count within
the same bucket are merged — a known abstraction that may limit
generalisation but keeps the Q-table tractable.

Action space
------------
At each step the agent chooses a pair of indices (i, j) into the current
component list (i < j) such that there exists at least one join edge between
a base table in component i and one in component j.  Invalid pairs are
masked out before any policy decision.

The canonical action ID for pair (i, j) with i<j in a list of k components:
  action_id = i * k + j    (note: k changes each step, so action IDs are
                             re-computed each step; the *pair* is what matters)

Episode termination
-------------------
The episode ends when exactly 1 component remains (all joins executed).

Info dict
---------
step() returns an `info` dict with:
  'join_cost' : raw modelled cost of this join (not normalised)
  'rows_r'    : rows of left component before join
  'rows_s'    : rows of right component before join
  'rows_result': rows of the joined result
  'tables_joined': the two frozensets that were joined
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .query_generator import QueryInstance
from .cardinality import (
    estimate_join_cardinality,
    get_connecting_selectivities,
)
from .cost_model import join_cost, step_reward, DEFAULT_COST_SCALE


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

@dataclass
class Component:
    """Represents one base table or intermediate join result."""
    tables: FrozenSet[int]   # base-table indices included in this component
    rows: int                # current estimated row count

    def __hash__(self):
        return hash((self.tables, self.rows))

    def __eq__(self, other):
        return self.tables == other.tables and self.rows == other.rows

    def __repr__(self):
        names = ",".join(f"T{t}" for t in sorted(self.tables))
        return f"Component({{{names}}}, rows={self.rows})"


# ---------------------------------------------------------------------------
# State key (for Q-table indexing)
# ---------------------------------------------------------------------------

_ROW_BINS = [10, 100, 1_000, 10_000, 100_000, 1_000_000]


def _discretise_rows(rows: int) -> int:
    """
    Map row count to a discrete bucket (0–6).
    Bucket 0: rows < 10, …, bucket 6: rows >= 1M.
    Documented abstraction: states within the same bucket are merged.
    """
    for i, threshold in enumerate(_ROW_BINS):
        if rows < threshold:
            return i
    return len(_ROW_BINS)


def make_state_key(components: List[Component]) -> tuple:
    """
    Produce a hashable, order-invariant state key for Q-table lookup.

    Format: tuple of (frozenset_of_table_indices, row_bucket) pairs,
    sorted lexicographically by frozenset to ensure order-invariance.

    Example (3-table query mid-episode):
      ((frozenset({0,1}), 4), (frozenset({2}), 2))
    """
    items = [
        (c.tables, _discretise_rows(c.rows))
        for c in components
    ]
    # Sort by the frozenset for a canonical ordering
    items.sort(key=lambda x: sorted(x[0]))
    return tuple((fs, bucket) for fs, bucket in items)


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def get_valid_actions(
    components: List[Component],
    query: QueryInstance,
) -> List[Tuple[int, int]]:
    """
    Return all valid (i, j) pairs with i<j such that at least one join edge
    connects a base table in components[i] to one in components[j].

    Always returns a list; empty only if the episode is terminal (shouldn't
    happen if called before termination).
    """
    valid = []
    k = len(components)
    for i in range(k):
        for j in range(i + 1, k):
            sels = get_connecting_selectivities(
                components[i].tables,
                components[j].tables,
                query.selectivities,
                query.edges,
            )
            if sels:  # at least one join edge exists
                valid.append((i, j))
    return valid


# ---------------------------------------------------------------------------
# JoinEnvironment
# ---------------------------------------------------------------------------

class JoinEnvironment:
    """
    Episode-based environment for join-order optimisation.

    Usage
    -----
    env = JoinEnvironment(query, cost_scale=1_000_000)
    state_key, valid_actions = env.reset()
    while not env.terminated:
        action = agent.select(state_key, valid_actions)
        state_key, reward, terminated, valid_actions, info = env.step(action)

    The caller is responsible for using `valid_actions` to mask invalid
    choices; the environment raises ValueError on illegal actions.
    """

    def __init__(self, query: QueryInstance,
                 cost_scale: int = DEFAULT_COST_SCALE):
        self.query = query
        self.cost_scale = cost_scale
        self._components: List[Component] = []
        self._terminated = False
        self._total_cost: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[tuple, List[Tuple[int, int]]]:
        """
        Reset to the beginning of a new episode (same query).

        Returns
        -------
        state_key    : hashable key for Q-table
        valid_actions: list of (i, j) pairs that are legal at step 0
        """
        self._components = [
            Component(tables=frozenset({i}), rows=self.query.cardinalities[i])
            for i in range(self.query.n_tables)
        ]
        self._terminated = False
        self._total_cost = 0
        valid = get_valid_actions(self._components, self.query)
        return make_state_key(self._components), valid

    def step(
        self, action: Tuple[int, int]
    ) -> Tuple[tuple, float, bool, List[Tuple[int, int]], dict]:
        """
        Execute one join action.

        Parameters
        ----------
        action : (i, j) with i < j — indices into the current component list

        Returns
        -------
        next_state_key  : state key after the join
        reward          : step reward (non-positive float)
        terminated      : True if only one component remains
        valid_actions   : valid actions from next state (empty if terminated)
        info            : diagnostic dict (see module docstring)
        """
        if self._terminated:
            raise RuntimeError("Episode already terminated; call reset() first.")

        i, j = action
        if i >= j:
            raise ValueError(f"Action must have i < j; got ({i}, {j})")

        components = self._components
        if i < 0 or j >= len(components):
            raise ValueError(
                f"Action ({i},{j}) out of range for {len(components)} components"
            )

        comp_r = components[i]
        comp_s = components[j]

        # Validate: must have at least one connecting edge
        edge_sels = get_connecting_selectivities(
            comp_r.tables, comp_s.tables,
            self.query.selectivities, self.query.edges,
        )
        if not edge_sels:
            raise ValueError(
                f"Illegal action ({i},{j}): no join edge between "
                f"{comp_r.tables} and {comp_s.tables}"
            )

        # Cardinality estimation
        rows_result = estimate_join_cardinality(
            comp_r.rows, comp_s.rows, edge_sels
        )

        # Cost and reward
        cost = join_cost(comp_r.rows, comp_s.rows, rows_result)
        reward = step_reward(comp_r.rows, comp_s.rows, rows_result,
                             self.cost_scale)
        self._total_cost += cost

        # Merge components (remove j first to preserve indices)
        merged = Component(
            tables=comp_r.tables | comp_s.tables,
            rows=rows_result,
        )
        new_components = [
            c for idx, c in enumerate(components) if idx not in (i, j)
        ]
        new_components.append(merged)
        self._components = new_components

        # Termination
        self._terminated = (len(self._components) == 1)
        valid = [] if self._terminated else get_valid_actions(
            self._components, self.query
        )
        next_key = make_state_key(self._components)

        info = {
            "join_cost": cost,
            "rows_r": comp_r.rows,
            "rows_s": comp_s.rows,
            "rows_result": rows_result,
            "tables_joined": (comp_r.tables, comp_s.tables),
            "total_cost_so_far": self._total_cost,
        }
        return next_key, reward, self._terminated, valid, info

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def components(self) -> List[Component]:
        return list(self._components)  # defensive copy

    @property
    def total_cost(self) -> int:
        """Cumulative modelled join cost for the current episode."""
        return self._total_cost

    @property
    def n_components(self) -> int:
        return len(self._components)
