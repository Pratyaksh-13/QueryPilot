"""
join_environment.py
-------------------
Episode-based environment for join-order optimisation (no gym dependency).

State
-----
The state is a list of Component objects. Each component tracks which base
tables it contains (as a frozenset) and its current estimated row count.

For Q-table indexing we need a hashable, order-invariant key. We sort
components by their table frozenset and discretise row counts into 7
log-10 buckets (<10, <100, <1K, <10K, <100K, <1M, ≥1M).

Documented abstraction: components with similar row counts within the same
bucket are treated identically. This keeps the Q-table tractable for up to
6 tables while accepting some loss of state resolution.

Actions
-------
An action is a pair (i, j) with i < j — indices into the current component
list. The pair is valid only if at least one join edge exists between a base
table in component i and one in component j. Invalid pairs are always masked
out before the policy sees them, and the environment raises ValueError if an
illegal action is passed in anyway.

Episode termination
-------------------
The episode ends when exactly one component remains (n_tables - 1 steps).

step() return values
---------------------
next_state_key, reward, terminated, valid_actions, info

info keys: join_cost, rows_r, rows_s, rows_result, tables_joined,
           total_cost_so_far
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Tuple

from .query_generator import QueryInstance
from .cardinality import estimate_join_cardinality, get_connecting_selectivities
from .cost_model import join_cost, step_reward, DEFAULT_COST_SCALE
from .state_representation import (
    make_feature_state_key,
    compute_action_feature,
    discretise_selectivity,
)


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

@dataclass
class Component:
    """One base table or intermediate join result."""

    tables: FrozenSet[int]  # base-table indices included in this component
    rows: int               # current estimated row count

    def __hash__(self):
        return hash((self.tables, self.rows))

    def __eq__(self, other):
        return self.tables == other.tables and self.rows == other.rows

    def __repr__(self):
        names = ",".join(f"T{t}" for t in sorted(self.tables))
        return f"Component({{{names}}}, rows={self.rows})"


# ---------------------------------------------------------------------------
# State key
# ---------------------------------------------------------------------------

_ROW_BINS = [10, 100, 1_000, 10_000, 100_000, 1_000_000]


def _discretise_rows(rows: int) -> int:
    """
    Map a row count to a discrete bucket (0–6).
      0: <10,  1: <100,  2: <1K,  3: <10K,  4: <100K,  5: <1M,  6: ≥1M
    """
    for i, threshold in enumerate(_ROW_BINS):
        if rows < threshold:
            return i
    return len(_ROW_BINS)


def make_state_key(components: List[Component]) -> tuple:
    """
    Produce a hashable, order-invariant state key for Q-table lookup.

    Each component contributes a (frozenset_of_tables, row_bucket) pair.
    Pairs are sorted by frozenset so the key is identical regardless of
    the order components appear in the list.

    Example mid-episode key for a 3-table query:
        ((frozenset({0, 1}), 4), (frozenset({2}), 2))
    """
    items = [(c.tables, _discretise_rows(c.rows)) for c in components]
    items.sort(key=lambda x: sorted(x[0]))
    return tuple(items)


# ---------------------------------------------------------------------------
# Valid-action helper
# ---------------------------------------------------------------------------

def get_valid_actions(
    components: List[Component],
    query: QueryInstance,
) -> List[Tuple[int, int]]:
    """
    Return all (i, j) pairs with i < j where at least one join edge connects
    a base table in components[i] to one in components[j].
    """
    valid = []
    k = len(components)
    for i in range(k):
        for j in range(i + 1, k):
            if get_connecting_selectivities(
                components[i].tables, components[j].tables,
                query.selectivities, query.edges,
            ):
                valid.append((i, j))
    return valid


# ---------------------------------------------------------------------------
# JoinEnvironment
# ---------------------------------------------------------------------------

class JoinEnvironment:
    """
    Episode-based environment for join-order optimisation.

    Basic usage
    -----------
    env = JoinEnvironment(query)
    state_key, valid_actions = env.reset()
    while not env.terminated:
        action = agent.select(state_key, valid_actions)
        state_key, reward, terminated, valid_actions, info = env.step(action)

    The caller must choose from `valid_actions`; the environment raises
    ValueError if an illegal action is passed to step().
    """

    def __init__(
        self,
        query: QueryInstance,
        cost_scale: int = DEFAULT_COST_SCALE,
        state_mode: str = "raw",
    ):
        if state_mode not in ("raw", "feature"):
            raise ValueError(f"state_mode must be 'raw' or 'feature', got {state_mode!r}")
        self.query = query
        self.cost_scale = cost_scale
        self.state_mode = state_mode
        self._components: List[Component] = []
        self._terminated = False
        self._total_cost = 0

    # ------------------------------------------------------------------
    # State and Action abstraction helpers
    # ------------------------------------------------------------------

    @property
    def current_state_key(self) -> tuple:
        """Return the state key corresponding to self.state_mode."""
        if self.state_mode == "feature":
            return make_feature_state_key(self._components, self.query, _discretise_rows)
        return make_state_key(self._components)

    def get_feature_state_key(self) -> tuple:
        """Return query-agnostic feature-based state key for current components."""
        return make_feature_state_key(self._components, self.query, _discretise_rows)

    def get_action_feature(self, action: Tuple[int, int]) -> tuple:
        """Return feature tuple for candidate join action (i, j)."""
        return compute_action_feature(action, self._components, self.query, _discretise_rows)

    def get_valid_action_features(self) -> List[Tuple[Tuple[int, int], tuple]]:
        """Return a list of ((i, j), action_feature) for all currently valid actions."""
        valid = get_valid_actions(self._components, self.query)
        return [(a, self.get_action_feature(a)) for a in valid]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[tuple, List[Tuple[int, int]]]:
        """
        Start a new episode with the same query.

        Returns (state_key, valid_actions) for the initial state.
        """
        self._components = [
            Component(tables=frozenset({i}), rows=self.query.cardinalities[i])
            for i in range(self.query.n_tables)
        ]
        self._terminated = False
        self._total_cost = 0
        valid = get_valid_actions(self._components, self.query)
        return self.current_state_key, valid

    def step(self, action: Tuple[int, int]) -> Tuple[tuple, float, bool, List[Tuple[int, int]], dict]:
        """
        Execute one join action.

        Parameters
        ----------
        action : (i, j) with i < j — indices into the current component list

        Returns
        -------
        next_state_key, reward, terminated, valid_actions, info
        """
        if self._terminated:
            raise RuntimeError("Episode is already terminated. Call reset() first.")

        i, j = action
        if i >= j:
            raise ValueError(f"Action must have i < j; got ({i}, {j})")
        if i < 0 or j >= len(self._components):
            raise ValueError(f"Action ({i}, {j}) out of range for {len(self._components)} components")

        comp_r = self._components[i]
        comp_s = self._components[j]

        # Validate connectivity — raises if the pair has no join edge.
        edge_sels = get_connecting_selectivities(
            comp_r.tables, comp_s.tables,
            self.query.selectivities, self.query.edges,
        )
        if not edge_sels:
            raise ValueError(
                f"Illegal action ({i}, {j}): no join edge between "
                f"{comp_r.tables} and {comp_s.tables}"
            )

        # Estimate cardinality and compute cost/reward.
        rows_result = estimate_join_cardinality(comp_r.rows, comp_s.rows, edge_sels)
        cost = join_cost(comp_r.rows, comp_s.rows, rows_result)
        reward = step_reward(comp_r.rows, comp_s.rows, rows_result, self.cost_scale)
        self._total_cost += cost

        # Replace the two joined components with the merged result.
        merged = Component(tables=comp_r.tables | comp_s.tables, rows=rows_result)
        self._components = [
            c for idx, c in enumerate(self._components) if idx not in (i, j)
        ]
        self._components.append(merged)

        self._terminated = len(self._components) == 1
        valid = [] if self._terminated else get_valid_actions(self._components, self.query)

        info = {
            "join_cost": cost,
            "rows_r": comp_r.rows,
            "rows_s": comp_s.rows,
            "rows_result": rows_result,
            "tables_joined": (comp_r.tables, comp_s.tables),
            "total_cost_so_far": self._total_cost,
        }
        return self.current_state_key, reward, self._terminated, valid, info

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
