"""
greedy_agent.py
---------------
Greedy baseline for join-order selection.

At each step, picks the valid join pair that has the lowest *immediate*
modelled join cost (rows_r + rows_s + rows_result). This is a local greedy
heuristic — it is not guaranteed to find the globally optimal plan.

Uses the same cost model as the environment so the comparison is fair.
"""

from __future__ import annotations

from src.environment.join_environment import JoinEnvironment, get_valid_actions
from src.environment.cardinality import (
    estimate_join_cardinality,
    get_connecting_selectivities,
)
from src.environment.cost_model import join_cost
from src.environment.query_generator import QueryInstance


def _greedy_action(env: JoinEnvironment) -> tuple[int, int]:
    """
    Choose the valid action with the lowest immediate join cost.

    Ties are broken by taking the first minimum found (deterministic).
    """
    components = env.components
    query = env.query
    best_action = None
    best_cost = float("inf")

    for i, j in get_valid_actions(components, query):
        comp_r = components[i]
        comp_s = components[j]
        edge_sels = get_connecting_selectivities(
            comp_r.tables, comp_s.tables, query.selectivities, query.edges
        )
        rows_result = estimate_join_cardinality(comp_r.rows, comp_s.rows, edge_sels)
        cost = join_cost(comp_r.rows, comp_s.rows, rows_result)
        if cost < best_cost:
            best_cost = cost
            best_action = (i, j)

    return best_action  # always non-None because episode is not terminal


def run_greedy_episode(query: QueryInstance) -> int:
    """
    Run one complete episode using the greedy policy.

    Returns
    -------
    Total modelled join cost for the episode.
    """
    env = JoinEnvironment(query)
    env.reset()
    while not env.terminated:
        action = _greedy_action(env)
        env.step(action)
    return env.total_cost
