"""
benchmark.py
------------
Comparative evaluation harness for QueryPilot.

Compares:
  1. Random baseline
  2. Greedy baseline
  3. Exhaustive search (evaluation oracle for <= 6 tables)
  4. QueryPilot RL agent (greedy evaluation)

All methods are evaluated on identical test query instances using the identical
cardinality estimator and cost model.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.environment.query_generator import QueryInstance
from src.environment.join_environment import JoinEnvironment
from src.baselines.random_agent import run_random_episode
from src.baselines.greedy_agent import run_greedy_episode
from src.baselines.exhaustive import exhaustive_min_cost
from src.agents.q_learning import QLearningAgent
from src.evaluation.metrics import optimality_gap, aggregate_values


def evaluate_query(
    query: QueryInstance,
    agent: Optional[QLearningAgent] = None,
    include_exhaustive: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run all methods on a single query instance and return individual costs,
    timings, and optimality gaps.
    """
    results: Dict[str, Any] = {
        "n_tables": query.n_tables,
        "costs": {},
        "times": {},
        "gaps": {},
    }

    # 1. Exhaustive Oracle (if enabled and tractable)
    opt_cost: Optional[int] = None
    if include_exhaustive and query.n_tables <= 6:
        t0 = time.perf_counter()
        opt_cost = exhaustive_min_cost(query)
        results["times"]["exhaustive"] = time.perf_counter() - t0
        results["costs"]["exhaustive"] = opt_cost
        results["gaps"]["exhaustive"] = 0.0

    # 2. Random Baseline
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    rand_cost = run_random_episode(query, rng=rng)
    results["times"]["random"] = time.perf_counter() - t0
    results["costs"]["random"] = rand_cost
    if opt_cost is not None:
        results["gaps"]["random"] = optimality_gap(rand_cost, opt_cost)

    # 3. Greedy Baseline
    t0 = time.perf_counter()
    greedy_cost = run_greedy_episode(query)
    results["times"]["greedy"] = time.perf_counter() - t0
    results["costs"]["greedy"] = greedy_cost
    if opt_cost is not None:
        results["gaps"]["greedy"] = optimality_gap(greedy_cost, opt_cost)

    # 4. Learned RL Agent
    if agent is not None:
        env = JoinEnvironment(query, state_mode=agent.config.state_mode)
        t0 = time.perf_counter()
        agent_res = agent.evaluate_episode(env)
        results["times"]["rl"] = time.perf_counter() - t0
        results["costs"]["rl"] = agent_res["total_cost"]
        if opt_cost is not None:
            results["gaps"]["rl"] = optimality_gap(agent_res["total_cost"], opt_cost)

    return results


def run_benchmark(
    queries: Sequence[QueryInstance],
    agent: Optional[QLearningAgent] = None,
    include_exhaustive: bool = True,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate all methods across a collection of query instances.

    Returns
    -------
    dict with:
      - 'per_query': list of query evaluation records
      - 'overall': aggregated metrics per method
      - 'by_table_count': aggregated metrics grouped by n_tables (3, 4, 5, 6)
    """
    per_query = []
    methods = ["random", "greedy"]
    if include_exhaustive:
        methods.append("exhaustive")
    if agent is not None:
        methods.append("rl")

    for i, q in enumerate(queries):
        record = evaluate_query(
            query=q,
            agent=agent,
            include_exhaustive=include_exhaustive,
            seed=base_seed + i,
        )
        record["query_idx"] = i
        per_query.append(record)

    # Aggregate overall
    overall: Dict[str, Any] = {}
    for method in methods:
        costs = [r["costs"][method] for r in per_query if method in r["costs"]]
        times = [r["times"][method] for r in per_query if method in r["times"]]
        gaps = [r["gaps"][method] for r in per_query if method in r.get("gaps", {})]

        overall[method] = {
            "cost": aggregate_values(costs),
            "time_sec": aggregate_values(times),
            "gap_percent": aggregate_values(gaps) if gaps else None,
        }

    # Aggregate by table count (3, 4, 5, 6)
    table_counts = sorted(list(set(q.n_tables for q in queries)))
    by_table_count: Dict[int, Dict[str, Any]] = {}

    for n in table_counts:
        sub_records = [r for r in per_query if r["n_tables"] == n]
        by_table_count[n] = {}
        for method in methods:
            costs = [r["costs"][method] for r in sub_records if method in r["costs"]]
            gaps = [r["gaps"][method] for r in sub_records if method in r.get("gaps", {})]
            by_table_count[n][method] = {
                "count": len(sub_records),
                "cost": aggregate_values(costs),
                "gap_percent": aggregate_values(gaps) if gaps else None,
            }

    return {
        "num_queries": len(queries),
        "methods": methods,
        "per_query": per_query,
        "overall": overall,
        "by_table_count": by_table_count,
    }
