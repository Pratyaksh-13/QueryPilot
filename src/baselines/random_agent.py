"""
random_agent.py
---------------
Random baseline for join-order selection.

At each step, picks uniformly at random from all valid join actions.
Uses a seeded numpy Generator so episodes are reproducible.
"""

from __future__ import annotations

import numpy as np

from src.environment.join_environment import JoinEnvironment
from src.environment.query_generator import QueryInstance


def run_random_episode(query: QueryInstance, rng: np.random.Generator) -> int:
    """
    Run one complete episode choosing joins at random.

    Parameters
    ----------
    query : the query to optimise
    rng   : seeded Generator — caller controls reproducibility

    Returns
    -------
    Total modelled join cost for the episode.
    """
    env = JoinEnvironment(query)
    _, valid = env.reset()
    while not env.terminated:
        action = valid[rng.integers(len(valid))]
        _, _, _, valid, _ = env.step(action)
    return env.total_cost
