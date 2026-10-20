"""
cost_model.py
-------------
Join cost model for QueryPilot.

Cost formula
------------
    join_cost(R, S) = rows(R) + rows(S) + rows(R ⋈ S)

This models reading both inputs and writing the intermediate result. It is a
simplified proxy — not a real database engine cost — but it is transparent,
cheap to compute, and used identically for the RL agent and all baselines,
which ensures fair comparisons.

Reward
------
    reward = -join_cost / COST_SCALE

COST_SCALE normalises rewards to roughly [-1, 0) so Q-values stay numerically
stable. With gamma = 1.0, maximising total episode return is equivalent to
minimising total modelled join cost.
"""

from __future__ import annotations

DEFAULT_COST_SCALE: int = 1_000_000  # 1 M rows ≈ reward of -1 per step


def join_cost(rows_r: int, rows_s: int, rows_result: int) -> int:
    """
    Modelled cost of joining two components.

    Returns a non-negative integer (reading both sides + writing the result).
    """
    return rows_r + rows_s + rows_result


def step_reward(
    rows_r: int,
    rows_s: int,
    rows_result: int,
    cost_scale: int = DEFAULT_COST_SCALE,
) -> float:
    """Per-step reward for the RL agent. Always non-positive."""
    return -join_cost(rows_r, rows_s, rows_result) / cost_scale
