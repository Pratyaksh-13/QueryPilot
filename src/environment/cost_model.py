"""
cost_model.py
-------------
Join cost model for QueryPilot.

Cost formula (transparent proxy)
---------------------------------
    join_cost(R, S) = rows(R) + rows(S) + rows(R ⋈ S)

Rationale: this models the cost of reading both inputs and writing the
intermediate result.  It is a *simplified proxy* for processing/memory cost
and is NOT a universal database engine cost model (real engines include I/O,
CPU, pipelining, buffer effects, etc.).

The SAME formula is used identically for the RL agent, all baselines, and the
exhaustive reference so that comparisons are fair.

Reward
------
    reward_t = -join_cost(R, S) / COST_SCALE

COST_SCALE normalises rewards into roughly [-1, 0) so Q-values stay
numerically well-behaved.  A documented default is provided; override via
config if needed.

Cumulative episode return
--------------------------
With gamma=1.0 the agent maximises sum of step rewards, which is equivalent
to minimising the total modelled join cost across all steps.  This is
intentional: the episode return directly reflects total intermediate-result
volume (a standard proxy for query execution effort).
"""

from __future__ import annotations

DEFAULT_COST_SCALE: int = 1_000_000  # 1 M rows ≈ reward of -1 per step


def join_cost(rows_r: int, rows_s: int, rows_result: int) -> int:
    """
    Compute the modelled cost of joining two components.

    Parameters
    ----------
    rows_r      : row count of the left component before the join
    rows_s      : row count of the right component before the join
    rows_result : estimated row count of the output

    Returns
    -------
    Non-negative integer cost.
    """
    return rows_r + rows_s + rows_result


def step_reward(rows_r: int, rows_s: int, rows_result: int,
                cost_scale: int = DEFAULT_COST_SCALE) -> float:
    """
    Compute the per-step reward signal for the RL agent.

    Always non-positive (a cost penalty).
    """
    cost = join_cost(rows_r, rows_s, rows_result)
    return -cost / cost_scale
