"""
metrics.py
----------
Evaluation metrics and statistical aggregation for QueryPilot.

Metrics:
  - optimality_gap: Percentage gap relative to exhaustive minimum.
  - aggregate_values: Summary statistics (mean, std, median, min, max).
  - bootstrap_ci: Non-parametric bootstrap confidence interval.
  - paired_comparison: Per-query win/tie/loss and median gap difference.
  - peak_intermediate_cardinality: Max intermediate rows in an episode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


def optimality_gap(cost: float, opt_cost: float) -> float:
    """
    Compute optimality gap percentage vs exhaustive optimal cost:
      gap = 100 * (cost - opt_cost) / opt_cost

    Handles opt_cost <= 0 safely (returns 0.0 if cost == opt_cost, else inf).
    """
    if opt_cost <= 0:
        return 0.0 if cost <= 0 else float("inf")
    return max(0.0, 100.0 * (cost - opt_cost) / opt_cost)


def aggregate_values(values: Sequence[float]) -> Dict[str, float]:
    """
    Compute summary statistics for a sequence of numbers:
    mean, std, median, min, max.
    """
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    arr = np.array(values, dtype=float)
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def bootstrap_ci(
    values: Sequence[float],
    stat_fn=np.median,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    values   : sample data
    stat_fn  : statistic to bootstrap (default: np.median)
    n_boot   : number of bootstrap resamples
    ci_level : confidence level (e.g. 0.95 for 95% CI)
    rng      : numpy random Generator; created fresh if None

    Returns
    -------
    (observed_stat, lower_bound, upper_bound)
    """
    arr = np.array(values, dtype=float)
    if rng is None:
        rng = np.random.default_rng(0)
    observed = float(stat_fn(arr))
    if len(arr) < 2:
        return observed, observed, observed
    boot_stats = [
        float(stat_fn(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_boot)
    ]
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return observed, lo, hi


def paired_comparison(
    gaps_a: Sequence[float],
    gaps_b: Sequence[float],
    tie_tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Paired per-query comparison of method A vs method B (lower gap = better).

    Parameters
    ----------
    gaps_a  : optimality gaps for method A (e.g., RL)
    gaps_b  : optimality gaps for method B (e.g., Greedy)
    tie_tol : absolute difference below which two gaps are declared a tie

    Returns
    -------
    dict with:
      wins_a      : queries where A strictly beats B
      ties        : queries where |gap_a - gap_b| <= tie_tol
      losses_a    : queries where B strictly beats A
      median_diff : median(gap_a - gap_b)  (negative means A usually wins)
      ci_diff_lo, ci_diff_hi : 95% bootstrap CI of median(gap_a - gap_b)
    """
    a = np.array(gaps_a, dtype=float)
    b = np.array(gaps_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("gaps_a and gaps_b must have the same length.")

    diffs = a - b
    wins_a = int(np.sum(diffs < -tie_tol))
    losses_a = int(np.sum(diffs > tie_tol))
    ties = len(diffs) - wins_a - losses_a

    _, lo, hi = bootstrap_ci(diffs, stat_fn=np.median, n_boot=2000)

    return {
        "n_queries": len(diffs),
        "wins_a": wins_a,
        "ties": ties,
        "losses_a": losses_a,
        "median_diff": float(np.median(diffs)),
        "ci_diff_lo": lo,
        "ci_diff_hi": hi,
    }
