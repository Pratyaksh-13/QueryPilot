"""
metrics.py
----------
Evaluation metrics and statistical aggregation for QueryPilot.

Metrics:
  - total_modeled_cost: Cumulative estimated join execution cost.
  - optimality_gap_percent: Percentage gap relative to exhaustive minimum:
        gap = 100 * (cost_method - cost_opt) / cost_opt
  - aggregate_metrics: Calculates mean, std, median, min, max for any metric.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np


def optimality_gap(cost: float, opt_cost: float) -> float:
    """
    Compute optimality gap percentage vs exhaustive optimal cost:
      gap = 100 * (cost - opt_cost) / opt_cost

    Handles opt_cost <= 0 safely (returns 0.0 if cost == opt_cost, else float('inf')).
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
