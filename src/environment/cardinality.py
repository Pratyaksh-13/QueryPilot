"""
cardinality.py
--------------
Cardinality estimation for intermediate join results.

We use the classical independence assumption:

    rows(R ⋈ S) ≈ rows(R) × rows(S) × ∏ selectivities(connecting edges)

This is a well-known simplifying approximation. Real optimisers use
histograms and samples; we use this as a transparent cost proxy only.

When R and S are connected by multiple join edges their selectivities are
applied multiplicatively, which further reduces the estimated output size.
Results are clamped to [1, MAX_CARD] to prevent numeric overflow.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

MAX_CARD: int = 10_000_000  # hard upper bound on intermediate cardinality


def estimate_join_cardinality(
    rows_r: int,
    rows_s: int,
    edge_selectivities: list[float],
) -> int:
    """
    Estimate the output row count when joining two components.

    Parameters
    ----------
    rows_r, rows_s      : current row counts of the two components
    edge_selectivities  : selectivities for all join edges between R and S;
                          must be non-empty (caller ensures a legal join)

    Returns
    -------
    Estimated row count, clamped to [1, MAX_CARD].
    """
    if not edge_selectivities:
        raise ValueError("edge_selectivities must be non-empty for a valid join")

    # Multiply all selectivities; clamp each one to avoid degenerate values.
    combined_sel = 1.0
    for sel in edge_selectivities:
        combined_sel *= max(1e-9, min(1.0, sel))

    return max(1, min(MAX_CARD, int(rows_r * rows_s * combined_sel)))


def get_connecting_selectivities(
    tables_r: FrozenSet[int],
    tables_s: FrozenSet[int],
    selectivities: Dict[Tuple[int, int], float],
    edges: FrozenSet[Tuple[int, int]],
) -> list[float]:
    """
    Collect the selectivities for every join edge between component R and S.

    Returns an empty list when no such edge exists (illegal join).
    The environment checks this before executing any action.
    """
    result = []
    for u in tables_r:
        for v in tables_s:
            key = (min(u, v), max(u, v))
            if key in edges:
                result.append(selectivities[key])
    return result
