"""
cardinality.py
--------------
Cardinality estimation for intermediate join results.

Design notes
------------
We use the simple independence assumption:

    rows(R ⋈ S) ≈ rows(R) * rows(S) * selectivity(R, S)

This is the same estimator used in many classical query optimisers and is a
well-known simplifying approximation (it assumes attribute value independence
and uniform distributions).  It is NOT a universal database rule; real
optimisers use histograms, samples, and more.

For intermediate results that join multiple base tables we treat the combined
component as a single virtual relation.  The selectivity used is the
*minimum* per-edge selectivity between any base table in R and any base table
in S that share a join edge (i.e., the tightest filter on the first available
predicate).  If there are multiple edges between R and S we apply each
additional selectivity multiplicatively, capped at a minimum cardinality of 1.

Numeric safety
--------------
- Cardinalities are clipped to [1, MAX_CARD] to prevent explosion.
- Selectivities are clipped to (0, 1].
"""

from __future__ import annotations

from typing import FrozenSet, Tuple, Dict

MAX_CARD: int = 10_000_000  # hard cap on intermediate cardinality (10 M rows)


def estimate_join_cardinality(
    rows_r: int,
    rows_s: int,
    edge_selectivities: list[float],
) -> int:
    """
    Estimate the output cardinality when joining two components.

    Parameters
    ----------
    rows_r, rows_s      : current row counts of the two components
    edge_selectivities  : list of selectivities for every join edge that
                          connects a base table in R to one in S.
                          Must be non-empty (caller validates legal join).

    Returns
    -------
    Estimated integer row count, clipped to [1, MAX_CARD].
    """
    if not edge_selectivities:
        raise ValueError("edge_selectivities must be non-empty for a valid join")

    combined_sel = 1.0
    for sel in edge_selectivities:
        combined_sel *= max(1e-9, min(1.0, sel))  # clamp each selectivity

    estimated = rows_r * rows_s * combined_sel
    return max(1, min(MAX_CARD, int(estimated)))


def get_connecting_selectivities(
    tables_r: FrozenSet[int],
    tables_s: FrozenSet[int],
    selectivities: Dict[Tuple[int, int], float],
    edges: FrozenSet[Tuple[int, int]],
) -> list[float]:
    """
    Return the list of join-edge selectivities between component R and
    component S.  A join edge exists in the query join graph when a base
    table from R is connected to a base table from S.

    Returns an empty list if the two components are not connected (illegal
    join; the environment must not allow this action).
    """
    result = []
    for u in tables_r:
        for v in tables_s:
            key = (min(u, v), max(u, v))
            if key in edges:
                result.append(selectivities[key])
    return result
