"""
state_representation.py
-----------------------
State and action feature abstractions for join-order reinforcement learning.

Why abstraction is necessary
----------------------------
The naive state key uses raw table index sets, e.g. `(frozenset({0, 1}), bucket)`.
This has critical limitations:
  1. No cross-query generalization: Table 0 in Query A is completely unrelated
     to Table 0 in Query B. A Q-table trained on raw indices cannot generalize
     to held-out test queries.
  2. Uninformative action space: Action (i, j) is just an index into the current
     component list. Its meaning changes dynamically as components are merged.
  3. Ignores join-graph topology: Raw table indices do not convey whether a
     relation is a leaf, a central hub, or isolated from others.

Proposed Feature Abstraction
----------------------------
1. Component Features:
   Each component c is represented by:
     - size: number of base tables in the component (|c.tables|)
     - row_bucket: log-10 discretised cardinality (0–6)
     - degree: number of other available components connected to c in the
       current query join graph.

2. State Feature:
   - Multiset of component features (sorted for permutation invariance).
   - Graph structure metadata: total active join edges between current components.
   - Result: a hashable, query-agnostic tuple describing the state.

3. Action Feature:
   For a candidate join between components i and j:
     - Component features: feat_r, feat_s sorted canonically: (min(feat), max(feat))
     - Selectivity bucket: discretised connecting selectivity (0–3)
     - Result row bucket: log-10 discretised estimated output rows
   - Result: an interpretable tuple describing what kind of join is being executed,
     e.g., "join a small leaf table with a medium hub table under high selectivity".

Trade-offs
----------
- Benefit: Enables true zero-shot policy evaluation on held-out queries. The Q-table
  learns structural database optimization rules (e.g., "join small, selective leaves first").
- Trade-off (Loss of Resolution / State Aliasing): Discretization and local degree
  features may merge distinct states that differ only in downstream multi-hop graph
  topology or fine-grained row counts within the same bucket.
"""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING
from .cardinality import get_connecting_selectivities, estimate_join_cardinality

if TYPE_CHECKING:
    from .join_environment import Component
    from .query_generator import QueryInstance


# ---------------------------------------------------------------------------
# Selectivity discretisation
# ---------------------------------------------------------------------------

_SELECTIVITY_BINS = [0.05, 0.15, 0.30]


def discretise_selectivity(sel: float) -> int:
    """
    Map an overall join selectivity into discrete bins:
      0: highly selective (< 0.05)
      1: selective (0.05 <= sel < 0.15)
      2: moderate (0.15 <= sel < 0.30)
      3: weak (>= 0.30)
    """
    for i, threshold in enumerate(_SELECTIVITY_BINS):
        if sel < threshold:
            return i
    return len(_SELECTIVITY_BINS)


# ---------------------------------------------------------------------------
# Component and Graph features
# ---------------------------------------------------------------------------

def compute_component_degree(
    comp_idx: int,
    components: List[Component],
    query: QueryInstance,
) -> int:
    """
    Count how many other currently available components share at least one
    join edge with components[comp_idx].
    """
    target = components[comp_idx]
    deg = 0
    for other_idx, other in enumerate(components):
        if other_idx == comp_idx:
            continue
        sels = get_connecting_selectivities(
            target.tables, other.tables,
            query.selectivities, query.edges,
        )
        if sels:
            deg += 1
    return deg


def get_active_edge_count(
    components: List[Component],
    query: QueryInstance,
) -> int:
    """
    Count the total number of unordered component pairs (i, j) that share
    at least one join edge.
    """
    k = len(components)
    edge_count = 0
    for i in range(k):
        for j in range(i + 1, k):
            if get_connecting_selectivities(
                components[i].tables, components[j].tables,
                query.selectivities, query.edges,
            ):
                edge_count += 1
    return edge_count


def compute_component_feature(
    comp_idx: int,
    components: List[Component],
    query: QueryInstance,
    discretise_rows_fn,
) -> Tuple[int, int, int]:
    """
    Extract (size, row_bucket, degree) for components[comp_idx].
    """
    c = components[comp_idx]
    size = len(c.tables)
    row_bucket = discretise_rows_fn(c.rows)
    deg = compute_component_degree(comp_idx, components, query)
    return (size, row_bucket, deg)


# ---------------------------------------------------------------------------
# State feature abstraction
# ---------------------------------------------------------------------------

def make_feature_state_key(
    components: List[Component],
    query: QueryInstance,
    discretise_rows_fn,
) -> tuple:
    """
    Produce a query-agnostic, hashable state key from component and graph features.

    Format:
      (
        n_components,
        n_active_edges,
        ((size_1, row_bucket_1, degree_1), (size_2, row_bucket_2, degree_2), ...)
      )
    The component feature tuples are sorted canonically so the key is invariant
    to the list order of components.
    """
    comp_feats = [
        compute_component_feature(i, components, query, discretise_rows_fn)
        for i in range(len(components))
    ]
    comp_feats.sort()
    n_edges = get_active_edge_count(components, query)
    return (len(components), n_edges, tuple(comp_feats))


# ---------------------------------------------------------------------------
# Action feature abstraction
# ---------------------------------------------------------------------------

def compute_action_feature(
    action: Tuple[int, int],
    components: List[Component],
    query: QueryInstance,
    discretise_rows_fn,
) -> tuple:
    """
    Produce a query-agnostic, hashable feature tuple for a candidate join action (i, j).

    Format:
      (
        min(feat_r, feat_s),
        max(feat_r, feat_s),
        selectivity_bucket,
        result_row_bucket
      )

    Parameters
    ----------
    action : (i, j) with i < j
    """
    i, j = action
    comp_r = components[i]
    comp_s = components[j]

    feat_r = compute_component_feature(i, components, query, discretise_rows_fn)
    feat_s = compute_component_feature(j, components, query, discretise_rows_fn)

    # Canonical order for symmetric join representation
    c1, c2 = (feat_r, feat_s) if feat_r <= feat_s else (feat_s, feat_r)

    edge_sels = get_connecting_selectivities(
        comp_r.tables, comp_s.tables,
        query.selectivities, query.edges,
    )
    if not edge_sels:
        raise ValueError(f"Cannot compute action feature for disconnected pair ({i}, {j})")

    # Combined selectivity across all connecting edges
    prod_sel = 1.0
    for s in edge_sels:
        prod_sel *= s

    sel_bucket = discretise_selectivity(prod_sel)
    rows_result = estimate_join_cardinality(comp_r.rows, comp_s.rows, edge_sels)
    result_bucket = discretise_rows_fn(rows_result)

    return (c1, c2, sel_bucket, result_bucket)
