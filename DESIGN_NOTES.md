# QueryPilot — Design Notes (Day 1)

## State representation

The episode state is a **set of Components**, each carrying:

| Field | Type | Description |
|-------|------|-------------|
| `tables` | `frozenset[int]` | Which base tables (0-indexed) are in this component |
| `rows` | `int` | Current estimated row count |

### State key for Q-table

State keys must be hashable and order-invariant. We sort components by their
`frozenset` and discretise `rows` into 7 log-10 buckets:

```
bucket 0: rows < 10
bucket 1: 10 ≤ rows < 100
bucket 2: 100 ≤ rows < 1 000
bucket 3: 1 000 ≤ rows < 10 000
bucket 4: 10 000 ≤ rows < 100 000
bucket 5: 100 000 ≤ rows < 1 000 000
bucket 6: rows ≥ 1 000 000
```

**Abstraction caveat**: states that differ only in row-count within the same
bucket are treated identically. This limits per-plan differentiation but
keeps the Q-table tractable for 3–6 tables.

**State space size (upper bound, 6 tables)**:
- Number of distinct partition shapes of {0..5}: 203 (Bell number B₆)
- Row buckets per component: 7
- Upper-bound Q-table entries: 203 × 7^6 ≈ 24 M
- In practice far fewer states are visited per episode

## Action space

At each step with k components, valid actions are pairs (i,j), i<j, such
that at least one join edge connects a base table in component i to one in
component j. The number of candidate pairs is at most k(k−1)/2.

**Key properties:**
- Invalid actions (disconnected component pairs) are **masked** before any
  epsilon-greedy selection.
- The `get_valid_actions()` function re-computes valid pairs each step since
  the component list changes.

## Join connectivity

Queries are generated with a guaranteed-connected spanning tree plus optional
extra edges. This prevents accidental cross-joins. An action joining two
components is legal iff `get_connecting_selectivities()` returns a non-empty
list.

## Cardinality model

```
rows(R ⋈ S) ≈ rows(R) × rows(S) × ∏ selectivities(edge)
```

- All selectivities in [0.01, 0.5] by default.
- Multiple edges between R and S multiply their selectivities.
- Result clamped to [1, 10 000 000].
- This is a **proxy estimator**, not a real engine plan cost.

## Cost model

```
join_cost(R, S) = rows(R) + rows(S) + rows(R ⋈ S)
step_reward     = -join_cost / COST_SCALE   (COST_SCALE = 1 000 000)
```

Used **identically** for RL agent and all baselines. With gamma=1.0,
maximising episode return minimises total modelled join cost.

## Episode termination

An episode ends when exactly 1 component remains (n_tables−1 join steps).

## Known risks

1. State abstraction (row buckets) may merge distinct planning situations
   → limited generalisation.
2. Independence-assumption cardinality can over/under-estimate badly for
   correlated attributes (labelled as proxy in all output).
3. Q-table size grows with Bell number of partition shapes; manageable for
   ≤6 tables but exponential. DQN would be needed beyond 8–10 tables.

---

## Day 3: State & Action Feature Abstraction for Generalisation

### Limitations of Raw Table Indices
1. **Raw table IDs (`frozenset({0, 1})`)**: Bound learned values to arbitrary table labels. If Table 0 is 100 rows in Query A and 1,000,000 rows in Query B, the Q-table cannot transfer knowledge to held-out test queries.
2. **Positional Action Pairs (`(i, j)`)**: Indices into the dynamic component list lack stable semantic meaning across queries or even episode steps.
3. **Missing Graph Topology**: Fails to capture component degree, central hubs vs. leaf tables, or candidate join selectivities.

### Feature-Based Abstraction (`state_representation.py`)
1. **Component Features**:
   `(size, row_bucket, degree)` where:
   - `size`: number of base tables in the component (`|c.tables|`)
   - `row_bucket`: log-10 discretised rows (0–6)
   - `degree`: number of other available components connected to `c` in the current join graph
2. **State Feature**:
   `(n_components, n_active_edges, sorted_tuple_of_component_features)`.
   Completely query-agnostic, hashable, and invariant to component permutation and table indices.
3. **Action Feature**:
   `(feat_1, feat_2, sel_bucket, result_row_bucket)`:
   - Canonical symmetric ordering: `feat_1 <= feat_2`
   - `sel_bucket`: discretised cumulative selectivity of connecting edges (0–3: <0.05, <0.15, <0.30, >=0.30)
   - `result_row_bucket`: log-10 discretised estimated result cardinality
   Directly represents the join type: e.g. "join a small leaf table with a medium hub under high selectivity".

### Trade-Offs
- **Pros (Generalisation & Transferability)**:
  - Enables zero-shot transfer of learned policies to held-out query instances.
  - Compresses state-action space from combinatorial permutations to structural equivalence classes.
  - Learns universal heuristics (e.g. early reduction, joining small selective relations first).
- **Cons (State Aliasing & Resolution Loss)**:
  - Discretisation merges similar cardinalities (e.g. 1,100 vs 9,500 in bucket 3).
  - Degree and immediate selectivity do not capture multi-hop downstream graph paths; two components with identical local features but different distant neighbors will be treated identically.
