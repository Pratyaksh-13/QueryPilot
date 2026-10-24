# QueryPilot

Reinforcement learning for database query optimization: choosing the join order for a multi-table query with tabular Q-learning, compared against Random, Greedy and Exhaustive (optimal) baselines on a custom simulator.

**Headline result.** The learned policy is far better than random but does **not** beat a simple greedy heuristic. On held-out 4 to 6 table queries it closes about 55% of the distance from Random to Greedy, and loses to Greedy on 56 of 75 queries. The repository includes the diagnostics, two negative-result experiments and the limitations of the simulator that explain this.

## Results

Held-out test set (seed 888, 25 queries per table count). Values are the mean optimality gap in percent, `100 * (C_method - C_opt) / C_opt`, where `C_opt` comes from exhaustive search. Lower is better. RL is the mean over 5 independently trained agents; "sd" is the sample standard deviation across those 5 training seeds.

| Tables | n | Random | Greedy | RL (mean of 5 seeds) | RL sd |
|---|---|---|---|---|---|
| 3 | 25 | 23.72 | 0.00 | 2.68 | 1.07 |
| 4 | 25 | 40.45 | 9.99 | 22.23 | 4.18 |
| 5 | 25 | 33.50 | 4.93 | 17.34 | 4.22 |
| 6 | 25 | 42.37 | 9.85 | 25.96 | 3.51 |
| **4 to 6** | 75 | **38.77** | **8.26** | **21.84** | **2.87** |

The headline row excludes 3-table queries, which are nearly trivial. Exhaustive search is the oracle (0% by definition) and is used for evaluation only, never for training.

Paired per-query comparison on the 75 queries with 4 to 6 tables (RL gap averaged over seeds, minus Greedy gap, in percentage points):

| Statistic | Value |
|---|---|
| Mean difference | 13.59 (95% bootstrap CI 8.86 to 19.09) |
| Median difference | 7.82 (95% bootstrap CI 5.55 to 10.68) |
| RL better / tied / worse | 8 / 11 / 56 |
| Share of Random-to-Greedy distance closed by RL | 55.5% |

All of these are recomputed from saved per-query records by `scripts/summarize_test_results.py` and written to `results/final_test_summary.json`.

Figures: `results/figures/test_gap_by_size.png` (the table above, test seed 888). `results/figures/val_gap_curve_baseline.png` shows the validation gap (seed 500) during training, and `results/figures/optimality_gap_baseline.png` shows gap by query size on the validation set. The `*_fallback.png` figures are the same plots for the greedy-fallback variant.

## Experimental protocol

- **Training:** 5 seeds (101 to 105), 5,000 episodes each. Every episode uses a freshly generated query with 3 to 6 tables. Epsilon decays linearly from 1.0 to 0.05 over the first 2,500 episodes; alpha = 0.1, gamma = 1.0.
- **Validation seed 500:** used for every decision (diagnostics, choosing between variants).
- **Test seed 888:** a fresh seed used for the final evaluation of the chosen configuration. Earlier seeds (99 and 999) were looked at during development and are not used for conclusions. The ablation and diagnostic variants were evaluated on validation only.

## Design

- **Query generator** (`src/environment/query_generator.py`): 3 to 6 tables, 100 to 10,000 rows per table, a random spanning tree plus optional extra edges (so the join graph is always connected), selectivities in [0.01, 0.5].
- **Cardinality and cost** (`cardinality.py`, `cost_model.py`): independence assumption, `rows(R join S) = rows(R) * rows(S) * product of selectivities`, clamped to [1, 1e7]. Cost of a join is `rows_r + rows_s + rows_result`, including the final join. Reward is `-cost / 1e6`. One cost model is shared by the agent and every baseline.
- **Environment** (`join_environment.py`): state is the set of current components (base tables or intermediate results). An action joins two components that share at least one join edge; invalid pairs are masked. An episode ends when one component remains.
- **State and action abstraction** (`state_representation.py`): query-agnostic so the Q-table can transfer to unseen queries. A component is `(size, row_bucket, degree)` with 7 decade-wide row buckets. The state key is `(n_components, n_active_edges, sorted component features)`. The action key is the canonical pair of component features plus a selectivity bucket (4 levels) and a result-size bucket.
- **Agent** (`src/agents/q_learning.py`): tabular Q-learning with epsilon-greedy exploration, valid-action masking, no bootstrapping on terminal transitions, and JSON save/load.
- **Baselines** (`src/baselines/`): Random, Greedy (lowest immediate join cost), Exhaustive (optimal, small queries only).

## Setup

Python 3.12 or newer. Tested on Windows (Python 3.14, PowerShell) and Linux (Python 3.12).

```bash
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# Linux / macOS:       source .venv/bin/activate
pip install -r requirements.txt
python -m pytest        # 122 tests
```

## Reproducing the results

Run from the repository root, in this order. Later steps read checkpoints written by step 1.

```bash
# 1. Train 5 agents (Run A: pure RL, and Run B: greedy-fallback variant),
#    evaluate on validation seed 500 and test seed 888, write figures and
#    results/main_experiment_summary.json. This overwrites checkpoints and logs.
python scripts/run_main_experiments.py

# 2. Recompute the headline test numbers from the saved per-query records
python scripts/summarize_test_results.py        # -> results/final_test_summary.json, results/figures/test_gap_by_size.png

# 3. Replay the worst queries and check the cardinality clamp
python scripts/failure_analysis.py              # -> results/failure_analysis.json, results/clamp_analysis.json

# 4. Ablation: 14 half-decade row buckets instead of 7 (validation seed 500 only)
python scripts/ablation_bins14.py               # -> results/ablation_bins14.json
```

A quick smoke test of the single-run command-line tools (these write `results/q_table.json`, `results/training_log.json` and `results/eval_results.json`, which are from this small test and are not used for any reported number):

```bash
python scripts/train.py --episodes 1000 --seed 42 --save-path results/q_table.json
python scripts/evaluate.py --agent-path results/q_table.json --num-queries 50 --test-seed 99
```

Checkpoints for the greedy-fallback run and the 14-bucket ablation are not committed (they are listed in `.gitignore`); rerunning steps 1 and 4 regenerates them.

## Findings

1. **RL beats Random but not Greedy.** About 20% of the policy's steps on validation queries chose an action that was never updated during training (17% to 22% per seed), and the learned policy loses to Greedy on most queries.
2. **A greedy fallback made things worse.** Restricting the policy to trained actions and using the cheapest one-step join otherwise raised the mean validation gap from 14.06% (pure RL) to 18.42%. This variant also uses the exact one-step cost that Greedy uses, so it is a hybrid, not pure RL. Pure RL is the reported method.
3. **Finer row buckets did not help.** With 14 buckets the mean validation gap was 19.42% (sd 2.01 across seeds) versus 13.95% (sd 1.99) for the 7-bucket agents recomputed from the saved checkpoints, and it was worse on all five seeds. This is consistent with a sparse Q-table (the same 5,000 episodes spread over more states), but sparsity was not tested directly.
4. **Failure replays (q40, q88, q91 of the test set; diagnostic only).** In the replays, an unseen action with the default value of 0 outranked trained actions with negative values (all rewards are non-positive), five different joins looked identical to the agent in one state, and the root state of 6-table queries was often never seen in training. These are hypotheses drawn from three queries, not established causes. The five worst test queries are q40, q88, q33, q49 and q28.

## Limitations

- **The cardinality clamp saturates the simulator.** With base tables of up to 10,000 rows and selectivities up to 0.5, joins grow very quickly: the final result hits the 1e7 clamp in 100 of 100 test queries, and 38% (Greedy) to 47% (RL, seed 101) of intermediate results do as well. Every plan therefore pays a constant final-join cost, which also compresses the reported percentage gaps. A peak-cardinality metric is not informative here and is not reported.
- **Evaluation is not bit-for-bit repeatable.** Ties between equal Q-values are broken with the agent's random number generator, so re-evaluating a loaded checkpoint can change individual query results slightly (for example, mean validation gap 14.06% in the original run versus 13.95% when recomputed from the checkpoints). Headline numbers are means over 5 seeds and 75 queries; single-query results should be read with this in mind.
- **Synthetic queries and a simplified cost model.** Results do not claim to transfer to a real database engine.
- **3-table queries are not solved perfectly** (2.68% mean gap for RL against 0% for Greedy). This was not investigated further.
- **One test seed.** The test set has 100 queries from a single seed.

## Repository layout

```
config/            default configuration
src/environment/   query generator, cardinality, cost model, environment, state abstraction
src/agents/        tabular Q-learning agent
src/baselines/     random, greedy, exhaustive
src/evaluation/    metrics (optimality gap, bootstrap CI, paired comparison) and benchmark harness
scripts/           train, evaluate, main experiments, failure analysis, ablation, summary
tests/             unit tests
results/           saved logs, summaries, diagnostics, figures, Run A checkpoints
```
