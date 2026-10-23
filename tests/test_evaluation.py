"""
tests/test_evaluation.py
------------------------
Tests for metrics, query evaluation, and benchmark harness.
"""

import pytest
import numpy as np

from src.environment.query_generator import QueryInstance, make_query_batch
from src.evaluation.metrics import (
    optimality_gap, aggregate_values, bootstrap_ci, paired_comparison
)
from src.evaluation.benchmark import evaluate_query, run_benchmark
from src.agents.q_learning import QLearningAgent, QLearningConfig


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_optimality_gap_zero_when_equal(self):
        assert optimality_gap(100.0, 100.0) == 0.0

    def test_optimality_gap_percentage(self):
        # 150 vs opt 100 is a 50% gap
        assert pytest.approx(optimality_gap(150.0, 100.0)) == 50.0

    def test_optimality_gap_zero_opt_safe(self):
        assert optimality_gap(0.0, 0.0) == 0.0
        assert optimality_gap(100.0, 0.0) == float("inf")

    def test_aggregate_values_empty(self):
        res = aggregate_values([])
        assert res["count"] == 0
        assert res["mean"] == 0.0

    def test_aggregate_values_statistics(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        res = aggregate_values(vals)
        assert res["count"] == 5
        assert pytest.approx(res["mean"]) == 30.0
        assert pytest.approx(res["median"]) == 30.0
        assert res["min"] == 10.0
        assert res["max"] == 50.0

    def test_bootstrap_ci_contains_true_median(self):
        """95% CI should contain the true median across many bootstrap runs."""
        rng = np.random.default_rng(0)
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        obs, lo, hi = bootstrap_ci(data, stat_fn=np.median, n_boot=500, rng=rng)
        assert obs == pytest.approx(4.0)  # true median
        assert lo <= obs <= hi
        assert lo < hi  # non-degenerate

    def test_bootstrap_ci_single_value(self):
        obs, lo, hi = bootstrap_ci([42.0], n_boot=100)
        assert obs == 42.0 and lo == 42.0 and hi == 42.0

    def test_paired_comparison_win_loss_counts(self):
        # A beats B on every query
        gaps_a = [0.0, 0.0, 0.0]
        gaps_b = [10.0, 20.0, 30.0]
        res = paired_comparison(gaps_a, gaps_b)
        assert res["wins_a"] == 3
        assert res["losses_a"] == 0
        assert res["ties"] == 0
        assert res["median_diff"] < 0  # A gap < B gap => negative diff

    def test_paired_comparison_all_ties(self):
        gaps = [5.0, 10.0, 15.0]
        res = paired_comparison(gaps, gaps)
        assert res["ties"] == 3
        assert res["wins_a"] == 0
        assert res["losses_a"] == 0
        assert res["median_diff"] == pytest.approx(0.0)

    def test_paired_comparison_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_comparison([1.0, 2.0], [3.0])


# ---------------------------------------------------------------------------
# Benchmark & Query Evaluation tests
# ---------------------------------------------------------------------------

class TestBenchmark:
    @pytest.fixture
    def sample_queries(self):
        return make_query_batch(n_queries=4, seed=100, min_tables=3, max_tables=4)

    def test_evaluate_single_query(self, sample_queries):
        q = sample_queries[0]
        res = evaluate_query(q, agent=None, include_exhaustive=True, seed=42)

        assert "costs" in res
        assert "times" in res
        assert "gaps" in res
        assert "random" in res["costs"]
        assert "greedy" in res["costs"]
        assert "exhaustive" in res["costs"]

        # Exhaustive must be <= greedy
        assert res["costs"]["exhaustive"] <= res["costs"]["greedy"]
        assert res["gaps"]["exhaustive"] == 0.0
        assert res["gaps"]["greedy"] >= 0.0

    def test_benchmark_with_agent(self, sample_queries):
        agent = QLearningAgent(QLearningConfig(epsilon_start=0.1, seed=42))
        res = run_benchmark(sample_queries, agent=agent, include_exhaustive=True)

        assert res["num_queries"] == 4
        assert "random" in res["methods"]
        assert "greedy" in res["methods"]
        assert "exhaustive" in res["methods"]
        assert "rl" in res["methods"]

        # Check overall summary
        assert "overall" in res
        for m in res["methods"]:
            assert "cost" in res["overall"][m]
            assert res["overall"][m]["cost"]["count"] == 4

        # Check table count grouping
        assert "by_table_count" in res
        for n in res["by_table_count"]:
            assert isinstance(n, int)
            for m in res["methods"]:
                assert "cost" in res["by_table_count"][n][m]
