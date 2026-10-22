"""
scripts/evaluate.py
-------------------
Command-line evaluation script for QueryPilot.

Evaluates:
  1. Random Baseline
  2. Greedy Baseline
  3. Exhaustive Oracle (for <= 6 tables)
  4. QueryPilot RL Agent (if checkpoint provided)

All methods are evaluated on identical held-out test queries generated from a
strictly separate test seed.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --agent-path results/q_table.json --num-queries 50 --test-seed 99
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import yaml

from src.environment.query_generator import make_query_batch
from src.agents.q_learning import QLearningAgent
from src.evaluation.benchmark import run_benchmark


def load_config(config_path: Path) -> dict:
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def format_table_row(cols, widths, alignments=None):
    if alignments is None:
        alignments = ["<"] + [">"] * (len(cols) - 1)
    formatted = []
    for col, width, align in zip(cols, widths, alignments):
        if align == "<":
            formatted.append(f"{str(col):<{width}}")
        else:
            formatted.append(f"{str(col):>{width}}")
    return " | ".join(formatted)


def evaluate(
    agent_path: Optional[Path],
    num_queries: int,
    test_seed: int,
    config_path: Path,
    output_path: Path,
    include_exhaustive: bool = True,
):
    print("=" * 78)
    print(" QueryPilot Comparative Evaluation ")
    print("=" * 78)
    print(f"Test Seed        : {test_seed} (strictly held-out)")
    print(f"Num Test Queries : {num_queries}")
    print(f"Agent Checkpoint : {agent_path}")
    print(f"Output Path      : {output_path}")
    print("-" * 78)

    cfg = load_config(config_path)
    env_cfg = cfg.get("environment", {})
    min_tables = env_cfg.get("min_tables", 3)
    max_tables = env_cfg.get("max_tables", 6)
    min_card = env_cfg.get("min_cardinality", 100)
    max_card = env_cfg.get("max_cardinality", 10_000)
    min_sel = env_cfg.get("min_selectivity", 0.01)
    max_sel = env_cfg.get("max_selectivity", 0.5)

    # 1. Load trained agent if present
    agent = None
    if agent_path and agent_path.is_file():
        agent = QLearningAgent.load(agent_path)
        print(f"Loaded agent: state_mode={agent.config.state_mode}, Q-table entries={len(agent.q_table)}")
    elif agent_path:
        print(f"Warning: Agent checkpoint not found at {agent_path}. Running baselines only.")

    # 2. Generate held-out test queries
    test_queries = make_query_batch(
        n_queries=num_queries,
        seed=test_seed,
        min_tables=min_tables,
        max_tables=max_tables,
        min_cardinality=min_card,
        max_cardinality=max_card,
        min_selectivity=min_sel,
        max_selectivity=max_sel,
    )

    # 3. Run Benchmark
    benchmark_res = run_benchmark(
        queries=test_queries,
        agent=agent,
        include_exhaustive=include_exhaustive,
        base_seed=test_seed,
    )

    # 4. Display Overall Summary Table
    headers = ["Method", "Mean Cost", "Std Cost", "Median Cost", "Mean Gap (%)", "Time/Query (ms)"]
    widths = [14, 13, 13, 13, 14, 17]

    print("\n" + "=" * 90)
    print(" OVERALL PERFORMANCE SUMMARY (Held-Out Test Queries) ")
    print("=" * 90)
    print(format_table_row(headers, widths))
    print("-" * 90)

    for method in benchmark_res["methods"]:
        m_cost = benchmark_res["overall"][method]["cost"]
        m_gap = benchmark_res["overall"][method]["gap_percent"]
        m_time = benchmark_res["overall"][method]["time_sec"]

        cost_mean = f"{m_cost['mean']:,.0f}"
        cost_std = f"{m_cost['std']:,.0f}"
        cost_med = f"{m_cost['median']:,.0f}"
        gap_str = f"{m_gap['mean']:.2f}%" if m_gap else "N/A"
        time_ms = f"{m_time['mean'] * 1000.0:.2f} ms"

        row = [method.upper(), cost_mean, cost_std, cost_med, gap_str, time_ms]
        print(format_table_row(row, widths))
    print("-" * 90)

    # 5. Display Breakdown by Query Size (Table Count)
    print("\n" + "=" * 90)
    print(" PERFORMANCE BREAKDOWN BY QUERY SIZE (TABLE COUNT) ")
    print("=" * 90)
    sub_headers = ["Tables", "Method", "Count", "Mean Cost", "Median Cost", "Mean Gap (%)"]
    sub_widths = [8, 14, 7, 14, 14, 14]
    print(format_table_row(sub_headers, sub_widths))
    print("-" * 90)

    for n_tbl in sorted(benchmark_res["by_table_count"].keys()):
        tbl_data = benchmark_res["by_table_count"][n_tbl]
        first = True
        for method in benchmark_res["methods"]:
            if method not in tbl_data:
                continue
            m_cost = tbl_data[method]["cost"]
            m_gap = tbl_data[method]["gap_percent"]
            count = tbl_data[method]["count"]
            gap_str = f"{m_gap['mean']:.2f}%" if m_gap else "N/A"

            tbl_label = f"{n_tbl} tbl" if first else ""
            first = False

            row = [
                tbl_label,
                method.upper(),
                count,
                f"{m_cost['mean']:,.0f}",
                f"{m_cost['median']:,.0f}",
                gap_str,
            ]
            print(format_table_row(row, sub_widths))
        print("-" * 90)

    # 6. Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_res, f, indent=2)
    print(f"\nSaved evaluation benchmark results to: {output_path}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Evaluate QueryPilot RL agent and baselines.")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--agent-path", type=str, default="results/q_table.json", help="Path to trained agent JSON")
    parser.add_argument("--test-seed", type=int, default=99, help="Held-out evaluation seed")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of held-out test queries")
    parser.add_argument("--output-path", type=str, default="results/eval_results.json", help="Path to save evaluation results")
    parser.add_argument("--no-exhaustive", action="store_true", help="Skip exhaustive baseline")

    args = parser.parse_args()

    evaluate(
        agent_path=Path(args.agent_path) if args.agent_path else None,
        num_queries=args.num_queries,
        test_seed=args.test_seed,
        config_path=Path(args.config),
        output_path=Path(args.output_path),
        include_exhaustive=not args.no_exhaustive,
    )


if __name__ == "__main__":
    main()
