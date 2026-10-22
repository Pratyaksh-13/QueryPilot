"""
scripts/run_main_experiments.py
-------------------------------
Day 6 Main Experiment Pipeline for QueryPilot.

Requirements enforced:
  1. 5,000 training episodes per seed with epsilon reaching epsilon_min (0.05)
     at episode 2,500 (well before the end). Fresh query sampled each episode (3-6 tables).
  2. 5 independent training seeds: [101, 102, 103, 104, 105].
  3. Strict seed separation: Training (101-105), Validation (500), Final Test (999).
  4. 100 held-out test queries (25 each of 3, 4, 5, 6 tables) with test seed 999.
  5. Multi-seed reporting: Mean, std, and 95% CI of optimality gaps.
  6. Headline claim excludes 3-table queries (reported on 4-6 table queries).
  7. Learning curves and optimality gap figures saved to results/figures/.
  8. Honest, unskewed reporting of all baselines and RL performance.
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

from src.environment.query_generator import generate_query, QueryInstance
from src.environment.join_environment import JoinEnvironment
from src.agents.q_learning import QLearningAgent, QLearningConfig
from src.baselines.random_agent import run_random_episode
from src.baselines.greedy_agent import run_greedy_episode
from src.baselines.exhaustive import exhaustive_min_cost
from src.evaluation.metrics import optimality_gap, aggregate_values


# ---------------------------------------------------------------------------
# Configuration & Seed Sets
# ---------------------------------------------------------------------------

TRAIN_SEEDS = [101, 102, 103, 104, 105]
VAL_SEED = 500
TEST_SEED = 999

EPISODES = 5000
DECAY_EPISODES = 2500
EPSILON_START = 1.0
EPSILON_MIN = 0.05
ALPHA = 0.1
GAMMA = 1.0
COST_SCALE = 1_000_000
STATE_MODE = "feature"

TABLE_COUNTS = [3, 4, 5, 6]
QUERIES_PER_SIZE = 25  # Total: 100 test queries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_balanced_test_set(seed: int) -> list[QueryInstance]:
    """Generate 100 held-out test queries with equal representation of 3, 4, 5, 6 tables."""
    rng = np.random.default_rng(seed)
    queries = []
    for n_tables in TABLE_COUNTS:
        for _ in range(QUERIES_PER_SIZE):
            q = generate_query(
                rng=rng,
                min_tables=n_tables,
                max_tables=n_tables,
                min_cardinality=100,
                max_cardinality=10_000,
                min_selectivity=0.01,
                max_selectivity=0.5,
            )
            queries.append(q)
    return queries


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


# ---------------------------------------------------------------------------
# Main Experiment Execution
# ---------------------------------------------------------------------------

def run_main_experiments():
    print("=" * 86)
    print(" QueryPilot — Day 6 Main Experiments ")
    print("=" * 86)
    print(f"Training Seeds   : {TRAIN_SEEDS} (5 independent runs)")
    print(f"Validation Seed  : {VAL_SEED}")
    print(f"Final Test Seed  : {TEST_SEED} (strictly held-out, used only once)")
    print(f"Training Setup   : {EPISODES} episodes/seed, linear decay over {DECAY_EPISODES} eps to {EPSILON_MIN}")
    print(f"Test Set         : 100 queries ({QUERIES_PER_SIZE} queries for each of {TABLE_COUNTS} tables)")
    print("-" * 86)

    results_dir = ROOT_DIR / "results"
    checkpoints_dir = results_dir / "checkpoints"
    figures_dir = results_dir / "figures"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Train 5 Agents Across Independent Seeds
    # -----------------------------------------------------------------------
    agents = {}
    training_histories = {}

    print("\n>>> Phase 1: Training 5 RL Agents (5,000 episodes each)...")
    t_train_start = time.perf_counter()

    for seed in TRAIN_SEEDS:
        print(f"\n--- Training Seed {seed} ---")
        cfg = QLearningConfig(
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_min=EPSILON_MIN,
            decay_schedule="linear",
            decay_episodes=DECAY_EPISODES,
            state_mode=STATE_MODE,
            seed=seed,
        )
        agent = QLearningAgent(cfg)
        train_rng = np.random.default_rng(seed)

        history = []
        t0 = time.perf_counter()

        for ep in range(1, EPISODES + 1):
            # Sample fresh random query across 3–6 tables
            n_t = int(train_rng.integers(3, 7))
            q = generate_query(
                rng=train_rng,
                min_tables=n_t,
                max_tables=n_t,
                min_cardinality=100,
                max_cardinality=10_000,
                min_selectivity=0.01,
                max_selectivity=0.5,
            )
            env = JoinEnvironment(q, cost_scale=COST_SCALE, state_mode=STATE_MODE)
            metrics = agent.train_episode(env)
            history.append({
                "episode": ep,
                "return": metrics["return"],
                "cost": metrics["total_cost"],
                "epsilon": metrics["epsilon"],
            })

            if ep % 1000 == 0 or ep == EPISODES:
                recent = history[-500:]
                avg_cost = float(np.mean([m["cost"] for m in recent]))
                avg_ret = float(np.mean([m["return"] for m in recent]))
                print(
                    f"Seed {seed} | Ep {ep:5d}/{EPISODES} | "
                    f"Avg Cost: {avg_cost:11,.0f} | "
                    f"Avg Ret: {avg_ret:7.3f} | "
                    f"Eps: {agent.epsilon:.3f} | "
                    f"Q-entries: {len(agent.q_table):4d}"
                )

        ep_duration = time.perf_counter() - t0
        print(f"Seed {seed} finished in {ep_duration:.2f}s ({EPISODES / ep_duration:.1f} ep/s)")

        # Save checkpoint and log
        agent.save(checkpoints_dir / f"agent_seed_{seed}.json")
        with open(results_dir / f"training_log_seed_{seed}.json", "w", encoding="utf-8") as f:
            json.dump(history, f)

        agents[seed] = agent
        training_histories[seed] = history

    total_train_time = time.perf_counter() - t_train_start
    print(f"\nAll 5 agents trained in {total_train_time:.2f}s total.")

    # -----------------------------------------------------------------------
    # Step 2: Generate 100 Held-Out Test Queries
    # -----------------------------------------------------------------------
    print("\n>>> Phase 2: Generating 100 held-out test queries (Test Seed 999)...")
    test_queries = generate_balanced_test_set(TEST_SEED)
    print(f"Generated {len(test_queries)} queries:")
    for n in TABLE_COUNTS:
        count = sum(1 for q in test_queries if q.n_tables == n)
        print(f"  - {n}-table queries: {count}")

    # -----------------------------------------------------------------------
    # Step 3: Run Baseline and RL Evaluations on Held-Out Queries
    # -----------------------------------------------------------------------
    print("\n>>> Phase 3: Evaluating Random, Greedy, Exhaustive, and 5 RL Seeds...")
    t_eval_start = time.perf_counter()

    query_records = []
    for i, q in enumerate(test_queries):
        record = {
            "query_idx": i,
            "n_tables": q.n_tables,
            "costs": {},
            "gaps": {},
        }

        # 1. Exhaustive optimum
        opt_cost = exhaustive_min_cost(q)
        record["costs"]["exhaustive"] = opt_cost
        record["gaps"]["exhaustive"] = 0.0

        # 2. Random baseline (seeded for reproducibility)
        rand_rng = np.random.default_rng(TEST_SEED + i)
        rand_cost = run_random_episode(q, rng=rand_rng)
        record["costs"]["random"] = rand_cost
        record["gaps"]["random"] = optimality_gap(rand_cost, opt_cost)

        # 3. Greedy baseline
        greedy_cost = run_greedy_episode(q)
        record["costs"]["greedy"] = greedy_cost
        record["gaps"]["greedy"] = optimality_gap(greedy_cost, opt_cost)

        # 4. Each of the 5 RL agent seeds
        rl_costs_for_q = []
        rl_gaps_for_q = []
        for seed in TRAIN_SEEDS:
            agent = agents[seed]
            env = JoinEnvironment(q, state_mode=STATE_MODE)
            res = agent.evaluate_episode(env)
            cost = res["total_cost"]
            gap = optimality_gap(cost, opt_cost)
            record["costs"][f"rl_seed_{seed}"] = cost
            record["gaps"][f"rl_seed_{seed}"] = gap
            rl_costs_for_q.append(cost)
            rl_gaps_for_q.append(gap)

        # Average RL across seeds for this query
        record["costs"]["rl_mean"] = float(np.mean(rl_costs_for_q))
        record["gaps"]["rl_mean"] = float(np.mean(rl_gaps_for_q))
        record["gaps"]["rl_std"] = float(np.std(rl_gaps_for_q))

        query_records.append(record)

    eval_duration = time.perf_counter() - t_eval_start
    print(f"Evaluation of 100 queries completed in {eval_duration:.2f}s.")

    # -----------------------------------------------------------------------
    # Step 4: Statistical Aggregation (Mean, Std, 95% CI across Seeds)
    # -----------------------------------------------------------------------
    # For RL: calculate the metric per seed across queries first, then mean/std across seeds!
    seed_overall_gaps = []
    seed_headline_gaps = []  # Excludes 3-table queries (4, 5, 6 tables)
    seed_by_table_gaps = {n: [] for n in TABLE_COUNTS}

    for seed in TRAIN_SEEDS:
        key = f"rl_seed_{seed}"
        all_gaps = [r["gaps"][key] for r in query_records]
        headline_gaps = [r["gaps"][key] for r in query_records if r["n_tables"] >= 4]

        seed_overall_gaps.append(float(np.mean(all_gaps)))
        seed_headline_gaps.append(float(np.mean(headline_gaps)))

        for n in TABLE_COUNTS:
            n_gaps = [r["gaps"][key] for r in query_records if r["n_tables"] == n]
            seed_by_table_gaps[n].append(float(np.mean(n_gaps)))

    # Compute baseline gaps
    random_all_gaps = [r["gaps"]["random"] for r in query_records]
    random_headline_gaps = [r["gaps"]["random"] for r in query_records if r["n_tables"] >= 4]

    greedy_all_gaps = [r["gaps"]["greedy"] for r in query_records]
    greedy_headline_gaps = [r["gaps"]["greedy"] for r in query_records if r["n_tables"] >= 4]

    # CI helper (Student's t or 1.96 for n=5)
    # For n=5, t_{0.025, df=4} = 2.776
    T_CRIT = 2.776
    def compute_ci(data):
        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
        ci95 = T_CRIT * std / np.sqrt(len(data))
        return mean, std, ci95

    rl_overall_mean, rl_overall_std, rl_overall_ci = compute_ci(seed_overall_gaps)
    rl_headline_mean, rl_headline_std, rl_headline_ci = compute_ci(seed_headline_gaps)

    # -----------------------------------------------------------------------
    # Step 5: Report Results Tables
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" HEADLINE CLAIM COMPARISON: 4–6 TABLE QUERIES (3-Table Queries Excluded) ")
    print("=" * 90)
    h_headers = ["Method", "Queries", "Mean Gap (%)", "Std / Variability", "95% Confidence Interval"]
    h_widths = [14, 9, 14, 19, 26]
    print(format_table_row(h_headers, h_widths))
    print("-" * 90)

    print(format_table_row(["EXHAUSTIVE", "75", "0.00%", "± 0.00%", "[0.00%, 0.00%] (Oracle)"], h_widths))
    print(format_table_row(["GREEDY", "75", f"{np.mean(greedy_headline_gaps):.2f}%", f"± {np.std(greedy_headline_gaps):.2f}%", f"[{np.mean(greedy_headline_gaps):.2f}%] (Deterministic)"], h_widths))
    print(format_table_row(["RANDOM", "75", f"{np.mean(random_headline_gaps):.2f}%", f"± {np.std(random_headline_gaps):.2f}%", f"[{np.mean(random_headline_gaps):.2f}%]"], h_widths))
    print(format_table_row([
        "RL (5 SEEDS)",
        "75",
        f"{rl_headline_mean:.2f}%",
        f"± {rl_headline_std:.2f}%",
        f"[{rl_headline_mean - rl_headline_ci:.2f}%, {rl_headline_mean + rl_headline_ci:.2f}%]",
    ], h_widths))
    print("-" * 90)

    print("\n" + "=" * 90)
    print(" DETAILED BREAKDOWN BY QUERY SIZE (TABLE COUNT) ")
    print("=" * 90)
    b_headers = ["Size", "Method", "Queries", "Mean Cost", "Median Cost", "Mean Gap (%)", "RL Seed Std / CI"]
    b_widths = [8, 14, 9, 13, 13, 13, 16]
    print(format_table_row(b_headers, b_widths))
    print("-" * 90)

    for n in TABLE_COUNTS:
        sub_records = [r for r in query_records if r["n_tables"] == n]
        n_queries = len(sub_records)

        # Baseline costs & gaps
        ex_costs = [r["costs"]["exhaustive"] for r in sub_records]
        gr_costs = [r["costs"]["greedy"] for r in sub_records]
        rd_costs = [r["costs"]["random"] for r in sub_records]

        gr_gaps = [r["gaps"]["greedy"] for r in sub_records]
        rd_gaps = [r["gaps"]["random"] for r in sub_records]

        # RL across seeds for this table count
        rl_n_mean, rl_n_std, rl_n_ci = compute_ci(seed_by_table_gaps[n])
        rl_n_costs = [r["costs"]["rl_mean"] for r in sub_records]

        print(format_table_row([f"{n} tbl", "EXHAUSTIVE", n_queries, f"{np.mean(ex_costs):,.0f}", f"{np.median(ex_costs):,.0f}", "0.00%", "—"], b_widths))
        print(format_table_row(["", "GREEDY", n_queries, f"{np.mean(gr_costs):,.0f}", f"{np.median(gr_costs):,.0f}", f"{np.mean(gr_gaps):.2f}%", "—"], b_widths))
        print(format_table_row(["", "RANDOM", n_queries, f"{np.mean(rd_costs):,.0f}", f"{np.median(rd_costs):,.0f}", f"{np.mean(rd_gaps):.2f}%", "—"], b_widths))
        print(format_table_row([
            "",
            "RL (5 SEEDS)",
            n_queries,
            f"{np.mean(rl_n_costs):,.0f}",
            f"{np.median(rl_n_costs):,.0f}",
            f"{rl_n_mean:.2f}%",
            f"±{rl_n_std:.2f}% (CI:{rl_n_ci:.2f}%)",
        ], b_widths))
        print("-" * 90)

    # -----------------------------------------------------------------------
    # Step 6: Generate and Save Figures
    # -----------------------------------------------------------------------
    print("\n>>> Phase 4: Generating and saving figures to results/figures/...")

    # Figure 1: Learning Curves across 5 Seeds
    plt.figure(figsize=(10, 5.5), dpi=150)
    ax1 = plt.subplot(1, 1, 1)

    # Calculate rolling mean return across episodes for each seed
    window = 100
    all_smoothed_returns = []

    for seed in TRAIN_SEEDS:
        rets = [m["return"] for m in training_histories[seed]]
        # Rolling average
        smoothed = np.convolve(rets, np.ones(window) / window, mode="valid")
        all_smoothed_returns.append(smoothed)
        ax1.plot(
            range(window, EPISODES + 1),
            smoothed,
            alpha=0.35,
            linewidth=1.2,
            label=f"Seed {seed}",
        )

    all_smoothed_returns = np.array(all_smoothed_returns)
    mean_curve = np.mean(all_smoothed_returns, axis=0)
    std_curve = np.std(all_smoothed_returns, axis=0)
    eps_x = range(window, EPISODES + 1)

    ax1.plot(eps_x, mean_curve, color="darkblue", linewidth=2.5, label="Mean (5 Seeds)")
    ax1.fill_between(
        eps_x,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color="royalblue",
        alpha=0.2,
        label="±1 Std Dev",
    )

    ax1.set_xlabel("Training Episode", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Episode Return (-Cost / 1M)", fontsize=11, fontweight="bold")
    ax1.set_title("QueryPilot Q-Learning Training Curves Across 5 Seeds (5,000 Episodes)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot epsilon decay on twin axis
    ax2 = ax1.twinx()
    eps_vals = [
        max(EPSILON_MIN, EPSILON_START - (EPSILON_START - EPSILON_MIN) * (e / DECAY_EPISODES))
        if e <= DECAY_EPISODES else EPSILON_MIN
        for e in eps_x
    ]
    ax2.plot(eps_x, eps_vals, color="darkred", linestyle=":", linewidth=1.8, label="Epsilon (Decay)")
    ax2.set_ylabel("Exploration Rate (ε)", color="darkred", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="darkred")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.9)

    fig1_path = figures_dir / "learning_curves.png"
    plt.tight_layout()
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Optimality Gap by Table Count
    plt.figure(figsize=(9, 5.5), dpi=150)
    categories = ["3 Tables", "4 Tables", "5 Tables", "6 Tables", "Headline\n(4–6 Tables)"]
    x = np.arange(len(categories))
    width = 0.22

    # Baseline means
    rand_vals = [
        float(np.mean([r["gaps"]["random"] for r in query_records if r["n_tables"] == n]))
        for n in TABLE_COUNTS
    ] + [float(np.mean(random_headline_gaps))]

    greedy_vals = [
        float(np.mean([r["gaps"]["greedy"] for r in query_records if r["n_tables"] == n]))
        for n in TABLE_COUNTS
    ] + [float(np.mean(greedy_headline_gaps))]

    rl_vals = [float(np.mean(seed_by_table_gaps[n])) for n in TABLE_COUNTS] + [rl_headline_mean]
    rl_errs = [float(np.std(seed_by_table_gaps[n])) for n in TABLE_COUNTS] + [rl_headline_std]

    plt.bar(x - width, rand_vals, width, label="Random", color="#e74c3c", alpha=0.85)
    plt.bar(x, greedy_vals, width, label="Greedy", color="#2ecc71", alpha=0.85)
    plt.bar(x + width, rl_vals, width, yerr=rl_errs, capsize=4, label="RL (5 Seeds ± std)", color="#3498db", alpha=0.85)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Query Complexity", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Optimality Gap (%) vs Exhaustive", fontsize=11, fontweight="bold")
    plt.title("Optimality Gap Across Query Sizes on 100 Held-Out Queries (Seed 999)", fontsize=12, fontweight="bold")
    plt.xticks(x, categories, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(framealpha=0.9)

    fig2_path = figures_dir / "optimality_gap_by_size.png"
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved: {fig2_path}")

    # -----------------------------------------------------------------------
    # Step 7: Save Comprehensive JSON Summary
    # -----------------------------------------------------------------------
    summary = {
        "setup": {
            "train_seeds": TRAIN_SEEDS,
            "val_seed": VAL_SEED,
            "test_seed": TEST_SEED,
            "episodes": EPISODES,
            "decay_episodes": DECAY_EPISODES,
            "epsilon_start": EPSILON_START,
            "epsilon_min": EPSILON_MIN,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "state_mode": STATE_MODE,
            "num_test_queries": len(test_queries),
        },
        "headline_4_to_6_tables": {
            "greedy": {
                "mean_gap": float(np.mean(greedy_headline_gaps)),
                "std_gap": float(np.std(greedy_headline_gaps)),
            },
            "random": {
                "mean_gap": float(np.mean(random_headline_gaps)),
                "std_gap": float(np.std(random_headline_gaps)),
            },
            "rl": {
                "mean_gap": rl_headline_mean,
                "std_gap": rl_headline_std,
                "ci95": rl_headline_ci,
                "seed_gaps": seed_headline_gaps,
            },
        },
        "overall_all_tables": {
            "greedy": {
                "mean_gap": float(np.mean(greedy_all_gaps)),
                "std_gap": float(np.std(greedy_all_gaps)),
            },
            "random": {
                "mean_gap": float(np.mean(random_all_gaps)),
                "std_gap": float(np.std(random_all_gaps)),
            },
            "rl": {
                "mean_gap": rl_overall_mean,
                "std_gap": rl_overall_std,
                "ci95": rl_overall_ci,
                "seed_gaps": seed_overall_gaps,
            },
        },
        "by_table_count": {
            n: {
                "greedy_mean_gap": float(np.mean([r["gaps"]["greedy"] for r in query_records if r["n_tables"] == n])),
                "random_mean_gap": float(np.mean([r["gaps"]["random"] for r in query_records if r["n_tables"] == n])),
                "rl_mean_gap": float(np.mean(seed_by_table_gaps[n])),
                "rl_std_gap": float(np.std(seed_by_table_gaps[n])),
            }
            for n in TABLE_COUNTS
        },
        "query_records": query_records,
    }

    summary_path = results_dir / "main_experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved full experiment summary to: {summary_path}")
    print("=" * 86)
    print("Main experiments completed successfully!")


if __name__ == "__main__":
    run_main_experiments()
