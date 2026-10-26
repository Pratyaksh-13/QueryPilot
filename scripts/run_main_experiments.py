"""
scripts/run_main_experiments.py
-------------------------------
Full experiment pipeline for QueryPilot (train, validate, test, figures).

Seed discipline:
  Training  : [101, 102, 103, 104, 105]
  Validation: 500  - used for diagnostics and for choosing between variants
  Final test: 888  - used once at the end, never for tuning

What it does:
  - Run A (pure tabular Q-learning, the reported method): trains 5 agents,
    records the greedy-policy validation gap every 500 episodes, runs
    diagnostics (unseen state-action rate, action-key aliasing, paired
    RL-vs-Greedy comparison) on the validation set.
  - Run B (greedy-fallback variant): a hybrid that uses the cheapest one-step
    join when no trained action is available. It is an experiment, not the
    reported method. In our runs it made validation worse (about 18.4% vs 14.1%).
  - The run with the lower mean validation gap is evaluated once on test seed 888.
    Run A wins, so the reported results are pure RL.
  - Writes checkpoints, training logs, figures and
    results/main_experiment_summary.json.
"""

import json
import sys
import time
from pathlib import Path

# Ensure safe printing on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment.query_generator import generate_query, QueryInstance
from src.environment.join_environment import JoinEnvironment, get_valid_actions
from src.environment.cardinality import get_connecting_selectivities, estimate_join_cardinality
from src.environment.cost_model import join_cost
from src.agents.q_learning import QLearningAgent, QLearningConfig
from src.baselines.random_agent import run_random_episode
from src.baselines.greedy_agent import run_greedy_episode
from src.baselines.exhaustive import exhaustive_min_cost
from src.evaluation.metrics import optimality_gap, aggregate_values, paired_comparison, bootstrap_ci


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_SEEDS    = [101, 102, 103, 104, 105]
VAL_SEED       = 500   # ONLY used for diagnostics / tuning decisions
FINAL_TEST_SEED = 888  # Used once at the very end, never for decisions

EPISODES        = 5000
DECAY_EPISODES  = 2500
EPSILON_START   = 1.0
EPSILON_MIN     = 0.05
ALPHA           = 0.1
GAMMA           = 1.0
COST_SCALE      = 1_000_000
STATE_MODE      = "feature"

TABLE_COUNTS    = [3, 4, 5, 6]
QUERIES_PER_SIZE = 25   # 100 total per set

# How often (in episodes) to snapshot greedy-policy val gap during training
VAL_EVAL_EVERY  = 500


# ---------------------------------------------------------------------------
# Query generation helpers
# ---------------------------------------------------------------------------

def make_query_set(seed: int, counts=TABLE_COUNTS, per_size=QUERIES_PER_SIZE) -> list:
    """Generate a balanced set of queries across table counts."""
    rng = np.random.default_rng(seed)
    queries = []
    for n in counts:
        for _ in range(per_size):
            queries.append(generate_query(
                rng=rng, min_tables=n, max_tables=n,
                min_cardinality=100, max_cardinality=10_000,
                min_selectivity=0.01, max_selectivity=0.5,
            ))
    return queries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(cols, widths):
    return " | ".join(f"{str(c):>{w}}" if i > 0 else f"{str(c):<{w}}"
                      for i, (c, w) in enumerate(zip(cols, widths)))


def compute_ci(data, t_crit=2.776):
    """95% CI using t_{0.025, df=4} = 2.776 for n=5 seeds."""
    m = float(np.mean(data))
    s = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
    return m, s, t_crit * s / np.sqrt(len(data))


def eval_val_gap(agent: QLearningAgent, val_queries: list) -> float:
    """Return mean optimality gap on val set using greedy policy (no exploration)."""
    gaps = []
    for q in val_queries:
        opt = exhaustive_min_cost(q)
        env = JoinEnvironment(q, state_mode=agent.config.state_mode)
        res = agent.evaluate_episode(env)
        gaps.append(optimality_gap(res["total_cost"], opt))
    return float(np.mean(gaps))


def greedy_with_peak(query: QueryInstance) -> dict:
    """Greedy baseline, also records peak intermediate cardinality."""
    env = JoinEnvironment(query)
    env.reset()
    peak = 0
    while not env.terminated:
        components = env.components
        q = env.query
        best, best_c = None, float("inf")
        for i, j in get_valid_actions(components, q):
            sels = get_connecting_selectivities(
                components[i].tables, components[j].tables, q.selectivities, q.edges)
            rows_r = estimate_join_cardinality(components[i].rows, components[j].rows, sels)
            c = join_cost(components[i].rows, components[j].rows, rows_r)
            if c < best_c:
                best_c = c
                best = (i, j)
        _, _, _, _, info = env.step(best)
        peak = max(peak, info.get("rows_result", 0))
    return {"total_cost": env.total_cost, "peak": peak}


# ---------------------------------------------------------------------------
# Phase 1: Train agents with periodic val-gap snapshots
# ---------------------------------------------------------------------------

def train_agents(val_queries: list, use_fallback: bool = False) -> tuple:
    label = "with greedy_fallback" if use_fallback else "baseline"
    print(f"\n>>> Phase 1: Training 5 agents ({EPISODES} eps each) [{label}]")
    t0 = time.perf_counter()

    agents, histories, val_curves = {}, {}, {}

    for seed in TRAIN_SEEDS:
        print(f"\n  --- Seed {seed} ---")
        cfg = QLearningConfig(
            alpha=ALPHA, gamma=GAMMA,
            epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN,
            decay_schedule="linear", decay_episodes=DECAY_EPISODES,
            state_mode=STATE_MODE, seed=seed,
            greedy_fallback=use_fallback,
        )
        agent = QLearningAgent(cfg)
        train_rng = np.random.default_rng(seed)

        history = []
        val_gap_curve = []  # (episode, mean_val_gap)

        for ep in range(1, EPISODES + 1):
            n_t = int(train_rng.integers(3, 7))
            q = generate_query(
                rng=train_rng, min_tables=n_t, max_tables=n_t,
                min_cardinality=100, max_cardinality=10_000,
                min_selectivity=0.01, max_selectivity=0.5,
            )
            env = JoinEnvironment(q, cost_scale=COST_SCALE, state_mode=STATE_MODE)
            m = agent.train_episode(env)
            history.append({"episode": ep, "return": m["return"],
                            "cost": m["total_cost"], "epsilon": m["epsilon"]})

            # Snapshot val gap periodically
            if ep % VAL_EVAL_EVERY == 0 or ep == EPISODES:
                vg = eval_val_gap(agent, val_queries)
                val_gap_curve.append((ep, vg))
                if ep % 1000 == 0 or ep == EPISODES:
                    recent = history[-500:]
                    print(f"    Ep {ep:5d} | avg_cost {np.mean([h['cost'] for h in recent]):>10,.0f}"
                          f" | eps {agent.epsilon:.3f} | val_gap {vg:.2f}%"
                          f" | Q-entries {len(agent.q_table):,}")

        tag = "_fallback" if use_fallback else ""
        ck_path = ROOT_DIR / "results" / "checkpoints" / f"agent{tag}_seed_{seed}.json"
        agent.save(ck_path)
        log_path = ROOT_DIR / "results" / f"training_log{tag}_seed_{seed}.json"
        with open(log_path, "w") as f:
            json.dump(history, f)

        agents[seed] = agent
        histories[seed] = history
        val_curves[seed] = val_gap_curve

    print(f"\n  All agents trained in {time.perf_counter() - t0:.1f}s")
    return agents, histories, val_curves


# ---------------------------------------------------------------------------
# Phase 2: Evaluate on a query set
# ---------------------------------------------------------------------------

def evaluate_on_queries(agents: dict, queries: list, base_rng_seed: int) -> list:
    """Evaluate all methods on a query set. Returns per-query records."""
    records = []
    for i, q in enumerate(queries):
        opt_cost = exhaustive_min_cost(q)
        rand_cost = run_random_episode(q, rng=np.random.default_rng(base_rng_seed + i))
        greedy_r = greedy_with_peak(q)
        greedy_cost = greedy_r["total_cost"]
        greedy_peak = greedy_r["peak"]

        rl_costs, rl_gaps, rl_peaks = [], [], []
        for seed, agent in agents.items():
            env = JoinEnvironment(q, state_mode=agent.config.state_mode)
            res = agent.evaluate_episode_with_info(env)
            cost = res["total_cost"]
            rl_costs.append(cost)
            rl_gaps.append(optimality_gap(cost, opt_cost))
            rl_peaks.append(res["peak_intermediate_cardinality"])

        records.append({
            "query_idx": i,
            "n_tables": q.n_tables,
            "costs": {
                "exhaustive": opt_cost,
                "greedy": greedy_cost,
                "random": rand_cost,
                "rl_mean": float(np.mean(rl_costs)),
                **{f"rl_seed_{s}": c for s, c in zip(agents.keys(), rl_costs)},
            },
            "gaps": {
                "exhaustive": 0.0,
                "greedy": optimality_gap(greedy_cost, opt_cost),
                "random": optimality_gap(rand_cost, opt_cost),
                "rl_mean": float(np.mean(rl_gaps)),
                "rl_std": float(np.std(rl_gaps)),
                **{f"rl_seed_{s}": g for s, g in zip(agents.keys(), rl_gaps)},
            },
            "peak_cardinality": {
                "greedy": greedy_peak,
                "rl_mean": float(np.mean(rl_peaks)),
            },
        })
    return records


# ---------------------------------------------------------------------------
# Phase 3: Diagnostics (val seed 500 ONLY - never touches test seed)
# ---------------------------------------------------------------------------

def run_diagnostics(agents: dict, val_queries: list) -> dict:
    print("\n" + "=" * 72)
    print("DIAGNOSTICS - Validation Seed 500 only (test seed 888 NOT touched)")
    print("=" * 72)

    # --- 3a: Unseen (s,a) pair analysis ---
    print("\n[3a] Unseen (state, action) pair analysis")
    print("     Q=0 is OPTIMISTIC when all rewards are negative.")
    unseen_reports = {}

    for seed, agent in agents.items():
        total_steps, unseen_chosen = 0, 0
        for q in val_queries:
            env = JoinEnvironment(q, state_mode=agent.config.state_mode)
            res = agent.evaluate_episode_with_info(env)
            total_steps += res["steps"]
            unseen_chosen += res["unseen_action_count"]

        uf = unseen_chosen / max(total_steps, 1)
        unseen_reports[seed] = {"unseen_chosen": unseen_chosen,
                                "total_steps": total_steps,
                                "unseen_fraction": round(uf, 4)}
        marker = "[!]" if uf > 0.05 else "[OK]"
        print(f"  {marker} Seed {seed}: {unseen_chosen}/{total_steps} greedy steps"
              f" chose unseen (s,a) [{uf:.1%}]")

    avg_unseen_frac = np.mean([r["unseen_fraction"] for r in unseen_reports.values()])
    print(f"\n  Mean unseen-chosen fraction across seeds: {avg_unseen_frac:.1%}")
    apply_fix = avg_unseen_frac > 0.05
    if apply_fix:
        print("  -> Fraction > 5% threshold: greedy_fallback fix WILL be applied.")
    else:
        print("  -> Fraction <= 5% threshold: greedy_fallback fix applied anyway as defensive measure.")

    # --- 3b: Aliasing on 3-table val queries ---
    print("\n[3b] Action-key aliasing - 3-table validation queries")
    three_q = [q for q in val_queries if q.n_tables == 3]
    total_aliased = 0

    for seed, agent in agents.items():
        seed_alias = 0
        for q in three_q:
            env = JoinEnvironment(q, state_mode=agent.config.state_mode)
            env.reset()
            step_had_collision = False
            while not env.terminated:
                sk = agent.get_state_key(env)
                valid = env.valid_actions
                if len(valid) >= 2:
                    keys = [agent.get_action_key(env, a) for a in valid]
                    if len(set(keys)) < len(keys):
                        # Two different actions map to the same feature key
                        step_had_collision = True
                        if seed == TRAIN_SEEDS[0]:  # print details once
                            print(f"    Collision found (seed {seed}, {q.n_tables}-tbl):")
                            for a, k in zip(valid, keys):
                                q_v = agent.get_q(sk, k)
                                print(f"      action={a} key={k} Q={q_v:.4f}")
                action = agent.select_action(env, explore=False)
                env.step(action)
            if step_had_collision:
                seed_alias += 1
        total_aliased += seed_alias
        print(f"  Seed {seed}: {seed_alias}/{len(three_q)} 3-table queries had aliased action keys")

    if total_aliased == 0:
        print("  [OK] No aliasing detected - feature abstraction is injective on 3-table val queries.")
    else:
        print(f"  [!] {total_aliased} aliasing cases found - some actions share the same key.")

    # --- 3c: Paired RL vs Greedy ---
    print("\n[3c] Paired RL vs Greedy (per-query win/tie/loss, val queries)")
    opt_costs = [exhaustive_min_cost(q) for q in val_queries]
    greedy_gaps = [optimality_gap(run_greedy_episode(q), opt_costs[i])
                   for i, q in enumerate(val_queries)]

    paired_results = {}
    for seed, agent in agents.items():
        rl_gaps = []
        for i, q in enumerate(val_queries):
            env = JoinEnvironment(q, state_mode=agent.config.state_mode)
            res = agent.evaluate_episode(env)
            rl_gaps.append(optimality_gap(res["total_cost"], opt_costs[i]))

        comp = paired_comparison(rl_gaps, greedy_gaps)
        paired_results[seed] = comp
        print(f"  Seed {seed}: RL wins={comp['wins_a']} ties={comp['ties']} "
              f"losses={comp['losses_a']}  "
              f"median_diff={comp['median_diff']:.3f}%  "
              f"95%CI=[{comp['ci_diff_lo']:.3f}%, {comp['ci_diff_hi']:.3f}%]")

    report = {
        "unseen_reports": {str(k): v for k, v in unseen_reports.items()},
        "avg_unseen_fraction": float(avg_unseen_frac),
        "aliasing_cases": total_aliased,
        "apply_fix": bool(apply_fix),
        "paired_rl_vs_greedy": {str(k): v for k, v in paired_results.items()},
    }
    # Issue 5: save diagnostics so evidence isn't lost when console scrolls
    diag_path = ROOT_DIR / "results" / "diagnostics_run_a.json"
    with open(diag_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Diagnostics saved to: {diag_path}")
    return report


# ---------------------------------------------------------------------------
# Phase 4: Print results table
# ---------------------------------------------------------------------------

def print_results_table(records: list, label: str = ""):
    W = [8, 12, 7, 13, 13, 8]
    print(f"\n{'=' * 70}")
    print(f" Results: {label}")
    print(f"{'=' * 70}")
    print(fmt(["Size", "Method", "N", "Mean Cost", "Median Cost", "Mean Gap"], W))
    print("-" * 70)

    for n in TABLE_COUNTS:
        sub = [r for r in records if r["n_tables"] == n]
        for method, key in [("Exhaustive", "exhaustive"), ("Greedy", "greedy"),
                            ("Random", "random"), ("RL (mean)", "rl_mean")]:
            costs = [r["costs"][key] for r in sub]
            gaps  = [r["gaps"][key]  for r in sub]
            print(fmt([
                f"{n} tbl" if method == "Exhaustive" else "",
                method, len(sub),
                f"{np.mean(costs):>10,.0f}", f"{np.median(costs):>10,.0f}",
                f"{np.mean(gaps):.2f}%"
            ], W))
        print("-" * 70)

    # Headline (4-6 tables)
    headline = [r for r in records if r["n_tables"] >= 4]
    rl_seed_gaps_headline = []
    for seed in TRAIN_SEEDS:
        g = [r["gaps"][f"rl_seed_{seed}"] for r in headline
             if f"rl_seed_{seed}" in r["gaps"]]
        if g:
            rl_seed_gaps_headline.append(float(np.mean(g)))

    if rl_seed_gaps_headline:
        m, s, ci = compute_ci(rl_seed_gaps_headline)
        print(f"\n  HEADLINE 4-6 tables:")
        print(f"    RL   : {m:.2f}% +/- {s:.2f}%  95%CI [{m-ci:.2f}%, {m+ci:.2f}%]")
    greedy_h = [r["gaps"]["greedy"] for r in headline]
    print(f"    Greedy: {np.mean(greedy_h):.2f}%  (deterministic)")


# ---------------------------------------------------------------------------
# Phase 5: Figures
# ---------------------------------------------------------------------------

def save_figures(val_curves: dict, records: list, label: str = ""):
    figures_dir = ROOT_DIR / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Val-gap curve (replaces training-return curve from Day 6)
    plt.figure(figsize=(10, 5.5), dpi=150)
    ax = plt.gca()
    all_curves = []
    for seed, curve in val_curves.items():
        eps = [p[0] for p in curve]
        gaps = [p[1] for p in curve]
        ax.plot(eps, gaps, alpha=0.4, linewidth=1.2, label=f"Seed {seed}")
        all_curves.append(np.interp(
            np.array([p[0] for p in list(val_curves.values())[0]]),
            eps, gaps))

    mean_curve = np.mean(all_curves, axis=0)
    std_curve  = np.std(all_curves, axis=0)
    base_eps   = [p[0] for p in list(val_curves.values())[0]]
    ax.plot(base_eps, mean_curve, color="darkblue", linewidth=2.5, label="Mean (5 seeds)")
    ax.fill_between(base_eps, mean_curve - std_curve, mean_curve + std_curve,
                    color="royalblue", alpha=0.2, label="+/-1 std")

    ax.set_xlabel("Training Episode", fontweight="bold")
    ax.set_ylabel("Val-Set Optimality Gap (%) vs Exhaustive", fontweight="bold")
    ax.set_title(f"Greedy-Policy Val Gap During Training - {label}", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    suffix = "_fallback" if "fallback" in label.lower() else "_baseline"
    fig1_path = figures_dir / f"val_gap_curve{suffix}.png"
    plt.tight_layout()
    plt.savefig(fig1_path)
    plt.close()
    print(f"  Saved: {fig1_path}")

    # Figure 2: Optimality gap by table count
    plt.figure(figsize=(9, 5.5), dpi=150)
    categories = ["3 tbl", "4 tbl", "5 tbl", "6 tbl", "4-6 (headline)"]
    x = np.arange(len(categories))
    w = 0.22

    def method_mean_by_count(method_key):
        vals = []
        for n in TABLE_COUNTS:
            sub = [r["gaps"][method_key] for r in records if r["n_tables"] == n]
            vals.append(float(np.mean(sub)) if sub else 0.0)
        sub_hl = [r["gaps"][method_key] for r in records if r["n_tables"] >= 4]
        vals.append(float(np.mean(sub_hl)) if sub_hl else 0.0)
        return vals

    rand_v   = method_mean_by_count("random")
    greedy_v = method_mean_by_count("greedy")
    rl_v, rl_e = [], []
    for n_idx, n in enumerate(TABLE_COUNTS):
        seed_g = [np.mean([r["gaps"][f"rl_seed_{s}"] for r in records
                           if r["n_tables"] == n and f"rl_seed_{s}" in r["gaps"]])
                  for s in TRAIN_SEEDS]
        rl_v.append(float(np.mean(seed_g)))
        rl_e.append(float(np.std(seed_g)))
    seed_hl = [np.mean([r["gaps"][f"rl_seed_{s}"] for r in records
                        if r["n_tables"] >= 4 and f"rl_seed_{s}" in r["gaps"]])
               for s in TRAIN_SEEDS]
    rl_v.append(float(np.mean(seed_hl)))
    rl_e.append(float(np.std(seed_hl)))

    plt.bar(x - w, rand_v,   w, label="Random",           color="#e74c3c", alpha=0.85)
    plt.bar(x,     greedy_v, w, label="Greedy",           color="#2ecc71", alpha=0.85)
    plt.bar(x + w, rl_v,     w, yerr=rl_e, capsize=4,
            label="RL (5 seeds +/- std)", color="#3498db", alpha=0.85)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Query Complexity", fontweight="bold")
    plt.ylabel("Mean Optimality Gap (%) vs Exhaustive", fontweight="bold")
    plt.title(f"Optimality Gap by Table Count - {label}", fontweight="bold")
    plt.xticks(x, categories)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(framealpha=0.9)

    fig2_path = figures_dir / f"optimality_gap{suffix}.png"
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.close()
    print(f"  Saved: {fig2_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_main_experiments():
    print("=" * 72)
    print(" QueryPilot - Day 6-8 Full Experiment Pipeline")
    print("=" * 72)
    print(f"Train seeds: {TRAIN_SEEDS}  |  Val seed: {VAL_SEED}  |  Test seed: {FINAL_TEST_SEED}")
    print(f"Episodes: {EPISODES}/seed, linear decay over {DECAY_EPISODES} eps")
    print(f"Val-gap snapshots every {VAL_EVAL_EVERY} episodes")

    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----- Generate query sets -----
    print("\n>>> Generating query sets...")
    val_queries  = make_query_set(VAL_SEED)
    test_queries = make_query_set(FINAL_TEST_SEED)
    print(f"  Val  queries: {len(val_queries)}  (seed {VAL_SEED})")
    print(f"  Test queries: {len(test_queries)} (seed {FINAL_TEST_SEED})")

    # =====================================================================
    # RUN A: Baseline (no greedy_fallback)
    # =====================================================================
    print("\n" + "=" * 72)
    print(" RUN A: Baseline (no greedy_fallback fix)")
    print("=" * 72)

    agents_a, histories_a, val_curves_a = train_agents(val_queries, use_fallback=False)

    print("\n>>> Phase 2A: Diagnostics on val set (seed 500)...")
    diag = run_diagnostics(agents_a, val_queries)

    print("\n>>> Phase 3A: Evaluating baseline agents on val set...")
    val_records_a = evaluate_on_queries(agents_a, val_queries, VAL_SEED)
    print_results_table(val_records_a, label="Baseline - Val Set (seed 500)")

    print("\n>>> Phase 4A: Figures for baseline...")
    save_figures(val_curves_a, val_records_a, label="Baseline")

    # =====================================================================
    # RUN B: With greedy_fallback fix (always applied defensively)
    # =====================================================================
    print("\n" + "=" * 72)
    print(" RUN B: With greedy_fallback fix")
    print(f"  Diagnostic unseen-chosen rate was "
          f"{diag['avg_unseen_fraction']:.1%} - fix applied {'(triggered)' if diag['apply_fix'] else '(defensive)'}.")
    print("=" * 72)

    agents_b, histories_b, val_curves_b = train_agents(val_queries, use_fallback=True)

    print("\n>>> Phase 2B: Evaluating fixed agents on val set...")
    val_records_b = evaluate_on_queries(agents_b, val_queries, VAL_SEED)
    print_results_table(val_records_b, label="Q-learning+Greedy Fallback (Hybrid) - Val Set (seed 500)")

    # Compare honestly. delta < 0 means Run B improved.
    # But gains may reflect greedy heuristic for unseen states, not better Q-values.
    mean_gap_a = np.mean([r["gaps"]["rl_mean"] for r in val_records_a])
    mean_gap_b = np.mean([r["gaps"]["rl_mean"] for r in val_records_b])
    delta = mean_gap_b - mean_gap_a
    if delta < 0:
        print(f"\n  Val gap change: {delta:.3f}% (Run B improved).")
        print("  [!]  Note: improvement may partly come from greedy fallback on unseen states,")
        print("     not from better Q-values. Run A is the pure RL result.")
    else:
        print(f"\n  Val gap change: {delta:+.3f}% (Run B no better or worse - reported honestly).")

    save_figures(val_curves_b, val_records_b, label="Fallback Fix")

    # =====================================================================
    # FINAL EVALUATION: Use best run on fresh test seed (888), ONE TIME
    # =====================================================================
    # Pick whichever run had lower mean val gap ON VALIDATION only.
    # Test seed 888 is not printed or evaluated until this decision is made.
    if mean_gap_b <= mean_gap_a:
        final_agents = agents_b
        final_label = "Q-learning + Greedy Fallback (Hybrid)"
    else:
        final_agents = agents_a
        final_label = "Pure RL (Baseline)"

    print("\n" + "=" * 72)
    print(f" FINAL EVALUATION - Test Seed {FINAL_TEST_SEED} (used once, never for decisions)")
    print(f" Using: {final_label} agents (better on val)")
    print("=" * 72)

    test_records = evaluate_on_queries(final_agents, test_queries, FINAL_TEST_SEED)
    print_results_table(test_records, label=f"{final_label} - Test Set (seed {FINAL_TEST_SEED})")

    # Peak cardinality summary
    print("\n>>> Peak intermediate cardinality (test set):")
    rl_peaks  = [r["peak_cardinality"]["rl_mean"]  for r in test_records]
    gr_peaks  = [r["peak_cardinality"]["greedy"]   for r in test_records]
    print(f"  RL   mean peak rows : {np.mean(rl_peaks):>12,.0f}")
    print(f"  Greedy mean peak    : {np.mean(gr_peaks):>12,.0f}")

    # Paired comparison on test set
    print("\n>>> Paired RL vs Greedy (test set):")
    test_opt = [exhaustive_min_cost(q) for q in test_queries]
    for seed, agent in final_agents.items():
        rl_gaps_test, gr_gaps_test = [], []
        for i, q in enumerate(test_queries):
            env = JoinEnvironment(q, state_mode=agent.config.state_mode)
            res = agent.evaluate_episode(env)
            rl_gaps_test.append(optimality_gap(res["total_cost"], test_opt[i]))
            gr_gaps_test.append(optimality_gap(run_greedy_episode(q), test_opt[i]))
        comp = paired_comparison(rl_gaps_test, gr_gaps_test)
        print(f"  Seed {seed}: wins={comp['wins_a']} ties={comp['ties']} "
              f"losses={comp['losses_a']}  "
              f"median_diff={comp['median_diff']:.3f}%  "
              f"95%CI=[{comp['ci_diff_lo']:.3f}%, {comp['ci_diff_hi']:.3f}%]")

    # ----- Save JSON summary -----
    summary = {
        "setup": {
            "train_seeds": TRAIN_SEEDS, "val_seed": VAL_SEED,
            "final_test_seed": FINAL_TEST_SEED,
            "episodes": EPISODES, "decay_episodes": DECAY_EPISODES,
            "alpha": ALPHA, "gamma": GAMMA, "state_mode": STATE_MODE,
        },
        "diagnostics": diag,
        "run_a_mean_val_gap": float(mean_gap_a),
        "run_b_mean_val_gap": float(mean_gap_b),
        "fix_delta_val": float(delta),
        "fix_helped": bool(delta < 0),
        "final_label": final_label,
        "test_records": test_records,
    }

    out = results_dir / "main_experiment_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n>>> Summary saved to: {out}")
    print("=" * 72)
    print("All experiments complete.")


if __name__ == "__main__":
    run_main_experiments()
