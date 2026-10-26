"""
scripts/eval_checkpoints.py
---------------------------
Evaluate the SHIPPED trained agents (results/checkpoints/agent_seed_*.json)
on the held-out test set (seed 888, 25 queries per size for 3-6 tables),
without retraining.

Ties between equal Q-values are broken with the agent's RNG, so one evaluation
is one random draw. This script repeats the evaluation with several different
tie-break seeds and reports the mean and spread, so the number does not depend
on a lucky or unlucky draw.

Usage:  python scripts/eval_checkpoints.py
        python scripts/eval_checkpoints.py --tiebreak-seeds 20
Output: printed table + results/checkpoint_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.agents.q_learning import QLearningAgent  # noqa: E402
from src.baselines.exhaustive import exhaustive_min_cost  # noqa: E402
from src.baselines.greedy_agent import run_greedy_episode  # noqa: E402
from src.baselines.random_agent import run_random_episode  # noqa: E402
from src.environment.join_environment import JoinEnvironment  # noqa: E402
from src.environment.query_generator import generate_query  # noqa: E402
from src.evaluation.metrics import optimality_gap  # noqa: E402

TRAIN_SEEDS = [101, 102, 103, 104, 105]
TABLE_COUNTS = [3, 4, 5, 6]


def make_query_set(seed: int, per_size: int = 25) -> list:
    """Same generator as scripts/run_main_experiments.py (identical queries)."""
    rng = np.random.default_rng(seed)
    return [
        generate_query(rng=rng, min_tables=n, max_tables=n,
                       min_cardinality=100, max_cardinality=10_000,
                       min_selectivity=0.01, max_selectivity=0.5)
        for n in TABLE_COUNTS for _ in range(per_size)
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate shipped checkpoints on the test set.")
    p.add_argument("--checkpoint-dir", default="results/checkpoints")
    p.add_argument("--test-seed", type=int, default=888)
    p.add_argument("--tiebreak-seeds", type=int, default=10,
                   help="number of different tie-break RNG seeds to average over")
    args = p.parse_args()

    queries = make_query_set(args.test_seed)
    opt = [exhaustive_min_cost(q) for q in queries]
    rand_gap = [optimality_gap(run_random_episode(q, np.random.default_rng(args.test_seed + i)), o)
                for i, (q, o) in enumerate(zip(queries, opt))]
    greedy_gap = [optimality_gap(run_greedy_episode(q), o) for q, o in zip(queries, opt)]
    sizes = np.array([q.n_tables for q in queries])

    ckpt_dir = ROOT_DIR / args.checkpoint_dir
    # rl_gaps[t][s] = per-query gap array for tie-break seed t, training seed s
    rl_gaps = []
    for t in range(args.tiebreak_seeds):
        per_seed = []
        for s in TRAIN_SEEDS:
            agent = QLearningAgent.load(ckpt_dir / f"agent_seed_{s}.json")
            agent.rng = np.random.default_rng(10_000 + t)  # fixed, reproducible tie-breaks
            gaps = [optimality_gap(
                        agent.evaluate_episode(JoinEnvironment(q, state_mode=agent.config.state_mode))["total_cost"], o)
                    for q, o in zip(queries, opt)]
            per_seed.append(gaps)
        rl_gaps.append(np.array(per_seed))          # (5 seeds, n_queries)
    rl = np.array(rl_gaps)                          # (T, 5, n_queries)

    def row(mask):
        # mean over queries -> (T, 5); then mean over training seeds -> (T,)
        per_t = rl[:, :, mask].mean(axis=2).mean(axis=1)
        return (float(np.mean(np.array(rand_gap)[mask])),
                float(np.mean(np.array(greedy_gap)[mask])),
                float(per_t.mean()), float(per_t.std()), float(per_t.min()), float(per_t.max()))

    out = {"test_seed": args.test_seed, "tiebreak_seeds": args.tiebreak_seeds, "by_size": {}}
    print(f"Shipped checkpoints, test seed {args.test_seed}, "
          f"{args.tiebreak_seeds} tie-break seeds (mean gap to optimal, %)")
    print(f"{'tables':>7} {'Random':>8} {'Greedy':>8} {'RL mean':>8} {'RL sd':>7} {'RL min':>7} {'RL max':>7}")
    for label, mask in [(str(n), sizes == n) for n in TABLE_COUNTS] + [("4-6", sizes >= 4)]:
        r = row(mask)
        out["by_size"][label] = dict(zip(
            ["random", "greedy", "rl_mean", "rl_sd_over_tiebreaks", "rl_min", "rl_max"], r))
        print(f"{label:>7} " + " ".join(f"{v:8.2f}" for v in r[:2]) + " " + " ".join(f"{v:7.2f}" for v in r[2:]))

    out_path = ROOT_DIR / "results" / "checkpoint_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()