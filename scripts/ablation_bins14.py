"""
ablation_bins14.py
------------------
Single ablation: replace the 7 decade row-count bins with 14 half-decade bins,
train exactly like Run A (5 seeds, 5000 eps, linear decay), and compare on
VALIDATION seed 500 only. Test seed 888 is never touched here.

The bin change is applied by overriding join_environment._ROW_BINS at runtime,
so core source files are unchanged. Run A numbers are recomputed from the
committed checkpoints (unpatched bins) as the comparison baseline.

Usage:  python scripts/ablation_bins14.py [--episodes 5000]
Output: results/ablation_bins14.json
        results/checkpoints/agent_bins14_seed_*.json   (gitignored)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import src.environment.join_environment as je
from scripts.run_main_experiments import (
    make_query_set, eval_val_gap, TRAIN_SEEDS, VAL_SEED, EPISODES, DECAY_EPISODES,
    EPSILON_START, EPSILON_MIN, ALPHA, GAMMA, COST_SCALE, STATE_MODE,
)
from src.agents.q_learning import QLearningAgent, QLearningConfig
from src.environment.query_generator import generate_query

BINS_7 = list(je._ROW_BINS)  # 10 .. 1e6
BINS_14 = [10, 32, 100, 316, 1_000, 3_162, 10_000, 31_623,
           100_000, 316_228, 1_000_000, 3_162_278, 10_000_000]  # 13 thresholds -> 14 bins


def train_one(seed: int, episodes: int) -> QLearningAgent:
    cfg = QLearningConfig(alpha=ALPHA, gamma=GAMMA, epsilon_start=EPSILON_START,
                          epsilon_min=EPSILON_MIN, decay_schedule="linear",
                          decay_episodes=DECAY_EPISODES, state_mode=STATE_MODE, seed=seed)
    agent = QLearningAgent(cfg)
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        n_t = int(rng.integers(3, 7))
        q = generate_query(rng=rng, min_tables=n_t, max_tables=n_t,
                           min_cardinality=100, max_cardinality=10_000,
                           min_selectivity=0.01, max_selectivity=0.5)
        agent.train_episode(je.JoinEnvironment(q, cost_scale=COST_SCALE, state_mode=STATE_MODE))
    return agent


def summarize(gaps):
    return {"per_seed": gaps, "mean": float(np.mean(gaps)), "std": float(np.std(gaps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=EPISODES)
    args = ap.parse_args()
    val = make_query_set(VAL_SEED)

    # Run A baseline from committed checkpoints (7 bins)
    je._ROW_BINS = BINS_7
    run_a = [eval_val_gap(QLearningAgent.load(ROOT_DIR / "results" / "checkpoints" / f"agent_seed_{s}.json"), val)
             for s in TRAIN_SEEDS]
    print("Run A (7 bins) val gaps:", [round(g, 2) for g in run_a],
          f"mean {np.mean(run_a):.2f} std {np.std(run_a):.2f}", flush=True)

    # Ablation (14 bins)
    je._ROW_BINS = BINS_14
    abl = []
    t0 = time.perf_counter()
    for s in TRAIN_SEEDS:
        agent = train_one(s, args.episodes)
        agent.save(ROOT_DIR / "results" / "checkpoints" / f"agent_bins14_seed_{s}.json")
        g = eval_val_gap(agent, val)
        abl.append(g)
        print(f"  bins14 seed {s}: val gap {g:.2f}%  Q-entries {len(agent.q_table):,}  "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"Bins14 val gaps: mean {np.mean(abl):.2f} std {np.std(abl):.2f}")

    out = {"val_seed": VAL_SEED, "episodes": args.episodes, "bins14_thresholds": BINS_14,
           "run_a_7bins": summarize(run_a), "bins14": summarize(abl),
           "note": "Validation only. Seed 888 not evaluated."}
    with open(ROOT_DIR / "results" / "ablation_bins14.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()