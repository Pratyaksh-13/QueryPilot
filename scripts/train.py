"""
scripts/train.py
----------------
Command-line training script for the QueryPilot Q-learning agent.

Usage:
    python scripts/train.py
    python scripts/train.py --episodes 1000 --seed 42 --save-path results/q_table.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import yaml
import numpy as np

from src.environment.query_generator import generate_query
from src.environment.join_environment import JoinEnvironment
from src.agents.q_learning import QLearningAgent, QLearningConfig


def load_config(config_path: Path) -> dict:
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def train(
    episodes: int,
    seed: int,
    config_path: Path,
    save_path: Path,
    log_path: Path,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_min: float,
    epsilon_decay: float,
    state_mode: str,
    log_every: int = 100,
):
    print("=" * 70)
    print(" QueryPilot RL Agent Training ")
    print("=" * 70)
    print(f"Episodes      : {episodes}")
    print(f"Training Seed : {seed}")
    print(f"State Mode    : {state_mode}")
    print(f"Hyperparams   : alpha={alpha}, gamma={gamma}, eps_start={epsilon_start}, eps_min={epsilon_min}, decay={epsilon_decay}")
    print(f"Save Path     : {save_path}")
    print(f"Log Path      : {log_path}")
    print("-" * 70)

    cfg = load_config(config_path)
    env_cfg = cfg.get("environment", {})
    min_tables = env_cfg.get("min_tables", 3)
    max_tables = env_cfg.get("max_tables", 6)
    min_card = env_cfg.get("min_cardinality", 100)
    max_card = env_cfg.get("max_cardinality", 10_000)
    min_sel = env_cfg.get("min_selectivity", 0.01)
    max_sel = env_cfg.get("max_selectivity", 0.5)
    cost_scale = env_cfg.get("cost_scale", 1_000_000)

    agent_config = QLearningConfig(
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        state_mode=state_mode,
        seed=seed,
    )
    agent = QLearningAgent(agent_config)

    # Master PRNG for training queries
    train_rng = np.random.default_rng(seed)

    history = []
    t_start = time.perf_counter()

    for ep in range(1, episodes + 1):
        # Generate query on-the-fly using training PRNG
        q = generate_query(
            rng=train_rng,
            min_tables=min_tables,
            max_tables=max_tables,
            min_cardinality=min_card,
            max_cardinality=max_card,
            min_selectivity=min_sel,
            max_selectivity=max_sel,
        )

        env = JoinEnvironment(q, cost_scale=cost_scale, state_mode=state_mode)
        metrics = agent.train_episode(env)
        history.append({
            "episode": ep,
            "return": metrics["return"],
            "total_cost": metrics["total_cost"],
            "steps": metrics["steps"],
            "epsilon": metrics["epsilon"],
            "mean_td_error": metrics["mean_td_error"],
            "q_table_size": len(agent.q_table),
        })

        if ep % log_every == 0 or ep == episodes:
            recent = history[-log_every:]
            avg_ret = float(np.mean([m["return"] for m in recent]))
            avg_cost = float(np.mean([m["total_cost"] for m in recent]))
            print(
                f"Ep {ep:5d}/{episodes} | "
                f"Avg Cost: {avg_cost:11,.0f} | "
                f"Avg Ret: {avg_ret:8.3f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Q-entries: {len(agent.q_table):5d}"
            )

    elapsed = time.perf_counter() - t_start
    print("-" * 70)
    print(f"Training completed in {elapsed:.2f}s ({episodes / elapsed:.1f} ep/s)")
    print(f"Final Q-table size: {len(agent.q_table)} state-action entries")

    # Save agent checkpoint
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)
    print(f"Saved trained agent checkpoint to: {save_path}")

    # Save training log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "episodes": episodes,
            "seed": seed,
            "state_mode": state_mode,
            "hyperparameters": {
                "alpha": alpha,
                "gamma": gamma,
                "epsilon_start": epsilon_start,
                "epsilon_min": epsilon_min,
                "epsilon_decay": epsilon_decay,
            },
            "elapsed_sec": elapsed,
            "history": history,
        }, f, indent=2)
    print(f"Saved training log to: {log_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train QueryPilot Q-learning agent.")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.998, help="Epsilon decay rate")
    parser.add_argument("--state-mode", type=str, default="feature", choices=["feature", "raw"], help="State key mode")
    parser.add_argument("--save-path", type=str, default="results/q_table.json", help="Path to save Q-table")
    parser.add_argument("--log-path", type=str, default="results/training_log.json", help="Path to save training log")
    parser.add_argument("--log-every", type=int, default=200, help="Logging interval in episodes")

    args = parser.parse_args()

    train(
        episodes=args.episodes,
        seed=args.seed,
        config_path=Path(args.config),
        save_path=Path(args.save_path),
        log_path=Path(args.log_path),
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        state_mode=args.state_mode,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
