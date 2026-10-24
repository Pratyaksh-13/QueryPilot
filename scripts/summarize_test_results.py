"""
summarize_test_results.py
-------------------------
Recompute the headline test-set numbers from the per-query records saved in
results/main_experiment_summary.json (seed 888, Run A). Nothing is retrained or
re-evaluated, so every number in the report can be traced to
results/final_test_summary.json.

Usage:  python scripts/summarize_test_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.metrics import bootstrap_ci  # noqa: E402

TRAIN_SEEDS = [101, 102, 103, 104, 105]
TOL = 1e-6


def per_seed_mean(records, seed):
    return float(np.mean([r["gaps"][f"rl_seed_{seed}"] for r in records]))


def summarize_subset(records):
    seed_means = [per_seed_mean(records, s) for s in TRAIN_SEEDS]
    return {
        "n": len(records),
        "random_mean_gap": float(np.mean([r["gaps"]["random"] for r in records])),
        "greedy_mean_gap": float(np.mean([r["gaps"]["greedy"] for r in records])),
        "rl_mean_of_seed_means": float(np.mean(seed_means)),
        "rl_seed_means": seed_means,
        "rl_across_seed_std": float(np.std(seed_means)),
        "rl_across_seed_std_ddof1": float(np.std(seed_means, ddof=1)),
    }


def save_test_figure(out):
    """Bar chart of mean optimality gap by table count on the test set (seed 888)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["3", "4", "5", "6", "4-6"]
    subsets = [out["by_size"][k] for k in ("3", "4", "5", "6")] + [out["headline_4_to_6"]]
    rand = [x["random_mean_gap"] for x in subsets]
    greedy = [x["greedy_mean_gap"] for x in subsets]
    rl = [x["rl_mean_of_seed_means"] for x in subsets]
    rl_sd = [x["rl_across_seed_std_ddof1"] for x in subsets]

    xs = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.bar(xs - w, rand, w, label="Random", color="#e74c3c", alpha=0.85)
    ax.bar(xs, greedy, w, label="Greedy", color="#2ecc71", alpha=0.85)
    ax.bar(xs + w, rl, w, yerr=rl_sd, capsize=3, label="RL (mean of 5 seeds, +/- sd)",
           color="#3498db", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x} tables" if x != "4-6" else "4-6 (headline)" for x in labels])
    ax.set_ylabel("Mean optimality gap (%)")
    ax.set_title(f"Test set (seed {out['test_seed']}), pure RL")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_dir = ROOT_DIR / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "test_gap_by_size.png")
    plt.close(fig)


def main():
    path = ROOT_DIR / "results" / "main_experiment_summary.json"
    summary = json.load(open(path))
    records = summary["test_records"]
    out = {"source": str(path.name), "final_label": summary.get("final_label"),
           "test_seed": summary["setup"]["final_test_seed"], "by_size": {}}

    for n in (3, 4, 5, 6):
        out["by_size"][str(n)] = summarize_subset([r for r in records if r["n_tables"] == n])

    head = [r for r in records if r["n_tables"] >= 4]
    out["headline_4_to_6"] = summarize_subset(head)

    # Paired RL (mean over seeds) minus Greedy, per query, 4-6 tables
    diffs = np.array([r["gaps"]["rl_mean"] - r["gaps"]["greedy"] for r in head])
    m, mlo, mhi = bootstrap_ci(diffs, stat_fn=np.mean, n_boot=5000,
                               rng=np.random.default_rng(0))
    md, dlo, dhi = bootstrap_ci(diffs, stat_fn=np.median, n_boot=5000,
                                rng=np.random.default_rng(0))
    out["paired_rl_minus_greedy_4_to_6"] = {
        "definition": "per-query (RL gap averaged over 5 seeds) minus Greedy gap, percentage points",
        "n": len(diffs),
        "mean": m, "mean_ci95": [mlo, mhi],
        "median": md, "median_ci95": [dlo, dhi],
        "rl_better": int(np.sum(diffs < -TOL)),
        "tied": int(np.sum(np.abs(diffs) <= TOL)),
        "rl_worse": int(np.sum(diffs > TOL)),
    }

    # Fraction of the Random -> Greedy distance closed by RL
    h = out["headline_4_to_6"]
    out["fraction_random_to_greedy_closed"] = (
        (h["random_mean_gap"] - h["rl_mean_of_seed_means"])
        / (h["random_mean_gap"] - h["greedy_mean_gap"]))

    # Heavy tail: worst RL queries (4-6 tables)
    worst = sorted(head, key=lambda r: r["gaps"]["rl_mean"], reverse=True)[:5]
    out["worst_rl_queries"] = [
        {"query_idx": r["query_idx"], "n_tables": r["n_tables"],
         "rl_mean_gap": r["gaps"]["rl_mean"], "greedy_gap": r["gaps"]["greedy"]}
        for r in worst]

    with open(ROOT_DIR / "results" / "final_test_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    save_test_figure(out)

    print("RL sd = sample std (ddof=1) across the 5 training seeds")
    print(f"Test seed {out['test_seed']} ({out['final_label']})")
    print(f"{'size':>5} {'n':>3} {'Random':>8} {'Greedy':>8} {'RL':>8} {'RL sd':>7}")
    for n, s in out["by_size"].items():
        print(f"{n:>5} {s['n']:>3} {s['random_mean_gap']:8.2f} {s['greedy_mean_gap']:8.2f} "
              f"{s['rl_mean_of_seed_means']:8.2f} {s['rl_across_seed_std_ddof1']:7.2f}")
    print(f"{'4-6':>5} {h['n']:>3} {h['random_mean_gap']:8.2f} {h['greedy_mean_gap']:8.2f} "
          f"{h['rl_mean_of_seed_means']:8.2f} {h['rl_across_seed_std_ddof1']:7.2f}")
    p = out["paired_rl_minus_greedy_4_to_6"]
    print(f"\nPaired RL - Greedy (4-6 tables, n={p['n']}): mean {p['mean']:.2f} "
          f"CI [{p['mean_ci95'][0]:.2f}, {p['mean_ci95'][1]:.2f}] | median {p['median']:.2f} "
          f"CI [{p['median_ci95'][0]:.2f}, {p['median_ci95'][1]:.2f}]")
    print(f"RL better {p['rl_better']} / tied {p['tied']} / worse {p['rl_worse']}")
    print(f"Random-to-Greedy distance closed by RL: {out['fraction_random_to_greedy_closed']:.1%}")
    print("Worst RL queries:", [(w['query_idx'], round(w['rl_mean_gap'], 1)) for w in out["worst_rl_queries"]])


if __name__ == "__main__":
    main()
