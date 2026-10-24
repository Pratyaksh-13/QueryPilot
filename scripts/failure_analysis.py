"""
failure_analysis.py
-------------------
Diagnostic replay of queries where RL loses badly to greedy, plus cost-model
checks for the cardinality clamp.

This script is DIAGNOSTIC ONLY. It reads the seed-888 test set to understand
failures; nothing found here may be used to tune hyperparameters. Any change
adopted afterwards must be evaluated on a fresh seed (e.g. 777).

Usage:
    python scripts/failure_analysis.py                  # queries 40 88 91, seed 101
    python scripts/failure_analysis.py --queries 40 88 --agent-seed 102

Outputs:
    results/failure_analysis.json
    results/clamp_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.run_main_experiments import make_query_set, FINAL_TEST_SEED, TRAIN_SEEDS  # noqa: E402
from src.agents.q_learning import QLearningAgent  # noqa: E402
from src.baselines.exhaustive import exhaustive_min_cost  # noqa: E402
from src.environment.cardinality import MAX_CARD  # noqa: E402
from src.environment.join_environment import JoinEnvironment  # noqa: E402
from src.evaluation.metrics import optimality_gap  # noqa: E402
from src.baselines.greedy_agent import _greedy_action  # noqa: E402


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def comp_repr(c):
    return {"tables": sorted(c.tables), "rows": c.rows}


def replay_greedy(query):
    """Greedy plan with per-step info (matches run_greedy_episode exactly)."""
    env = JoinEnvironment(query)
    env.reset()
    steps = []
    while not env.terminated:
        comps = env.components
        action = _greedy_action(env)
        i, j = action
        left, right = comps[i], comps[j]
        _, _, _, _, info = env.step(action)
        steps.append({
            "joined": [sorted(left.tables), sorted(right.tables)],
            "rows_in": [left.rows, right.rows],
            "rows_out": info["rows_result"],
            "step_cost": info["join_cost"],
        })
    return {"total_cost": env.total_cost, "steps": steps}


def replay_rl(agent, query):
    """RL greedy-policy plan with full per-step Q-table detail."""
    env = JoinEnvironment(query, state_mode=agent.config.state_mode)
    env.reset()
    steps = []
    n_unseen = 0
    while not env.terminated:
        comps = env.components
        state_key = agent.get_state_key(env)
        options = []
        for a in env.valid_actions:
            ak = agent.get_action_key(env, a)
            options.append({
                "action": list(a),
                "joins": [sorted(comps[a[0]].tables), sorted(comps[a[1]].tables)],
                "action_key": ak,
                "q": agent.get_q(state_key, ak),
                "seen": (state_key, ak) in agent.seen_pairs,
            })
        chosen = agent.select_action(env, explore=False)
        ck = agent.get_action_key(env, chosen)
        chosen_seen = (state_key, ck) in agent.seen_pairs
        n_unseen += (not chosen_seen)
        left, right = comps[chosen[0]], comps[chosen[1]]
        _, _, _, _, info = env.step(chosen)
        steps.append({
            "state_key": state_key,
            "components": [comp_repr(c) for c in comps],
            "options": options,
            "chosen": list(chosen),
            "chosen_joins": [sorted(left.tables), sorted(right.tables)],
            "chosen_seen": chosen_seen,
            "rows_out": info["rows_result"],
            "step_cost": info["join_cost"],
        })
    return {"total_cost": env.total_cost, "n_unseen_chosen": n_unseen, "steps": steps}


def unclamped_peak_for_plan(query, plan_joins):
    """
    Re-simulate a plan with NO clamp and NO int truncation.
    plan_joins: list of (tables_left, tables_right) frozensets, in order.
    Returns (peak_rows, final_rows, unclamped_total_cost).
    """
    rows = {frozenset({i}): float(query.cardinalities[i]) for i in range(query.n_tables)}
    peak, total = 0.0, 0.0
    final_rows = 0.0
    for left, right in plan_joins:
        left, right = frozenset(left), frozenset(right)
        sel = 1.0
        for u in left:
            for v in right:
                key = (min(u, v), max(u, v))
                if key in query.edges:
                    sel *= query.selectivities[key]
        out = rows[left] * rows[right] * sel
        total += rows[left] + rows[right] + out
        peak = max(peak, out)
        final_rows = out
        rows[left | right] = out
    return peak, final_rows, total


def run_plan_from_agent(agent, query):
    env = JoinEnvironment(query, state_mode=agent.config.state_mode)
    env.reset()
    joins = []
    while not env.terminated:
        comps = env.components
        a = agent.select_action(env, explore=False)
        joins.append((comps[a[0]].tables, comps[a[1]].tables))
        env.step(a)
    return joins


def run_plan_from_greedy(query):
    env = JoinEnvironment(query)
    env.reset()
    joins = []
    while not env.terminated:
        comps = env.components
        a = _greedy_action(env)
        joins.append((comps[a[0]].tables, comps[a[1]].tables))
        env.step(a)
    return joins


def clamp_analysis(queries, agent):
    """Fraction of intermediate / final cardinalities at the MAX_CARD clamp."""
    out = {}
    for name, planner in (("greedy", run_plan_from_greedy),
                          ("rl", lambda q: run_plan_from_agent(agent, q))):
        inter_total = inter_clamped = final_total = final_clamped = 0
        peaks_unclamped, final_unclamped = [], []
        for q in queries:
            joins = planner(q)
            env = JoinEnvironment(q)
            env.reset()
            # replay clamped plan to read clamped cardinalities
            for left_t, right_t in joins:
                comps = env.components
                li = next(i for i, c in enumerate(comps) if c.tables == frozenset(left_t))
                ri = next(i for i, c in enumerate(comps) if c.tables == frozenset(right_t))
                a = (min(li, ri), max(li, ri))
                _, _, term, _, info = env.step(a)
                is_final = term
                hit = info["rows_result"] >= MAX_CARD
                if is_final:
                    final_total += 1
                    final_clamped += hit
                else:
                    inter_total += 1
                    inter_clamped += hit
            peak, fin, _ = unclamped_peak_for_plan(q, joins)
            peaks_unclamped.append(peak)
            final_unclamped.append(fin)
        out[name] = {
            "n_queries": len(queries),
            "intermediate_total": inter_total,
            "intermediate_at_clamp": int(inter_clamped),
            "intermediate_clamp_fraction": inter_clamped / max(1, inter_total),
            "final_total": final_total,
            "final_at_clamp": int(final_clamped),
            "final_clamp_fraction": final_clamped / max(1, final_total),
            "median_unclamped_peak": float(np.median(peaks_unclamped)),
            "max_unclamped_peak": float(np.max(peaks_unclamped)),
            "median_unclamped_final": float(np.median(final_unclamped)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, nargs="+", default=[40, 88, 91])
    ap.add_argument("--agent-seed", type=int, default=101)
    ap.add_argument("--set-seed", type=int, default=FINAL_TEST_SEED)
    args = ap.parse_args()

    queries = make_query_set(args.set_seed)
    agents = {s: QLearningAgent.load(ROOT_DIR / "results" / "checkpoints" / f"agent_seed_{s}.json")
              for s in TRAIN_SEEDS}
    agent = agents[args.agent_seed]

    # ---- Sanity check: do the indices match the reported failures? ----
    print(f"Query set: seed {args.set_seed}, {len(queries)} queries\n")
    print("SANITY CHECK (expected from summary: greedy gap 0, RL mean gap ~150 / ~100 / ~48 for 40 / 88 / 91)")
    sanity = {}
    for qi in args.queries:
        q = queries[qi]
        opt = exhaustive_min_cost(q)
        g = replay_greedy(q)["total_cost"]
        rl_gaps = {s: optimality_gap(replay_rl(a, q)["total_cost"], opt) for s, a in agents.items()}
        sanity[qi] = {
            "n_tables": q.n_tables,
            "greedy_gap": optimality_gap(g, opt),
            "rl_gap_by_seed": rl_gaps,
            "rl_mean_gap": float(np.mean(list(rl_gaps.values()))),
        }
        print(f"  q{qi} ({q.n_tables} tables): greedy {sanity[qi]['greedy_gap']:.2f}% | "
              f"RL mean {sanity[qi]['rl_mean_gap']:.2f}% | per seed "
              + ", ".join(f"{s}:{v:.1f}" for s, v in rl_gaps.items()))

    # ---- Detailed replay ----
    report = {"set_seed": args.set_seed, "agent_seed": args.agent_seed,
              "note": "Diagnostic only. Do not tune on this set.", "queries": {}}
    for qi in args.queries:
        q = queries[qi]
        opt = exhaustive_min_cost(q)
        gp, rp = replay_greedy(q), replay_rl(agent, q)
        report["queries"][qi] = {
            "n_tables": q.n_tables,
            "cardinalities": list(q.cardinalities),
            "edges": [[list(e), round(q.selectivities[e], 4)] for e in sorted(q.edges)],
            "optimal_cost": opt,
            "greedy": {**gp, "gap_pct": optimality_gap(gp["total_cost"], opt)},
            "rl": {**rp, "gap_pct": optimality_gap(rp["total_cost"], opt)},
            "sanity": sanity[qi],
        }
        print(f"\n=== Query {qi} ({q.n_tables} tables) | optimal {opt:,} ===")
        print(f"  base cards: {list(q.cardinalities)}")
        print(f"  greedy cost {gp['total_cost']:,} ({optimality_gap(gp['total_cost'], opt):.1f}%)")
        for s in gp["steps"]:
            print(f"    G join {s['joined'][0]}+{s['joined'][1]} -> {s['rows_out']:,}  cost {s['step_cost']:,}")
        print(f"  RL (seed {args.agent_seed}) cost {rp['total_cost']:,} "
              f"({optimality_gap(rp['total_cost'], opt):.1f}%) | unseen chosen {rp['n_unseen_chosen']}/{len(rp['steps'])}")
        for k, s in enumerate(rp["steps"]):
            print(f"    step {k}: state {s['state_key']}")
            for o in s["options"]:
                mark = "<- chosen" if o["action"] == s["chosen"] else ""
                print(f"       {o['joins'][0]}+{o['joins'][1]} key={o['action_key']} "
                      f"Q={o['q']:.4f} {'seen' if o['seen'] else 'UNSEEN'} {mark}")
            print(f"       -> rows {s['rows_out']:,} cost {s['step_cost']:,}")

    with open(ROOT_DIR / "results" / "failure_analysis.json", "w") as f:
        json.dump(jsonable(report), f, indent=2)

    # ---- Clamp analysis on the full set ----
    clamp = clamp_analysis(queries, agent)
    clamp["cost_model_notes"] = {
        "MAX_CARD": MAX_CARD,
        "final_result_term_in_cost": True,
        "explanation": "step cost = rows_r + rows_s + rows_result, applied to every "
                       "join including the last, so the final output size IS charged.",
    }
    with open(ROOT_DIR / "results" / "clamp_analysis.json", "w") as f:
        json.dump(jsonable(clamp), f, indent=2)
    print("\n=== CLAMP ANALYSIS (all 100 queries, seed", args.set_seed, ") ===")
    print(json.dumps(jsonable(clamp), indent=2))


if __name__ == "__main__":
    main()

