"""Hill climbing (with a little annealing) over the grammar's own rule-keeping operators, on pen_metrics' score.

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_design/search.py --seed-json <design.json> \
        --chain 0 --iters 150

Every child comes from `operators.mutate` under FEETECH_COARSE_RULES (so it keeps the rules and C1); children that
fail C2 are discarded. Objective: `search_score` = pen_metrics.score less 0.1 x (1 - reset_clean) (with --gait: plus 0.2 x gait_all).
Writes search_<chain>.jsonl (one line per evaluated hand) next to this file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pen_metrics as pm  # noqa: E402
from hand_sampler.grammar import hand as gh  # noqa: E402
from hand_sampler.grammar import operators as gops  # noqa: E402
from hand_sampler.grammar import viability as gvb  # noqa: E402

RULES = gh.FEETECH_COARSE_RULES


GAIT = False      # --gait: the objective also counts gait_all (needs 4+ fingers)


def search_score(m):
    s = pm.score(m) - 0.1 * (1.0 - m["reset_clean"])
    if GAIT:
        s += 0.2 * m["gait_all"]
    return s


def evaluate(h, fast=True):
    samples = [pm.finger_samples(h, i, 8000 if fast else pm.N_SAMPLES) for i in range(len(h.fingers))]
    m = pm.evaluate(h, n_goals=48 if fast else 96, n_axes=48 if fast else 96, samples=samples)
    m["reset_clean"] = pm.reset_clean(samples)
    if GAIT:
        m.update(pm.gait_hold(h, samples, n_goals=48))
    m["score"] = pm.score(m)
    m["search_score"] = search_score(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-json", required=True)
    ap.add_argument("--chain", type=int, default=0)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--temp", type=float, default=0.01)
    ap.add_argument("--gait", action="store_true")
    args = ap.parse_args()
    global GAIT
    GAIT = args.gait
    rng = np.random.default_rng(1000 + args.chain)
    h = gh.hand_from_dict(json.loads(Path(args.seed_json).read_text()))
    assert not gh.check(h, RULES) and gvb.clear(h)
    out = (HERE / f"search_{args.chain}.jsonl").open("w")
    cache = {}
    cur_m = evaluate(h)
    cur = h
    best, best_m = h, cur_m
    cache[gh.hand_to_json(h)] = cur_m
    out.write(json.dumps({"it": 0, "op": "seed", "accepted": True, **{k: v for k, v in cur_m.items()},
                          "design": gh.hand_to_dict(h)}) + "\n")
    out.flush()
    for it in range(1, args.iters + 1):
        t0 = time.time()
        res = gops.mutate(cur, rng, RULES, tries=6)
        if res is None:
            continue
        child, op, mv = res
        key = gh.hand_to_json(child)
        if key in cache:
            m = cache[key]
        else:
            if not gvb.c2_workspace_overlap(child).ok:
                continue
            m = evaluate(child)
            cache[key] = m
        d = m["search_score"] - cur_m["search_score"]
        acc = d >= 0 or rng.random() < math.exp(d / args.temp)
        if acc:
            cur, cur_m = child, m
        if m["search_score"] > best_m["search_score"]:
            best, best_m = child, m
        out.write(json.dumps({"it": it, "op": op, "accepted": bool(acc), **m, "design": gh.hand_to_dict(child)}) + "\n")
        out.flush()
        print(f"[{args.chain}] it {it} {op:13s} {m['search_score']:.3f} (cur {cur_m['search_score']:.3f}, best "
              f"{best_m['search_score']:.3f}) dof {m['dof']} {time.time() - t0:.1f}s", flush=True)
    (HERE / f"search_{args.chain}_best.json").write_text(json.dumps({"metrics": best_m,
                                                                      "design": gh.hand_to_dict(best)}, indent=1))


if __name__ == "__main__":
    main()
