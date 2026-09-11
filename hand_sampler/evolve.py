"""One generation of selection: rank designs by return, keep the top half,
mutate each survivor once to refill.

(mu + lambda) with mu = lambda: the survivors persist unchanged and each
contributes one child, so a generation's population is half proven designs and
half one-step variations of them. Nothing here trains -- it reads the per-design
table `coevolution.design_rewards` wrote during a generation, and writes the next
generation's population file plus a `selection.json` saying exactly who lived,
who died, and what each child was made from.

A design with no completed episodes -- which should not happen at 24 envs a
design, but a short generation could do it -- ranks last rather than being
guessed at.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from hand_sampler import design_space, mutate_design, population_io, validate_design

MUTATION_TRIES = 64
"""Independent draws before a survivor is copied unchanged instead of mutated.
Rejection is normal -- an operator that cannot act raises MutationImpossible --
but a survivor that cannot be mutated at all after this many is a corner of the
grammar, and copying it keeps the count right without inventing a design."""


def rank_designs(rewards: dict, n_designs: int, *, key: str = "return_mean") -> list[int]:
    """Design indices, best first. Unscored designs go last, in index order."""
    scored = [(i, rewards[i][key]) for i in range(n_designs) if i in rewards]
    unscored = [i for i in range(n_designs) if i not in rewards]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [i for i, _ in scored] + unscored


def mutate_once(rng: random.Random, hand, *, tries: int = MUTATION_TRIES):
    """One valid child, or ``None`` if the grammar would not give one."""
    for _ in range(tries):
        op = mutate_design.OPERATORS[rng.randrange(len(mutate_design.OPERATORS))]
        child = mutate_design.mutate(rng, hand, op)
        if child is not None and not validate_design.check(child):
            return child, op
    return None, None


def next_generation(hands, rewards: dict, *, keep: int, seed: int,
                    key: str = "return_mean"):
    """``(new_hands, selection)``. Survivors first, then their children in the
    same order, so child ``i`` is survivor ``i``'s."""
    n = len(hands)
    if keep <= 0 or keep > n:
        raise ValueError(f"keep must be in 1..{n}, got {keep}")
    order = rank_designs(rewards, n, key=key)
    survivors = order[:keep]
    culled = order[keep:]

    rng = random.Random(seed)
    new_hands, children = [hands[i] for i in survivors], []
    for parent in survivors:
        child, op = mutate_once(rng, hands[parent])
        children.append({"parent": parent, "op": op, "copied": child is None})
        new_hands.append(child if child is not None else hands[parent])

    # Refill to exactly n: with keep == n // 2 this is a no-op, but keep is a
    # knob and the population size is not.
    while len(new_hands) < n:
        parent = survivors[len(new_hands) % keep]
        child, op = mutate_once(rng, hands[parent])
        children.append({"parent": parent, "op": op, "copied": child is None})
        new_hands.append(child if child is not None else hands[parent])
    new_hands = new_hands[:n]

    score = lambda i: rewards.get(i, {}).get(key)
    selection = {
        "key": key, "keep": keep, "seed": seed, "n": n,
        "ranking": order,
        "survivors": survivors,
        "culled": culled,
        "survivor_scores": [score(i) for i in survivors],
        "culled_scores": [score(i) for i in culled],
        "children": children,
        "unscored": [i for i in range(n) if i not in rewards],
        "summary": {
            "best": score(order[0]),
            "survivor_mean": _mean([score(i) for i in survivors]),
            "culled_mean": _mean([score(i) for i in culled]),
            "population_mean": _mean([score(i) for i in order]),
            "copied_children": sum(1 for c in children if c["copied"]),
            "mean_joints_before": sum(h.n_joints for h in hands) / n,
            "mean_joints_after": sum(h.n_joints for h in new_hands) / n,
        },
    }
    return new_hands, selection


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    import argparse

    from coevolution.design_rewards import merge_rank_files

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("population", help="this generation's population .json")
    ap.add_argument("--rewards", nargs="+", required=True,
                    help="design_rewards_rank*.json files from this generation's run")
    ap.add_argument("--out", required=True, help="next generation's population .json")
    ap.add_argument("--keep", type=int, default=None, help="survivors (default: half)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--key", default="return_mean", choices=("return_mean", "success_rate"))
    ap.add_argument("--generation", type=int, default=None, help="for provenance")
    args = ap.parse_args()

    hands = population_io.load_population(args.population)
    rewards = merge_rank_files(args.rewards)
    keep = args.keep or len(hands) // 2
    new_hands, selection = next_generation(hands, rewards, keep=keep, seed=args.seed, key=args.key)

    out = Path(args.out)
    population_io.save_population(
        new_hands, out, name=out.stem,
        provenance=population_io.provenance(
            method="evolve", source=str(args.population), generation=args.generation,
            keep=keep, seed=args.seed, key=args.key,
            rewards=[str(p) for p in args.rewards], **selection["summary"]))
    out.with_name("selection.json").write_text(json.dumps(selection, indent=1))

    s = selection["summary"]
    fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
    n_ep = sum(r["episodes"] for r in rewards.values())
    print(f"[evolve] {len(hands)} designs, {n_ep} episodes scored, "
          f"{len(selection['unscored'])} designs unscored")
    if len(selection["unscored"]) > len(hands) // 2:
        # Ranking among unscored designs is index order, i.e. arbitrary. A
        # generation this short selected on nothing; say so loudly.
        print(f"[evolve] WARNING: {len(selection['unscored'])} of {len(hands)} designs "
              f"completed no episode -- this selection is mostly arbitrary", flush=True)
    print(f"[evolve] best {fmt(s['best'])}  survivors {fmt(s['survivor_mean'])}  "
          f"culled {fmt(s['culled_mean'])}  population {fmt(s['population_mean'])}")
    print(f"[evolve] joints/hand {s['mean_joints_before']:.2f} -> {s['mean_joints_after']:.2f}; "
          f"{s['copied_children']} children copied unmutated")
    print(f"[evolve] wrote {out} and selection.json")


if __name__ == "__main__":
    main()
