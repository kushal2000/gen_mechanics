"""Run the evolution loop and record population statistics -- resumably.

    python -m hand_sampler.experiments.run --out runs/null --gens 1000 --seed 7
    python -m hand_sampler.experiments.run --out runs/null --gens 9000

The second call CONTINUES the first. Both the population and the RNG state are
checkpointed, so 1k + 9k is the SAME chain as 10k in one go -- restarting the
RNG instead would give a statistically different run that looks identical from
the outside. Long runs are meant to be grown in stages; that only works if a
stage boundary leaves no trace in the statistics.

Selection modes:
  random      no selection at all. The grammar's own prior -- the null that any
              fitness result has to be read against.
  min_joints  / max_joints
              select on joint count at the same truncation as the real loop.
              Not a fitness function: a yardstick for how fast selection CAN
              move a statistic, so a real run's rate can be read as a fraction
              of it.

Plug a real evaluator in by replacing ``select``; nothing else here knows what
a fitness function is.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import random
import time

from hand_sampler import mutate as M
from hand_sampler import sample as S
from hand_sampler.experiments import stats as ST

MODES = ("random", "min_joints", "max_joints")
CHECKPOINT = "checkpoint.pkl.gz"
STATS = "stats.jsonl"
CONFIG = "config.json"


def select(rng: random.Random, pop: list, n: int, mode: str) -> list:
    """The ``n`` parents of the next generation."""
    if mode == "random":
        return rng.sample(pop, n)
    # Random tiebreak. Without it ties resolve by list position, which is
    # lineage order -- a selection pressure nobody asked for, and one that
    # would bite hardest exactly at a floor or ceiling where ties are the rule.
    ranked = sorted(pop, key=lambda h: (h.n_joints, rng.random()),
                    reverse=(mode == "max_joints"))
    return ranked[:n]


def step(rng: random.Random, pop: list, parents_n: int, children_n: int,
         mode: str) -> tuple[list, M.Stats, int]:
    """One generation: select, then mutate each parent ``children_n`` times."""
    stats, nulls, children = M.Stats(), 0, []
    for parent in select(rng, pop, parents_n, mode):
        for _ in range(children_n):
            child = M.mutate(rng, parent, stats=stats)
            if child is None:            # operator could not act on this hand
                nulls += 1
                child = parent
            children.append(child)
    return children, stats, nulls


def _save(out: str, pop: list, rng: random.Random, gen: int) -> None:
    """Atomically, so an interrupted stage cannot leave an unreadable chain."""
    tmp = os.path.join(out, CHECKPOINT + ".tmp")
    with gzip.open(tmp, "wb") as fh:
        pickle.dump({"pop": pop, "rng": rng.getstate(), "gen": gen}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, os.path.join(out, CHECKPOINT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run directory; resumed if it exists")
    ap.add_argument("--gens", type=int, required=True,
                    help="generations to run NOW, added to any already done")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", choices=MODES, default="random")
    ap.add_argument("--parents", type=int, default=2000, help="selected per generation")
    ap.add_argument("--children", type=int, default=10, help="children per parent")
    ap.add_argument("--every", type=int, default=50, help="progress print interval")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg_path = os.path.join(args.out, CONFIG)
    cfg = dict(seed=args.seed, mode=args.mode,
               parents=args.parents, children=args.children)

    ckpt = os.path.join(args.out, CHECKPOINT)
    if os.path.exists(ckpt):
        old = json.load(open(cfg_path))
        if old != cfg:
            # Resuming under a different shape would splice two different
            # experiments into one statistics file, with nothing in the output
            # to show where the seam is.
            raise SystemExit(
                f"{args.out} was created with {old}, but this run asks for "
                f"{cfg}. Use a new --out, or match the original settings.")
        with gzip.open(ckpt, "rb") as fh:
            state = pickle.load(fh)
        pop, gen0 = state["pop"], state["gen"]
        rng = random.Random()
        rng.setstate(state["rng"])
        print(f"resuming {args.out} at generation {gen0}", flush=True)
    else:
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        rng = random.Random(args.seed)
        pop = S.seed_population(args.seed, args.parents * args.children)
        gen0 = 0
        print(f"seeded {len(pop)} hands, mode={args.mode}", flush=True)

    t0 = time.perf_counter()
    with open(os.path.join(args.out, STATS), "a") as fh:
        for i in range(args.gens):
            gen = gen0 + i
            pop, st, nulls = step(rng, pop, args.parents, args.children, args.mode)
            row = ST.record(gen, pop, st, nulls)
            fh.write(json.dumps(row) + "\n")
            fh.flush()               # a killed run keeps every generation it finished
            if i % args.every == 0 or i == args.gens - 1:
                print(f"g{gen:>6d}  {row['n_fingers']:.2f}f {row['n_joints']:6.2f}j  "
                      f"link {row['link_mm']:4.1f}mm  topo-div "
                      f"{row['topology_diversity']:.3f}  null {row['null_rate']:.3f}"
                      f"  [{time.perf_counter() - t0:.0f}s]", flush=True)

    _save(args.out, pop, rng, gen0 + args.gens)
    print(f"{args.out}: {gen0 + args.gens} generations done "
          f"({time.perf_counter() - t0:.0f}s this stage)")


if __name__ == "__main__":
    main()
