"""Generate and cache a population of collision-free hands.

Run once; every consumer then loads from disk instead of resampling. The
geometric gate rejects the large majority of draws, so this is minutes of work
that has no business happening inside a demo, a viewer, or a training launch.

    .venv_isaacsim/bin/python -m genmech.tools.build_hand_population \\
        --seed 0 --count 64

Needs no Isaac Sim: sampling, the collision check and the URDF writer are all
plain Python plus yourdfpy/trimesh.
"""

from __future__ import annotations

import argparse

from genmech.robots.generated.population import (
    build_population,
    load_population,
    manifest_path,
    population_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--max_tries_per_hand", type=int, default=400)
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if a manifest already exists")
    parser.add_argument("--list", action="store_true",
                        help="show what is cached and exit")
    args = parser.parse_args()

    if args.list:
        path = manifest_path(args.seed)
        if not path.exists():
            print(f"[population] nothing cached for seed {args.seed} ({path})")
            return
        hands = load_population(args.seed)
        print(f"[population] {len(hands)} hands in {population_dir(args.seed)}")
        for h in hands:
            live = [f for f in h.fingers if f.active]
            joints = sum(sum(f.enabled) for f in live)
            print(f"   {h.name:<20} {len(live)} fingers, {joints} live joints")
        return

    hands = build_population(args.seed, args.count,
                            max_tries_per_hand=args.max_tries_per_hand,
                            force=args.force)
    print(f"[population] {len(hands)} hands ready; "
          f"load with population.load_population({args.seed})")


if __name__ == "__main__":
    main()
