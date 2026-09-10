"""Mutate a population with NO selection, until its statistics stop moving.

Generation 0 is deliberately small -- one or two joints per finger, two fingers,
so a hand starts with 2 to 4 motors -- and a hand that underactuated may simply
be unable to do the task, which would look like the multi-embodiment path
failing rather than the population being too simple.

This walks the population away from that starting point under the mutation
operators alone. No fitness, no selection: every hand takes its own random step
each round and keeps it if it validates. Where that settles is a property of the
GRAMMAR, not of any policy -- the operators are symmetric in pairs (split_link /
merge_links, add_finger / remove_finger, uniform over nine), so nothing pushes
toward complexity by construction. What decides the equilibrium is which
mutations remain FEASIBLE: `remove_finger` needs a single-joint finger,
`split_link` needs a link long enough to divide, and `MIN_FINGERS` and the
`MAX_FINGERS x MAX_JOINTS_PER_FINGER` envelope are hard walls.

So whether the equilibrium is more actuated than generation 0 is a question to
measure, not to predict.
"""

from __future__ import annotations

import random
from collections import Counter

from hand_sampler import design_space, mutate_design


def population_stats(hands) -> dict:
    """The summary a round is judged converged on."""
    joints = [h.n_joints for h in hands]
    fingers = [h.n_fingers for h in hands]
    return {
        "n": len(hands),
        "mean_joints": sum(joints) / len(hands),
        "mean_fingers": sum(fingers) / len(hands),
        "joint_hist": dict(sorted(Counter(joints).items())),
        "finger_hist": dict(sorted(Counter(fingers).items())),
        "distinct": len({repr(h) for h in hands}),
        "at_joint_cap": sum(
            1 for h in hands
            if any(f.n_joints >= design_space.MAX_JOINTS_PER_FINGER for f in h.fingers)),
        "at_finger_cap": sum(1 for h in hands if h.n_fingers >= design_space.MAX_FINGERS),
    }


def drift_round(rng: random.Random, hands, *, per_round: int = 1):
    """One step per hand. A hand whose operator cannot act keeps its parent --
    rejection is part of the walk, not an error."""
    out, accepted = [], Counter()
    for hand in hands:
        for _ in range(per_round):
            op = mutate_design.OPERATORS[rng.randrange(len(mutate_design.OPERATORS))]
            child = mutate_design.mutate(rng, hand, op)
            accepted[op] += child is not None
            if child is not None:
                hand = child
        out.append(hand)
    return out, accepted


def has_settled(trace, *, window: int = 10, tol: float = 0.01) -> bool:
    """True once the last ``window`` rounds look like the ``window`` before them.

    Compares block means rather than consecutive rounds: a single round moves by
    chance even at equilibrium, and stopping on that would report a stable point
    that is only a quiet step.
    """
    if len(trace) < 2 * window:
        return False
    for key in ("mean_joints", "mean_fingers"):
        prev = sum(r[key] for r in trace[-2 * window:-window]) / window
        last = sum(r[key] for r in trace[-window:]) / window
        if abs(last - prev) >= tol:
            return False
    return True


def drift(hands, *, max_rounds: int = 500, per_round: int = 1, seed: int = 0,
          window: int = 10, tol: float = 0.01, verbose: bool = True, on_round=None):
    """Walk until settled or ``max_rounds``. Returns ``(hands, trace)``.

    ``on_round(round, hands, row)`` is called after every round, including round
    zero, so a caller can snapshot the population as it moves rather than only
    where it stops.
    """
    rng = random.Random(seed)
    hands = list(hands)
    trace = [{"round": 0, **population_stats(hands)}]
    if verbose:
        _log(trace[-1], None)
    if on_round is not None:
        on_round(0, hands, trace[-1])

    for r in range(1, max_rounds + 1):
        hands, accepted = drift_round(rng, hands, per_round=per_round)
        row = {"round": r, **population_stats(hands), "accepted": dict(accepted)}
        trace.append(row)
        if on_round is not None:
            on_round(r, hands, row)
        if verbose and (r <= 3 or r % 10 == 0):
            _log(row, accepted)
        if has_settled(trace, window=window, tol=tol):
            if verbose:
                print(f"[drift] settled after {r} rounds "
                      f"(block means within {tol} over {window})", flush=True)
            break
    else:
        if verbose:
            print(f"[drift] STOPPED at max_rounds={max_rounds} without settling; "
                  f"the population is still moving", flush=True)
    return hands, trace


def _log(row, accepted) -> None:
    rate = ""
    if accepted:
        total = sum(accepted.values())
        rate = f"  accepted/round {total}"
    print(f"[drift] round {row['round']:4d}  joints {row['mean_joints']:5.2f}  "
          f"fingers {row['mean_fingers']:4.2f}  distinct {row['distinct']:6d}  "
          f"at finger cap {row['at_finger_cap']:6d}{rate}", flush=True)


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from hand_sampler import population_io
    from hand_sampler.robot_spec import population_from_ref

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="population name or .json to start from")
    ap.add_argument("--label", default=None,
                    help="names the experiment's subfolder under assets/populations/ "
                         "(default: <source>_drift_s<seed>)")
    ap.add_argument("--max-rounds", type=int, default=500)
    ap.add_argument("--per-round", type=int, default=1,
                    help="mutation attempts per hand per round")
    ap.add_argument("--snapshot-every", type=int, default=1,
                    help="write round_NNNN.json every Nth round; 0 writes only the ends")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--force", action="store_true",
                    help="proceed past the projected-size check")
    args = ap.parse_args()

    label = args.label or (
        f"{Path(args.source).stem if args.source.endswith('.json') else args.source}"
        f"_drift_s{args.seed}")
    out_dir = population_io.DEFAULT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[drift] loading {args.source} ...", flush=True)
    hands = list(population_from_ref(args.source).hands)

    # One round on disk, measured rather than guessed, so the projection below
    # is the real number: 24576 designs is ~18 MB a round and 600 rounds of that
    # is 11 GB. Cheap next to a training run, but not something to write by
    # accident onto a share that is 94% full.
    def snapshot_path(r: int) -> Path:
        return out_dir / f"round_{r:04d}.json"

    def write_round(r: int, population, row: dict) -> None:
        population_io.save_population(
            population, snapshot_path(r), name=f"{label}_round_{r:04d}",
            provenance=population_io.provenance(
                method="drift", source=args.source, seed=args.seed, round=r,
                per_round=args.per_round, mean_joints=row["mean_joints"],
                mean_fingers=row["mean_fingers"]))

    write_round(0, hands, {"mean_joints": 0.0, "mean_fingers": 0.0})
    per_round_bytes = snapshot_path(0).stat().st_size
    n_snaps = 1 if args.snapshot_every <= 0 else 1 + args.max_rounds // args.snapshot_every
    projected = per_round_bytes * n_snaps
    print(f"[drift] {per_round_bytes/1e6:.1f} MB a round x up to {n_snaps} snapshots "
          f"= up to {projected/1e9:.1f} GB in {out_dir}", flush=True)
    if projected > 50e9 and not args.force:
        raise SystemExit(
            f"projected {projected/1e9:.0f} GB. Raise --snapshot-every, lower "
            f"--max-rounds, or pass --force.")

    def on_round(r, population, row):
        if r == 0:
            write_round(0, population, row)          # rewrite with real stats
        elif args.snapshot_every > 0 and r % args.snapshot_every == 0:
            write_round(r, population, row)

    drifted, trace = drift(hands, max_rounds=args.max_rounds, per_round=args.per_round,
                           seed=args.seed, window=args.window, tol=args.tol,
                           on_round=on_round)

    last = trace[-1]["round"]
    if not snapshot_path(last).exists():
        write_round(last, drifted, trace[-1])
    settled = has_settled(trace, window=args.window, tol=args.tol)
    population_io.save_population(
        drifted, out_dir / "final.json", name=label,
        provenance=population_io.provenance(
            method="drift", source=args.source, seed=args.seed, rounds=last,
            per_round=args.per_round, window=args.window, tol=args.tol, settled=settled,
            mean_joints_before=trace[0]["mean_joints"],
            mean_joints_after=trace[-1]["mean_joints"]))
    (out_dir / "trace.json").write_text(json.dumps(trace, indent=1), encoding="utf-8")

    first = trace[0]
    (out_dir / "README.md").write_text(
        f"# {label}\n\n"
        f"Neutral drift -- mutation with NO selection -- from `{args.source}`.\n"
        f"Seed {args.seed}, {args.per_round} attempt(s) per hand per round, "
        f"{last} rounds, {'settled' if settled else 'STOPPED WITHOUT SETTLING'} "
        f"(block means within {args.tol} over {args.window}).\n\n"
        f"| | generation 0 | round {last} |\n|---|---|---|\n"
        f"| joints/hand | {first['mean_joints']:.2f} | {last and trace[-1]['mean_joints']:.2f} |\n"
        f"| fingers/hand | {first['mean_fingers']:.2f} | {trace[-1]['mean_fingers']:.2f} |\n"
        f"| at the {design_space.MAX_FINGERS}-finger cap | {first['at_finger_cap']} | "
        f"{trace[-1]['at_finger_cap']} | \n\n"
        f"`round_NNNN.json` is the population after that round; `final.json` is the\n"
        f"last one, and `trace.json` has per-round statistics for every round\n"
        f"whether or not it was snapshotted. Each file carries its own provenance.\n\n"
        f"The operators are symmetric in pairs, so nothing here pushes toward\n"
        f"complexity by construction -- the population climbs because generation 0\n"
        f"sits against the MIN_FINGERS wall, and where it stops is a property of\n"
        f"the grammar's feasibility boundaries.\n",
        encoding="utf-8")

    print(f"\n[drift] generation 0 -> equilibrium")
    print(f"  joints/hand  {first['mean_joints']:.2f} -> {trace[-1]['mean_joints']:.2f}")
    print(f"  fingers/hand {first['mean_fingers']:.2f} -> {trace[-1]['mean_fingers']:.2f}")
    print(f"  joint hist   {first['joint_hist']}")
    print(f"            -> {trace[-1]['joint_hist']}")
    print(f"[drift] {len(list(out_dir.glob('round_*.json')))} round snapshots in {out_dir}")


if __name__ == "__main__":
    main()
