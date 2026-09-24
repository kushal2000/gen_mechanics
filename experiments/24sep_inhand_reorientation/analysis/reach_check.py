"""Can these hands actually touch the cube?

In-hand reorientation puts a cube on a fixed palm-up palm. A design whose
fingers never rise above the palm surface cannot touch it at all, and one that
rises but not to the cube's equator can only push it around rather than turn
it. Those designs all score the same, so SELECTION HAS NOTHING TO RANK THEM BY
-- which is a property of the population, not of the training run, and is worth
knowing before spending GPUs rather than after.

Pure forward kinematics from ``design_space``: no Isaac Sim, no policy, seconds
to run. It is an upper bound on capability -- it asks whether a reachable joint
configuration EXISTS, ignoring whether the object is in the way, whether the
fingers collide, and whether a policy could ever find it. A design failing here
certainly cannot do the task; one passing merely might.

Geometry: the palm frame has +x = GRASP_DIR = the slab normal (thickness), and
the slab is centred at design_space.palm_center, so its outer face -- the
surface the cube rests on -- is at x = thickness/2. Reach is measured from
there. Touching a cube's equator needs edge/2 of clearance.

    .venv_isaacsim/bin/python experiments/24sep_inhand_reorientation/analysis/reach_check.py \
        handonly:gen_s0_n1024 --cube 0.045 --samples 600
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path[:0] = ["/share/portal/kk837/depthbasedRL/plot_figures"]
from hand_sampler import design_space, population_io, robot_spec


def fingertip_reach(hand, rng, n_samples: int) -> np.ndarray:
    """Best clearance above the palm's outer face, per finger, in metres.

    Replays link_frames' chain -- mount frame, then per joint a rotation of
    (offset + q) about axis_of and a translation of the previous link's length
    along local +x -- with q sampled inside each joint's limits instead of held
    at its rest offset.
    """
    palm_top = 0.5 * hand.palm.thickness          # +x face of the slab
    best = np.full(hand.n_fingers, -np.inf)
    for f, finger in enumerate(hand.fingers[:hand.n_fingers]):
        pos, rot = design_space.mount_frame(finger.mount, hand.palm)
        lo_hi = [(s.joint.limits or design_space.JOINT_LIMIT) for s in finger.segments]
        qs = np.stack([rng.uniform(lo, hi, n_samples) for lo, hi in lo_hi], axis=1)
        for k in range(n_samples):
            acc = np.eye(4)
            acc[:3, :3], acc[:3, 3] = rot, pos
            for d, seg in enumerate(finger.segments):
                step = np.eye(4)
                step[:3, :3] = design_space.rodrigues(
                    design_space.axis_of(seg.joint), seg.joint.offset + qs[k, d])
                if d:
                    step[:3, 3] = (finger.segments[d - 1].length, 0.0, 0.0)
                acc = acc @ step
            # the tip sits one link length past the last joint
            tip = acc @ np.array([finger.segments[-1].length, 0.0, 0.0, 1.0])
            best[f] = max(best[f], float(tip[0]) - palm_top)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("population", help="a population ref, name or .json (handonly: ok)")
    ap.add_argument("--cube", type=float, default=0.045, help="cube edge, metres")
    ap.add_argument("--samples", type=int, default=600, help="joint configs per finger")
    ap.add_argument("--designs", type=int, default=0, help="subsample; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    ref, _ = robot_spec.split_handonly(args.population)
    if robot_spec.is_population_file(ref):
        hands = population_io.load_population(ref)
    else:
        from hand_sampler import gen_init_pop
        import re
        m = re.fullmatch(robot_spec.POPULATION_NAME, ref)
        hands = gen_init_pop.seed_population(int(m.group(1)), int(m.group(2)))

    rng = np.random.default_rng(args.seed)
    if args.designs and args.designs < len(hands):
        idx = rng.choice(len(hands), args.designs, replace=False)
        hands = [hands[i] for i in sorted(idx)]

    print(f"{len(hands)} designs, {args.samples} joint configs per finger, "
          f"cube edge {1000*args.cube:.0f} mm\n", flush=True)
    reach = np.array([fingertip_reach(h, rng, args.samples).max() for h in hands])
    fingers = np.array([h.n_fingers for h in hands])
    joints = np.array([h.n_joints for h in hands])

    need = 0.5 * args.cube          # clearance to touch the equator
    rows = [("above the palm at all", 0.0), ("10 mm above", 0.010),
            ("cube equator", need), ("cube top", args.cube)]
    print("  best fingertip clearance above the palm's outer face")
    for label, thresh in rows:
        frac = float((reach > thresh).mean())
        print(f"    reaches {label:<24} ({1000*thresh:5.1f} mm): {100*frac:5.1f}% of designs")
    print(f"\n  reach percentiles (mm): "
          + "  ".join(f"p{p}={1000*np.percentile(reach, p):.0f}" for p in (5, 25, 50, 75, 95)))
    print(f"  fingers: mean {fingers.mean():.2f}  joints: mean {joints.mean():.2f}")

    can = reach > need
    if can.sum() and (~can).sum():
        print(f"\n  designs that CAN reach the equator: {fingers[can].mean():.2f} fingers, "
              f"{joints[can].mean():.2f} joints")
        print(f"  designs that CANNOT:                {fingers[~can].mean():.2f} fingers, "
              f"{joints[~can].mean():.2f} joints")
    if float(can.mean()) < 0.6:
        print(f"\n  WARNING: only {100*float(can.mean()):.0f}% of the population can reach the "
              f"cube's equator.\n  The rest score identically, so selection has little to rank. "
              f"Consider a smaller\n  cube, or seeding with more out-of-plane DOF, before "
              f"committing GPUs.")

    if args.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from _style import configure_rcparams, style_axis, COLORS
        configure_rcparams()
        out = pathlib.Path(__file__).resolve().parent.parent / "plots"; out.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(3.6, 2.7))
        ax.hist(1000 * reach, bins=40, color=COLORS["play2win"], alpha=0.85)
        ax.axvline(1000 * need, color=COLORS["separator"], ls=":", lw=1.2)
        ax.text(1000 * need, ax.get_ylim()[1] * 0.95, " cube equator", fontsize=7.5,
                color="#555", va="top")
        ax.set_xlabel("fingertip clearance above palm (mm)"); ax.set_ylabel("designs")
        ax.set_title("Can the hand reach the cube?", fontsize=11, pad=6)
        style_axis(ax); fig.tight_layout()
        p = out / f"reach_{pathlib.Path(ref).stem or ref}.png"
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
