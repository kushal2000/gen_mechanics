"""One row of population statistics per generation."""

from __future__ import annotations

import math
from collections import Counter

from hand_sampler import design_space
from hand_sampler import mutate_design


def topology(hand: design_space.Hand) -> tuple:
    """The discrete skeleton of a design -- what is left after forgetting every length and..."""
    return (tuple(sorted(f.n_joints for f in hand.fingers)),
            tuple(sorted((round(f.mount.y, 4), round(f.mount.z, 4),
                          round(f.mount.facing, 3)) for f in hand.fingers)))


def _q(sorted_xs: list, q: float):
    return sorted_xs[min(len(sorted_xs) - 1, int(q * len(sorted_xs)))]


def record(gen: int, pop: list, stats: mutate_design.Stats, nulls: int) -> dict:
    """Summarise one generation."""
    nf = [h.n_fingers for h in pop]
    nj = sorted(h.n_joints for h in pop)
    segs = [s for h in pop for f in h.fingers for s in f.segments]
    # Which quarter-turn each finger sits in, as a coarse read on how a
    # population spreads its fingers round the palm. There are no faces to count
    # any more; a bearing is a continuous angle on a disc.
    quads = Counter(int(f.mount.bearing / (math.pi / 2)) % 4
                    for h in pop for f in h.fingers)
    n_quads = sum(quads.values())
    mean = lambda xs: sum(xs) / len(xs)

    return dict(
        gen=gen,
        n_fingers=mean(nf),
        n_joints=mean(nj),
        joints_p10=_q(nj, 0.10), joints_p50=_q(nj, 0.50), joints_p90=_q(nj, 0.90),
        joints_hist={str(k): v for k, v in sorted(Counter(nj).items())},
        fingers_hist={str(k): v for k, v in sorted(Counter(nf).items())},
        link_mm=1000 * mean([s.length for s in segs]),
        kinds={n: sum(1 for s in segs if s.joint.kind == k) / len(segs)
               for k, n in zip(design_space.JOINT_KINDS, design_space.JOINT_KIND_NAMES)},
        leaning=sum(1 for s in segs if s.lean) / len(segs),
        palm_w_mm=1000 * mean([design_space.palm_extents(h)[1] for h in pop]),
        palm_l_mm=1000 * mean([design_space.palm_extents(h)[2] for h in pop]),
        mount_r_mm=1000 * mean([f.mount.radius for h in pop for f in h.fingers]),
        quadrant_share={str(q): quads.get(q, 0) / n_quads for q in range(4)},
        topology_diversity=len({topology(h) for h in pop}) / len(pop),
        null_rate=nulls / len(pop),
        rates={op: stats.rate(op) for op in mutate_design.OPERATORS},
    )
