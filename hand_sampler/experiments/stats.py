"""One row of population statistics per generation."""

from __future__ import annotations

import math
from collections import Counter

from hand_sampler import design_space
from hand_sampler import mutate_design


def topology(hand: design_space.Hand) -> tuple:
    """The discrete skeleton of a design -- what is left after forgetting every length and..."""
    return (tuple(sorted(f.n_joints for f in hand.fingers)),
            tuple(sorted(f.mount.face for f in hand.fingers)))


def _q(sorted_xs: list, q: float):
    return sorted_xs[min(len(sorted_xs) - 1, int(q * len(sorted_xs)))]


def record(gen: int, pop: list, stats: mutate_design.Stats, nulls: int) -> dict:
    """Summarise one generation."""
    nf = [h.n_fingers for h in pop]
    nj = sorted(h.n_joints for h in pop)
    segs = [s for h in pop for f in h.fingers for s in f.segments]
    faces = Counter(f.mount.face for h in pop for f in h.fingers)
    n_faces = sum(faces.values())
    mean = lambda xs: sum(xs) / len(xs)

    return dict(
        gen=gen,
        n_fingers=mean(nf),
        n_joints=mean(nj),
        joints_p10=_q(nj, 0.10), joints_p50=_q(nj, 0.50), joints_p90=_q(nj, 0.90),
        joints_hist={str(k): v for k, v in sorted(Counter(nj).items())},
        fingers_hist={str(k): v for k, v in sorted(Counter(nf).items())},
        link_mm=1000 * mean([s.length for s in segs]),
        theta_deg=math.degrees(mean([s.joint.theta for s in segs])),
        offset_deg=math.degrees(mean([abs(s.joint.offset) for s in segs])),
        palm_w_mm=1000 * mean([h.palm.width for h in pop]),
        palm_l_mm=1000 * mean([h.palm.length for h in pop]),
        face_share={f: faces.get(f, 0) / n_faces for f in design_space.FINGER_FACES},
        topology_diversity=len({topology(h) for h in pop}) / len(pop),
        null_rate=nulls / len(pop),
        rates={op: stats.rate(op) for op in mutate_design.OPERATORS},
    )
