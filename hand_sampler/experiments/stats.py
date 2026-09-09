"""One row of population statistics per generation.

Rows are recorded on the CHILDREN, before the next selection step, so a row
describes the population a fitness function would be handed.

Everything here is a plain dict of JSON scalars: a run's output has to outlive
the code that produced it, and a pickled numpy array does not.
"""

from __future__ import annotations

import math
from collections import Counter

from hand_sampler import design_space
from hand_sampler import mutate_design


def topology(hand: design_space.Hand) -> tuple:
    """The discrete skeleton of a design -- what is left after forgetting every
    length and angle.

    Diversity has to be counted on THIS, not on genotype identity. A seed's
    mount ``v`` is drawn from a continuous range, so essentially every hand is
    unique no matter how morphologically converged the population is: counting
    distinct genotypes measures the null-mutation copy rate and nothing else.
    """
    return (tuple(sorted(f.n_joints for f in hand.fingers)),
            tuple(sorted(f.mount.face for f in hand.fingers)))


def _q(sorted_xs: list, q: float):
    return sorted_xs[min(len(sorted_xs) - 1, int(q * len(sorted_xs)))]


def record(gen: int, pop: list, stats: mutate_design.Stats, nulls: int) -> dict:
    """Summarise one generation.

    The joint HISTOGRAM is recorded, not just percentiles: the question of
    whether the cheap end of the (performance, n_motors) front still gets
    sampled is about the tail, and a p10 cannot answer it.
    """
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
