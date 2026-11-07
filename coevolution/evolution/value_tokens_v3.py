"""Design value net tokens for v3 hands (7 Nov 2026): the same layout as design_value_net.finger_tokens -- one token per
finger, 5 joint slots of [joint point in the palm frame (3, per 100 mm), signed axis in the palm frame (3), link vector
(3, per 100 mm), limits (2, rad), present (1)] + the finger's joint count / 5; hand vector [fingers / 6, motors / 25]
-- but from the v3 link-only geometry (hand_sampler/grammar/v3.py: straight fingers from their bases, points relative
to the palm's centroid), not derive.finger_geometry's Feetech servo geometry.

The v3 co-evolution driver installs it into design_value_net in its own process (`install()`); v1 / v2 processes keep
the grammar tokens.
"""
from __future__ import annotations

import math

import numpy as np

SLOTS, MAX_FINGERS = 5, 6


def finger_tokens(design: dict):
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import v3
    hand = gh.hand_from_dict(design)
    c = v3.palm_centre_mm(hand)
    toks = []
    for f in hand.fingers:
        B = v3.finger_frame(f)
        x = B[:, 0]
        p = np.array([0.0, f.y - c[0], f.z - c[1]])
        t = []
        for j in range(SLOTS):
            if j < len(f.joints):
                jt = f.joints[j]
                ax = B[:, 2] if tuple(jt.axis) == v3.FLEX else B[:, 1]
                lo, hi = (math.radians(v) for v in v3.KIND_RANGE_DEG[tuple(jt.axis)])
                link = x * float(jt.length)
                t += [*(p / 100.0), *ax, *(link / 100.0), lo, hi, 1.0]
                p = p + link
            else:
                t += [0.0] * 12
        toks.append(t + [len(f.joints) / SLOTS])
    motors = sum(len(f.joints) for f in hand.fingers)
    return np.asarray(toks, np.float32), np.asarray([len(hand.fingers) / MAX_FINGERS, motors / 25.0], np.float32)


def install() -> None:
    """Make design_value_net use these tokens (this process only)."""
    import design_value_net as dvn
    dvn.finger_tokens = finger_tokens


__all__ = ["finger_tokens", "install"]
