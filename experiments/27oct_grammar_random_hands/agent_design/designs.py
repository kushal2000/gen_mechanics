"""Hand-written candidate designs (helpers to build a grammar hand from a compact spec)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from hand_sampler.grammar import hand as gh  # noqa: E402
from hand_sampler.grammar import operators as gops  # noqa: E402
from hand_sampler.grammar import viability as gvb  # noqa: E402

RULES = gh.FEETECH_COARSE_RULES
BEND, SPREAD, TWIST = (0, 0), (90, 0), (0, 90)
AX = {"b": BEND, "s": SPREAD, "t": TWIST}


def finger(y, z, facing, tilt, chain, lengths):
    """chain: e.g. "tbb"; lengths: wanted mm per link, snapped to the nearest allowed (fit from the tip back)."""
    js = [gh.Joint(axis=AX[c], length=int(L)) for c, L in zip(chain, lengths)]
    f = gh.Finger(y=y, z=z, facing=facing, tilt=tilt, joints=tuple(js))
    for _ in range(3):
        for j in range(len(js)):
            allowed = gh.allowed_lengths(f, j, True, RULES.max_link_mm, RULES.link_step_mm)
            L = min(allowed, key=lambda a: (abs(a - lengths[j]), a))
            jj = list(f.joints)
            jj[j] = gh.Joint(axis=jj[j].axis, length=L)
            f = gh.Finger(y=y, z=z, facing=facing, tilt=tilt, joints=tuple(jj))
    return f


def radial(y, z):
    import math
    return int(round(math.degrees(math.atan2(y, z)) / 30) * 30) % 360


def hand(spots, tilt, chain, lengths, facings=None):
    fs = []
    for k, (y, z) in enumerate(spots):
        fa = facings[k] if facings else radial(y, z)
        fs.append(finger(y, z, fa, tilt, chain, lengths))
    return gh.Hand(fingers=tuple(fs))


def status(h):
    v = gh.check(h, RULES)
    c1 = gvb.clear(h) if not v else None
    c2 = gvb.c2_workspace_overlap(h) if not v else None
    return v, c1, (c2.ok, round(c2.best_mm, 1)) if c2 else None


CROSS40 = [(0, 40), (40, 0), (0, -40), (-40, 0)]
CROSS80 = [(0, 80), (80, 0), (0, -80), (-80, 0)]
DIAG40 = [(40, 40), (40, -40), (-40, -40), (-40, 40)]
