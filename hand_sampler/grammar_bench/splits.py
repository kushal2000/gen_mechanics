"""Hands with a chosen split of their fingers over the palm and its palm parts,
for tests and simulator checks: ``split_hand(3, (2, 1))`` is 3 fingers on the
palm plus a jointed palm part with 2 and another with 1.

The fingers are copied from a hand of the one grammar (``variants.
build_distribution()``) sampled under ``limits.SIMULATOR``; palm parts are
straight 40 mm parts on the palm, each with a hinge joint (or rigid).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional, Sequence

from ..grammar.derive import Derivation, DerivationStep, sample_derivation, validate_derivation
from ..grammar.limits import SIMULATOR
from ..grammar.variants import build_distribution


def _donor_digits(n: int, seed: int):
    """``n`` (Digit step, its Phalanx steps) groups from grammar hands."""
    dist = build_distribution()
    out = []
    s = seed
    while len(out) < n:
        d = sample_derivation(s, dist, limits=SIMULATOR)
        digits = [st for st in d.steps if st.production == "Digit" and st.params["top_level"]]
        for dg in digits:
            did = dg.params["digit_id"]
            phal = [st for st in d.steps if st.production == "Phalanx" and st.params["digit_id"] == did]
            out.append((dg, phal))
        s += 1
    return out[:n]


def split_hand(root: int, palm_parts: Sequence[int] = (), jointed: Optional[Sequence[bool]] = None,
               seed: int = 0, palm_joint_limits=(-0.5, 0.5)) -> Derivation:
    """A derivation with ``root`` fingers on the palm and ``palm_parts[i]``
    fingers on palm part ``palm{i}`` (jointed unless ``jointed[i]`` is False)."""
    jointed = [True] * len(palm_parts) if jointed is None else list(jointed)
    n = root + sum(palm_parts)
    donors = _donor_digits(n, seed)
    root_length = 0.08
    steps: List[DerivationStep] = [DerivationStep(path="hand", production="Hand", params={
        "digit_count": n, "palm_body_count": len(palm_parts), "root_length": root_length, "capsule_radius_m": 0.01,
    })]
    uid = 0
    for i, _k in enumerate(palm_parts):
        a = 2.0 * math.pi * i / max(1, len(palm_parts))
        steps.append(DerivationStep(path=f"palm/{i}", production="PalmBody", params={
            "name": f"palm{i}", "parent": "root", "mount_frac": 0.0, "length": 0.04,
            "direction_rpy": (math.radians(60.0), 0.0, a), "has_joint": bool(jointed[i]),
            "axis": (1.0, 0.0, 0.0) if jointed[i] else (0.0, 0.0, 1.0),
            "limits": tuple(palm_joint_limits) if jointed[i] else None, "uid": uid,
        }))
        uid += 1
    hosts = ["root"] * root + [f"palm{i}" for i, k in enumerate(palm_parts) for _ in range(k)]
    counts = {h: hosts.count(h) for h in set(hosts)}
    seen = {h: 0 for h in counts}
    for k, ((dg, phal), host) in enumerate(zip(donors, hosts)):
        did = str(k + 1)
        frac = (seen[host] + 0.5) / counts[host] if host != "root" else 0.3 + 0.7 * (seen[host] + 0.5) / counts[host]
        seen[host] += 1
        p = dict(dg.params, digit_id=did, mount=host, mount_frac=round(frac, 6), uid=uid, top_level=True, depth=0)
        uid += 1
        steps.append(DerivationStep(path=f"digit/{did}", production="Digit", params=p))
        for ph in phal:
            q = dict(ph.params, digit_id=did, uid=uid)
            uid += 1
            steps.append(DerivationStep(path=f"digit/{did}/phalanx/{q['p']}", production="Phalanx", params=q))
    donor = sample_derivation(seed, build_distribution(), limits=SIMULATOR)
    d = replace(donor, steps=tuple(steps), lineage=())
    issues = validate_derivation(d)
    if issues:
        raise ValueError(f"split_hand({root}, {tuple(palm_parts)}): {issues}")
    return d


__all__ = ["split_hand"]
