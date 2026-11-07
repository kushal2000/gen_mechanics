"""v3 MAP-Elites cells (7 Nov 2026): fingers x motors x layout x knuckles x reach, every descriptor computed from the
hand's URDF the same way for a grammar design and a vendor hand, so commercial hands land in named cells.

  fingers   finger count
  motors    hand joints, 4-wide bins from 2n; every finger has 2-4 joints (the user's calls, 7 Nov): a hand with a
            one-joint or a 5-joint finger is outside the map
  layout    largest angular gap between neighbouring fingers about the palm normal: < 120 / 120-180 / 180-240 /
            >= 240 deg (the user's bins, 7 Nov; >= 240 is House of Dextra's "anthropomorphic": one-sided, opposing
            thumb). n fingers always leave a gap >= 360 / n, so 2 fingers have only the top two bins, 3 no < 120
  knuckles  each finger's base is one of three types (the user's definition, 7 Nov): 1-DOF, or a CO-LOCATED 2-DOF
            knuckle (first link < 2 mm, the grammar's 0 mm link) of a flexion and an abduction in either order -- kinds
            roll-invariant (joint_kinds: flexion parallel to the finger's last joint; for a v3 hand its own design), FE->AbAd or AbAd->FE; near-but-not-co-located joints (Wuji's 14 mm stub) and
            roll joints count as 1-DOF. Every finger of a design has the SAME base type (the user's
            simplification, 7 Nov): kn1dof / knFA / knAF; a mixed hand is outside the map
  reach     MAX palm-centre -> fingertip-pad distance over the fingers at q = 0 (mm; the joint tokens' frame): the
            hand's footprint. Stubs cannot pull a hand into a smaller bin (the mean could; the median is the mean for
            2 fingers) -- the user's call, 7 Nov

Why (experiments/06nov_coevolve_v2/WUJI_ZERO_SHOT_REPORT.md): the shared policy fails on structure its population
lacks -- one-sided layouts (42 -> 0 goals/ep), a base knuckle (47 -> 0), short base links -- and 6 of 7 commercial
hands sit in cells the v2 archive never visited.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

LAYOUT_EDGES = (0.0, 120.0, 180.0, 240.0, 361.0)
LAYOUT_NAMES = ("gap0-120", "gap120-180", "gap180-240", "gap240-360")
REACH_EDGES = (0.0, 120.0, 175.0, 1e9)            # MAX reach: tiny (no commercial hand) / Dex3 135, Wuji 145,
                                                  # Tesollo 151, Shadow 153, XHand 165 / LEAP 185, Allegro 202 + longer
REACH_NAMES = ("tiny", "mid", "large")
MOTOR_BIN = 4
KNUCKLE_LINK_MM = 2.0                              # co-located only (the grammar's 0 mm link)


def _frames(spec):
    from hand_sampler.design_space import rpy_to_mat
    root = ET.parse(spec.urdf_path).getroot()
    up = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        T = np.eye(4)
        if o is not None:
            T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            T[:3, :3] = rpy_to_mat([float(v) for v in o.get("rpy", "0 0 0").split()])
        up[j.find("child").get("link")] = (j.find("parent").get("link"), T, j.get("name"))

    def pose(link):
        M = np.eye(4)
        while link != spec.palm_body_name:
            p, T, _ = up[link]
            M = T @ M
            link = p
        return M
    child_of = {v[2]: k for k, v in up.items()}
    return pose, child_of


def joint_kinds(s, pose, child_of, ax_local, joints, d) -> list:
    """Each joint of a finger as "F" flexion / "A" abduction / "R" roll, ROLL-INVARIANTLY (7 Nov): roll if its axis
    lies along the finger (base -> fingertip), flexion if parallel to the finger's LAST joint axis (a commercial finger
    always ends in a flexion), abduction otherwise (square to both). Unlike an axis-vs-palm-normal rule it does not
    call a rolled thumb's flexions abductions (Wuji's MCP / IP axes are 25 deg off the palm normal)."""
    dirn = d / max(np.linalg.norm(d), 1e-9)
    A = [pose(child_of[s.hand_joint_names[j]])[:3, :3] @ ax_local[j] for j in joints]
    A = [a / max(np.linalg.norm(a), 1e-9) for a in A]
    last = A[-1]
    out = []
    for a in A:
        if abs(a @ dirn) > 0.7:
            out.append("R")
        elif abs(a @ last) > 0.7:
            out.append("F")
        else:
            out.append("A")
    return out


def _design_knuckles(spec_path: str):
    """(FA, AF) knuckle counts from a v3 hand's own design (its kinds relative to each finger's roll), or None for any
    other hand."""
    import json
    from pathlib import Path
    from hand_sampler import REPO_ROOT
    p = Path(spec_path)
    p = p if p.is_absolute() else REPO_ROOT / p
    if not str(spec_path).endswith(".spec.json") or not p.exists():
        return None
    d = json.loads(p.read_text())
    if d.get("rules") != "V3" or "design" not in d:
        return None
    from hand_sampler.grammar import v3
    h = v3.hand_from_dict(d["design"])
    return (sum(v3.base_type(f) == "knFA" for f in h.fingers), sum(v3.base_type(f) == "knAF" for f in h.fingers))


def features(name_or_spec: str) -> dict:
    from hand_sampler.design_space import joint_axes_and_link_vectors
    from isaacsimenvs.common.robots import get_robot_spec
    from isaacsimenvs.common.robots.finger_layout import finger_layout
    s = get_robot_spec(name_or_spec)
    pose, child_of = _frames(s)
    kp = np.asarray(s.palm_keypoints)
    e = kp[1:] - kp[:1]
    n_ax, w_ax, l_ax = (e / np.linalg.norm(e, axis=1)[:, None])
    c = np.asarray(s.palm_center_offset)
    ax_local, lv = joint_axes_and_link_vectors(s.urdf_path, s.hand_joint_names, s.fingertip_body_names,
                                               s.fingertip_offsets)
    lay = finger_layout(s)
    angs, reach, fa, af, rk = [], [], 0, 0, 0
    min_jpf = min(len(f_) for f_ in lay.fingers) if lay.fingers else 0
    max_jpf = max(len(f_) for f_ in lay.fingers) if lay.fingers else 0
    for f, tip in enumerate(s.fingertip_body_names):
        Mt = pose(tip)
        pad = Mt[:3, :3] @ np.asarray(s.fingertip_offsets[f]) + Mt[:3, 3]
        joints = lay.fingers[f]
        base = pose(child_of[s.hand_joint_names[joints[0]]])[:3, 3]
        d = pad - base
        angs.append(np.degrees(np.arctan2(d @ w_ax, d @ l_ax)))
        reach.append(float(np.linalg.norm(pad - c)))
        if len(joints) >= 2:
            kinds = joint_kinds(s, pose, child_of, ax_local, joints, d)[:2]
            if set(kinds) == {"F", "A"} and np.linalg.norm(lv[joints[0]]) * 1000 < KNUCKLE_LINK_MM:
                fa += kinds[0] == "F"
                af += kinds[0] == "A"
            elif kinds[0] != kinds[1] and "R" in kinds and np.linalg.norm(lv[joints[0]]) * 1000 < KNUCKLE_LINK_MM:
                rk += 1                                     # recorded, not a cell type (roll: Allegro, an outlier)
    dk = _design_knuckles(name_or_spec)
    if dk is not None:                                   # a v3 hand: its kinds by construction (rolls included)
        fa, af = dk
    a = np.sort(np.mod(angs, 360))
    gap = round(float(np.diff(np.concatenate([a, a[:1] + 360])).max()), 3) if len(a) > 1 else 360.0   # edges are on the 30 deg grid: no float drift across them
    return {"fingers": len(s.fingertip_body_names), "motors": len(s.hand_joint_names), "gap": gap,
            "reach_mm": 1000 * float(np.max(reach)), "reach_per_finger_mm": [round(1000 * r, 1) for r in reach], "knuckles_fa": fa, "knuckles_af": af, "knuckles_roll": rk,
            "min_joints_per_finger": int(min_jpf), "max_joints_per_finger": int(max_jpf)}


MIN_JOINTS_PER_FINGER = 2


def cell(f: dict) -> str | None:
    """The hand's v3 cell name, or None outside the map (a finger with fewer than MIN_JOINTS_PER_FINGER joints)."""
    if f.get("min_joints_per_finger", MIN_JOINTS_PER_FINGER) < MIN_JOINTS_PER_FINGER:
        return None
    if f.get("max_joints_per_finger", MAX_JOINTS_PER_FINGER) > MAX_JOINTS_PER_FINGER:
        return None
    b = lambda v, edges: int(np.searchsorted(edges, v, side="right") - 1)     # noqa: E731
    mb = motor_bin_of(f["fingers"], f["motors"]) if f["fingers"] in MOTOR_RANGE else None
    if mb is None:
        return None
    lo, hi = mb
    fa, af, n = f["knuckles_fa"], f["knuckles_af"], f["fingers"]
    if fa == 0 and af == 0:
        kn = "kn1dof"
    elif fa == n:
        kn = "knFA"
    elif af == n:
        kn = "knAF"
    else:
        return None                                                    # mixed base types: outside the map
    return (f"f{f['fingers']}_m{lo}-{hi}_{LAYOUT_NAMES[b(f['gap'], LAYOUT_EDGES)]}_{kn}_"
            f"{REACH_NAMES[b(f['reach_mm'], REACH_EDGES)]}")


__all__ = ["cell", "features"]


# ---------------------------------------------------------------- the feasible cell set (not the Cartesian product)
FINGERS = (2, 3, 4, 5, 6)
MAX_JOINTS_PER_FINGER = 4                         # v3 (the user, 7 Nov); the token grid has 5 slots
# v3 rules (7 Nov): 2-4 joints per finger -> motors 2n .. 4n, 4-wide bins starting at 2n
MOTOR_RANGE = {n: (MIN_JOINTS_PER_FINGER * n, MAX_JOINTS_PER_FINGER * n) for n in (2, 3, 4, 5, 6)}


# 4-wide motor bins from 2n, except 2 fingers: 4 / 5-6 / 7-8 (the user, 7 Nov)
MOTOR_BIN_WIDTH = {2: 4, 3: 4, 4: 4, 5: 4, 6: 4}
MOTOR_BINS_OVERRIDE = {2: [(4, 4), (5, 6), (7, 8)]}


def motor_bins(n: int) -> list[tuple[int, int]]:
    """Bins of MOTOR_BIN_WIDTH[n] joints from 2n; a last bin of half the width or less merges into the one before
    (4n is rarely on an edge)."""
    if n in MOTOR_BINS_OVERRIDE:
        return list(MOTOR_BINS_OVERRIDE[n])
    w = MOTOR_BIN_WIDTH[n]
    lo_m, hi_m = MOTOR_RANGE[n]
    bins = [(lo, min(lo + w - 1, hi_m)) for lo in range(lo_m, hi_m + 1, w)]
    if w > 1 and len(bins) > 1 and bins[-1][1] - bins[-1][0] + 1 <= w / 2:
        bins[-2:] = [(bins[-2][0], bins[-1][1])]
    return bins


def motor_bin_of(n: int, m: int) -> tuple[int, int] | None:
    return next(((lo, hi) for lo, hi in motor_bins(n) if lo <= m <= hi), None)
KNUCKLE_NAMES = ("kn1dof", "knFA", "knAF")


def knuckle_compositions(n: int) -> list[str]:
    return list(KNUCKLE_NAMES)


def analytic_cells() -> list[str]:
    """Cells allowed by construction: motors from 2 per finger to the finger count's top (4-wide bins), layout bins
    above 360 / n (n fingers always leave a gap at least that large), every knuckle composition of n fingers."""
    out = []
    for n in FINGERS:
        for lo, hi in motor_bins(n):
            for li, lay in enumerate(LAYOUT_NAMES):
                if LAYOUT_EDGES[li + 1] <= 360.0 / n:          # the largest gap is >= 360 / n
                    continue
                for kn in knuckle_compositions(n):
                    for rc in REACH_NAMES:
                        out.append(f"f{n}_m{lo}-{hi}_{lay}_{kn}_{rc}")
    return out
