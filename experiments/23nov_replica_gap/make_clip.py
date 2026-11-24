"""Vendor hands with grammar-range joint limits (23 Nov 2026): <hand>_left[_<var>]_canonclip -- the canonical-sign
hand (make_canon.py) with each joint's limits intersected with the grammar's range for its kind (v9: flexion -30..90,
abduction / twist -30..30 deg). A software joint limit: the robot is unchanged. Test of whether vendor abduction /
twist ranges far past the grammar's (Wuji -68..74, Tesollo 0..180, LEAP twist -27..140) are the replica -> vendor gap.

    .venv_isaacsim/bin/python experiments/23nov_replica_gap/make_clip.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
HANDS = ("wuji2", "tesollo", "xhand", "leap", "dex3", "allegro", "shadow")
D = REPO / "assets/urdf/unified_dynamics_commercial_hands"


def kinds_by_joint(name: str) -> dict:
    from hand_sampler.design_space import joint_axes_and_link_vectors
    from isaacsimenvs.common.robots import get_robot_spec
    from isaacsimenvs.common.robots.finger_layout import finger_layout
    from coevolution.evolution.descriptors_v3 import _frames, joint_kinds
    s = get_robot_spec(name)
    pose, child_of = _frames(s)
    ax, _ = joint_axes_and_link_vectors(s.urdf_path, s.hand_joint_names, s.fingertip_body_names, s.fingertip_offsets)
    out = {}
    for fi, (js, tip) in enumerate(zip(finger_layout(s).fingers, s.fingertip_body_names)):
        Mt = pose(tip)
        pad = Mt[:3, :3] @ np.asarray(s.fingertip_offsets[fi]) + Mt[:3, 3]
        ks = joint_kinds(s, pose, child_of, ax, js, pad - pose(child_of[s.hand_joint_names[js[0]]])[:3, 3])
        out.update({s.hand_joint_names[j]: k for j, k in zip(js, ks)})
    return out


def main() -> None:
    from hand_sampler.grammar import v3
    v3.configure("v9")
    rng = {"F": v3.KIND_RANGE_DEG["flex"], "A": v3.KIND_RANGE_DEG["abd"], "R": v3.KIND_RANGE_DEG["roll"]}
    for h in HANDS:
        if not (D / h / f"{h}_left_canon.spec.json").exists():
            continue
        kinds = kinds_by_joint(f"{h}_left_uniform_handonly_canon")
        for var in ("", "_boxlinks", "_plate"):
            sp = D / h / f"{h}_left{var}_canon.spec.json"
            if not sp.exists():
                continue
            s = json.loads(sp.read_text())
            txt = (REPO / s["urdf"]).read_text()
            changed = []
            for jn, k in kinds.items():
                lo, hi = s["joint_limits"][jn]
                glo, ghi = (math.radians(x) for x in rng[k])
                nlo, nhi = max(lo, glo), min(hi, ghi)
                if nlo >= nhi:                        # no overlap (a joint wholly outside the grammar range): keep a sliver
                    nlo, nhi = (glo, glo + 1e-3) if hi < glo else (ghi - 1e-3, ghi)
                if (nlo, nhi) == (lo, hi):
                    continue
                s["joint_limits"][jn] = [nlo, nhi]
                d = float(s["hand_default_joint_pos"].get(jn, 0.0))
                s["hand_default_joint_pos"][jn] = min(max(d, nlo), nhi)
                m = re.search(r'<joint[^>]*name="%s"[^>]*>.*?</joint>' % re.escape(jn), txt, re.S)
                blk = m.group(0)
                new = re.sub(r'lower="[^"]+"', f'lower="{nlo:g}"', blk)
                new = re.sub(r'upper="[^"]+"', f'upper="{nhi:g}"', new)
                txt = txt.replace(blk, new)
                changed.append(f"{jn}({k}) {math.degrees(lo):.0f}..{math.degrees(hi):.0f}->{math.degrees(nlo):.0f}..{math.degrees(nhi):.0f}")
            u = (REPO / s["urdf"]).with_name(f"{h}_left{var}_canonclip.urdf")
            u.write_text(txt)
            s["urdf"] = str(u.relative_to(REPO))
            s["notes_clip"] = f"limits clipped to the v9 grammar's kind ranges (23nov_replica_gap/make_clip.py): {changed}"
            (D / h / f"{h}_left{var}_canonclip.spec.json").write_text(json.dumps(s, indent=1))
            print(f"{h + var:16s} clipped {len(changed)}: {changed[:4]}{' ...' if len(changed) > 4 else ''}")


if __name__ == "__main__":
    main()
