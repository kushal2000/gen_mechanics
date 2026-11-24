"""Canonical joint signs for the vendor hands (23 Nov 2026): <hand>_left_canon.{urdf,spec.json}, registered as
<hand>_left_uniform_handonly_canon.

The grammar's joint convention: per finger, flexion axis x abduction axis = -(finger direction), i.e. +q on an abduction
joint turns the finger about +palm normal. Every vendor hand's abduction joints have the mirror convention (+q turns
the finger the other way; joint_conventions.py: cos -1 against the grammar replicas on all of Wuji / LEAP / Tesollo /
XHand's abduction joints), so the same action abducts a vendor finger the opposite way from any hand a grammar policy
trained on, and its joint_pos / joint-token axis read flipped. Negating such a joint's axis and its limits (lower,
upper) -> (-upper, -lower) is the same robot with q -> -q.

    .venv_isaacsim/bin/python experiments/23nov_replica_gap/make_canon.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
HANDS = ("wuji2", "tesollo", "xhand", "leap", "dex3", "allegro", "shadow")
D = REPO / "assets/urdf/unified_dynamics_commercial_hands"


def joints_to_flip(name: str) -> list[str]:
    from hand_sampler.design_space import joint_axes_and_link_vectors
    from isaacsimenvs.common.robots import get_robot_spec
    from isaacsimenvs.common.robots.finger_layout import finger_layout
    from coevolution.evolution.descriptors_v3 import _frames, joint_kinds
    s = get_robot_spec(name)
    pose, child_of = _frames(s)
    ax, _ = joint_axes_and_link_vectors(s.urdf_path, s.hand_joint_names, s.fingertip_body_names, s.fingertip_offsets)
    out = []
    for fi, (joints, tip) in enumerate(zip(finger_layout(s).fingers, s.fingertip_body_names)):
        base = pose(child_of[s.hand_joint_names[joints[0]]])[:3, 3]
        Mt = pose(tip)
        pad = Mt[:3, :3] @ np.asarray(s.fingertip_offsets[fi]) + Mt[:3, 3]
        f = (pad - base) / np.linalg.norm(pad - base)
        kinds = joint_kinds(s, pose, child_of, ax, joints, pad - base)
        wax = [pose(child_of[s.hand_joint_names[j]])[:3, :3] @ ax[j] for j in joints]
        flex = [k for k, kd in enumerate(kinds) if kd == "F"]
        if not flex:
            continue
        a_flex = wax[flex[-1]]
        for k, kd in enumerate(kinds):
            if kd == "A" and float(np.cross(a_flex, wax[k]) @ f) > 0:
                out.append(s.hand_joint_names[joints[k]])
    return out


def main() -> None:
    for h in HANDS:
      flip = joints_to_flip(f"{h}_left_uniform_handonly") if (D / h / f"{h}_left.spec.json").exists() else None
      for var in ("", "_boxlinks", "_plate"):          # the variants share the base hand's joints
        sp = D / h / f"{h}_left{var}.spec.json"
        if flip is None or not sp.exists():
            continue
        s = json.loads(sp.read_text())
        urdf_in = REPO / s["urdf"]
        txt = urdf_in.read_text()
        for jn in flip:
            m = re.search(r'<joint[^>]*name="%s"[^>]*>.*?</joint>' % re.escape(jn), txt, re.S)
            assert m, (h, jn)
            blk = m.group(0)
            new = re.sub(r'(<axis[^>]*xyz=")([^"]+)(")',
                         lambda a: a.group(1) + " ".join(f"{-float(x):g}" for x in a.group(2).split()) + a.group(3), blk)
            lo = re.search(r'lower="([^"]+)"', blk); hi = re.search(r'upper="([^"]+)"', blk)
            if lo and hi:
                new = re.sub(r'lower="[^"]+"', f'lower="{-float(hi.group(1)):g}"', new)
                new = re.sub(r'upper="[^"]+"', f'upper="{-float(lo.group(1)):g}"', new)
            assert new != blk, (h, jn)
            txt = txt.replace(blk, new)
            lo_s, hi_s = s["joint_limits"][jn]
            s["joint_limits"][jn] = [-hi_s, -lo_s]
            s["hand_default_joint_pos"][jn] = -s["hand_default_joint_pos"].get(jn, 0.0) + 0.0
        out_urdf = urdf_in.with_name(f"{h}_left{var}_canon.urdf")
        out_urdf.write_text(txt)
        s["urdf"] = str(out_urdf.relative_to(REPO))
        s["notes_canon"] = f"canonical joint signs (23nov_replica_gap/make_canon.py): flipped {flip}"
        (D / h / f"{h}_left{var}_canon.spec.json").write_text(json.dumps(s, indent=1))
        print(f"{h + var:16s} flipped {len(flip)}: {flip}")


if __name__ == "__main__":
    main()
