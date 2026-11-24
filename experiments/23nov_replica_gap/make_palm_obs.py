"""Vendor hands whose OBSERVED palm matches the grammar's (23 Nov 2026) -- the robot, its URDF and its palm frame / centre
unchanged; only spec palm_extents_ours, which sets palm_keypoints and the actor's palm_extents observation:
  <hand>_left[_plate]_canonclip_thin    thickness 20 mm (every grammar palm; vendor palms 35-50 mm)
  <hand>_left[_plate]_canonclip_repext  all three extents from the hand's exact replica (commercial_exact_axes_canon)

    .venv_isaacsim/bin/python experiments/23nov_replica_gap/make_palm_obs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
HANDS = ("wuji2", "tesollo", "xhand", "leap", "dex3", "allegro", "shadow")
D = REPO / "assets/urdf/unified_dynamics_commercial_hands"
REP = REPO / "debug_outputs/coevolve/v9/commercial_exact_axes_canon"


def main() -> None:
    from isaacsimenvs.common.robots import get_robot_spec
    for h in HANDS:
        rep = REP / f"{h}_v7" / f"{h}_v7.spec.json"
        rep_ext = None
        if rep.exists():
            kp = np.asarray(get_robot_spec(str(rep.relative_to(REPO))).palm_keypoints)
            rep_ext = [float(x) for x in np.linalg.norm(kp[1:] - kp[:1], axis=1)]
        for var in ("", "_plate"):
            sp = D / h / f"{h}_left{var}_canonclip.spec.json"
            if not sp.exists():
                continue
            base = json.loads(sp.read_text())
            for tag, ext in (("thin", [0.02] + list(base["palm_extents_ours"][1:])), ("repext", rep_ext)):
                if ext is None:
                    continue
                s = dict(base, palm_extents_ours=ext)
                s["notes_palm_obs"] = f"observed palm extents {np.round(ext, 3).tolist()} (was {np.round(base['palm_extents_ours'], 3).tolist()})"
                (D / h / f"{h}_left{var}_canonclip_{tag}.spec.json").write_text(json.dumps(s, indent=1))
                print(f"{h + var:14s} {tag:6s} {s['notes_palm_obs']}")


if __name__ == "__main__":
    main()
