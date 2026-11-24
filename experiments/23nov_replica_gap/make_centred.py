"""Vendor hands with the grammar's palm centre (23 Nov 2026): <hand>_left_<tag>_centred, tag canon / canonclip.

The env measures every joint token from the palm centre and places each hand by it (palm centre at the same spot over
the pen). The grammar's palm centre is the centre of the outline around the finger bases (v3.centred); a vendor
spec's is the middle of the vendor palm box, which reaches back toward the wrist -- so in the policy's frame every
vendor joint sits ~20 mm further along the palm (and 6-9 mm along the normal) than on the hand's exact replica
(compare_tokens.py). Here the palm centre moves by the mean token offset to the exact replica (palm_centre_in_palm_body
and palm_centre_ours together); the robot is unchanged, its placement moves with the centre, as the replica's does.

    .venv_isaacsim/bin/python experiments/23nov_replica_gap/make_centred.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))
HANDS = ("wuji2", "tesollo", "xhand", "leap", "dex3", "allegro")
D = REPO / "assets/urdf/unified_dynamics_commercial_hands"


def main() -> None:
    from compare_tokens import tokens
    from isaacsimenvs.common.robots import get_robot_spec
    for h in (HANDS if "--check" not in sys.argv else ()):
        rep = f"debug_outputs/coevolve/v9/commercial_exact_axes_canon/{h}_v7/{h}_v7.spec.json"
        V, R = tokens(f"{h}_left_uniform_handonly_canon"), tokens(rep)
        d = np.mean([r[1] - v[1] for v, r in zip(V, R)], axis=0)          # canonical frame, m
        s0 = get_robot_spec(f"{h}_left_uniform_handonly_canon")
        kp = np.asarray(s0.palm_keypoints)
        e = kp[1:] - kp[:1]
        A = (e / np.linalg.norm(e, axis=1)[:, None]).T
        delta_body = -A @ d
        for tag in ("canon", "canonclip"):
            sp = D / h / f"{h}_left_{tag}.spec.json"
            s = json.loads(sp.read_text())
            F = np.asarray(s["palm_frame"], float)
            s["palm_centre_in_palm_body"] = (np.asarray(s["palm_centre_in_palm_body"]) + delta_body).tolist()
            s["palm_centre_ours"] = (np.asarray(s["palm_centre_ours"]) + F.T @ delta_body).tolist()
            s["notes_centred"] = f"palm centre moved {np.round(d * 1000, 1).tolist()} mm (canonical n, w, l) to the grammar's"
            (D / h / f"{h}_left_{tag}_centred.spec.json").write_text(json.dumps(s, indent=1))
        print(f"{h:8s} centre moved {np.round(d * 1000, 1)} mm (n, w, l); mean token origin error before "
              f"{np.mean([np.linalg.norm(v[1] - r[1]) for v, r in zip(V, R)]) * 1000:.0f} mm")
    if len(sys.argv) > 1 and sys.argv[1] == "--check":     # a fresh process: the new specs register at import
        for h in HANDS:
            R = tokens(f"debug_outputs/coevolve/v9/commercial_exact_axes_canon/{h}_v7/{h}_v7.spec.json")
            C = tokens(f"{h}_left_uniform_handonly_canon_centred")
            print(f"{h:8s} mean token origin error after {np.mean([np.linalg.norm(c[1] - r[1]) for c, r in zip(C, R)]) * 1000:.0f} mm")


if __name__ == "__main__":
    main()
