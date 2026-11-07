"""Each commercial hand beside its v3 replica (snap_commercial.py), at q = 0 in the canonical palm frame: vendor face /
side views, then the replica's; joints coloured by kind (abduction = axis along the palm normal) on both.

    .venv_isaacsim/bin/python experiments/07nov_v3/plot_commercial_vs_v3.py [--flex-deg 60]
Writes debug_outputs/coevolve/v3/commercial_vs_v3.png (q = 0), or commercial_vs_v3_flex<deg>.png with every
flexion joint at --flex-deg (abductions at 0) -- where a finger's roll shows: a rolled thumb curls across the palm.
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments/06nov_coevolve_v2")]


def vendor_kinds(name):
    """Each vendor joint's kind, roll-invariantly (descriptors_v3.joint_kinds: flexion parallel to the finger's last
    joint, roll along the finger -- drawn as flexion -- abduction otherwise)."""
    from hand_sampler.design_space import joint_axes_and_link_vectors
    from isaacsimenvs.common.robots import get_robot_spec
    from isaacsimenvs.common.robots.finger_layout import finger_layout
    from coevolution.evolution.descriptors_v3 import _frames, joint_kinds
    s = get_robot_spec(name)
    pose, child_of = _frames(s)
    ax, _ = joint_axes_and_link_vectors(s.urdf_path, s.hand_joint_names, s.fingertip_body_names, s.fingertip_offsets)
    out = {}
    for f, (joints, tip) in enumerate(zip(finger_layout(s).fingers, s.fingertip_body_names)):
        Mt = pose(tip)
        pad = Mt[:3, :3] @ np.asarray(s.fingertip_offsets[f]) + Mt[:3, 3]
        d = pad - pose(child_of[s.hand_joint_names[joints[0]]])[:3, 3]
        for j, k in zip(joints, joint_kinds(s, pose, child_of, ax, joints, d)):
            out[s.hand_joint_names[j]] = "abd" if k == "A" else "flex"
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--flex-deg", type=float, default=0.0); a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    from plot_morph_gallery import canon, KIND_RGB, PALM_RGB
    from render_rollout import posed_tris
    from coevolution.evolution.descriptors_v3 import features, cell
    man = json.loads((REPO / "debug_outputs/coevolve/v3/commercial_snapped/manifest.json").read_text())["hands"]
    fig, axs = plt.subplots(len(man), 4, figsize=(4 * 2.6, len(man) * 2.7), squeeze=False, facecolor="white")
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    for r, h in enumerate(man):
        rows = []
        vk = vendor_kinds(h["vendor"])
        rows.append((h["vendor"], lambda jn, vk=vk: vk.get(jn, "flex"), f"vendor {h['vendor'].split('_')[0]}", h["vendor_cell"]))
        spec = json.loads((REPO / h["spec"]).read_text())
        kinds = {f"f{i}_j{j}": ("flex" if (int(jt["axis"][0]) - int(f.get("roll", 0))) % 180 == 0 else "abd")
                 for i, f in enumerate(spec["design"]["fingers"]) for j, jt in enumerate(f["joints"])}
        rows.append((h["spec"], lambda jn, k=kinds: k.get(jn, "flex"), "v3 replica" + ("" if h["valid"] else " (INVALID: collides)"), h["niche"]))
        for k, (src, kind_of, title, cl) in enumerate(rows):
            rs, ctr, R = canon(src)
            import math as _m
            q = {jn: (_m.radians(a.flex_deg) if kind_of(jn) == "flex" else 0.0) for jn in rs.hand_joint_names}
            tris, cols = posed_tris(REPO / rs.urdf_path, q, kind_of)
            P = (tris - ctr) @ R.T * 1000
            for v, (ia, ib, idp) in enumerate([(1, 2, 0), (2, 0, 1)]):
                ax = axs[r, 2 * k + v]; ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_color("#e4e3df")
                xy = P[..., [ia, ib]]; depth = P[..., idp].mean(1)
                nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
                o = np.argsort(depth); sh = (0.35 + 0.65 * np.abs(nrm @ light))[o, None]
                ax.add_collection(PolyCollection(xy[o], facecolors=np.clip(cols[o] * sh, 0, 1), edgecolors="none"))
                ax.plot([0], [0], "k+", ms=5)
                ax.set_xlim(-170, 170); ax.set_ylim(-170, 170); ax.set_aspect("equal")
                if v == 0:
                    ax.set_title(f"{title}\n{cl}", fontsize=7, color="#b42318" if "INVALID" in title else "#1f1f1e")
                else:
                    ax.set_title("side", fontsize=7, color="#8a8985")
    fig.legend(handles=[Patch(color=KIND_RGB["flex"], label="flexion"), Patch(color=KIND_RGB["abd"], label="abduction"),
                        Patch(color=PALM_RGB, label="palm")], loc="upper right", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Commercial hands (vendor URDF) beside their v3 replicas, flexion joints at {a.flex_deg:g} deg, "
                 f"canonical palm frame (mm)", x=0.01,
                 ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = REPO / ("debug_outputs/coevolve/v3/commercial_vs_v3.png" if not a.flex_deg
                  else f"debug_outputs/coevolve/v3/commercial_vs_v3_flex{a.flex_deg:g}.png"); fig.savefig(out, dpi=110, facecolor="white"); print(out)


if __name__ == "__main__":
    main()
