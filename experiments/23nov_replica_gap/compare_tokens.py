"""Joint tokens at q = 0 as the olympics env computes them (joint_origin, joint_axis, joint_link_vec in the canonical
palm frame at the palm centre; observations.py), for a vendor hand and its exact replica, joint by joint (23 Nov 2026).

    .venv_isaacsim/bin/python experiments/23nov_replica_gap/compare_tokens.py leap
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def tokens(name: str):
    import yourdfpy
    from hand_sampler.design_space import joint_axes_and_link_vectors
    from isaacsimenvs.common.robots import get_robot_spec
    s = get_robot_spec(name)
    u = yourdfpy.URDF.load(s.urdf_path, load_meshes=False, build_collision_scene_graph=False)
    Tp = u.get_transform(s.palm_body_name, u.base_link)
    kp = np.asarray(s.palm_keypoints)
    e = kp[1:] - kp[:1]
    A = (e / np.linalg.norm(e, axis=1)[:, None]).T                 # palm body -> canonical columns (n, w, l)
    c = np.asarray(s.palm_center_offset)
    ax, lv = joint_axes_and_link_vectors(s.urdf_path, s.hand_joint_names, s.fingertip_body_names, s.fingertip_offsets)
    out = []
    for j, body in enumerate(s.joint_link_bodies):
        T = np.linalg.inv(Tp) @ u.get_transform(body, u.base_link)   # body in the palm body frame
        R, p = T[:3, :3], T[:3, 3]
        out.append((s.hand_joint_names[j], A.T @ (p - c), A.T @ (R @ ax[j]), A.T @ (R @ lv[j])))
    return out


def main() -> None:
    h = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "canonclip"
    rep = f"debug_outputs/coevolve/v9/commercial_exact_axes_canon/{h}_v7/{h}_v7.spec.json"
    V, R = tokens(f"{h}_left_uniform_handonly_{tag}"), tokens(rep)
    print(f"{h}: joint | origin mm vendor / replica | axis vendor / replica | link vec mm vendor / replica")
    for (n, ov, av, lvv), (_, orr, ar, lr) in zip(V, R):
        f = lambda x, k=1000: np.array2string(np.round(x * k).astype(int), separator=",")
        g = lambda x: np.array2string(np.round(x, 2), separator=",")
        print(f"  {n:28s} {f(ov)} / {f(orr)}  d {np.linalg.norm(ov - orr) * 1000:4.0f} | {g(av)} / {g(ar)} | {f(lvv)} / {f(lr)}")


if __name__ == "__main__":
    main()
