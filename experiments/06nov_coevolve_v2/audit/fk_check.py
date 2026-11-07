"""Sim vs URDF: every link's position in the palm body frame as the sim reports it (scene_dump probe) against a
Kit-free forward kinematics of the URDF at the sim's own joint angles. A converter sign flip or frame change shows
up as a mismatch. Also: per probed slot, the joint's motion direction (a x link) in the canonical palm frame."""
import json, sys
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0, ".")
from hand_sampler.design_space import rpy_to_mat
from isaacsimenvs.common.robots import get_robot_spec

npz, specname = sys.argv[1], sys.argv[2]
P = np.load(npz)
spec = get_robot_spec(specname)
root = ET.parse(spec.urdf_path).getroot()
J = []
for j in root.findall("joint"):
    o = j.find("origin"); T = np.eye(4)
    if o is not None:
        T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
        T[:3, :3] = rpy_to_mat([float(v) for v in o.get("rpy", "0 0 0").split()])
    ax = j.find("axis")
    J.append((j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"), T,
              np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])))

def rot(axis, q):
    a = axis / np.linalg.norm(axis); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K

def fk(qmap):
    out = {spec.palm_body_name: np.eye(4)}; changed = True
    while changed:
        changed = False
        for n, t, p, c, T, ax in J:
            if p in out and c not in out:
                M = out[p] @ T
                if t in ("revolute", "continuous"):
                    R = np.eye(4); R[:3, :3] = rot(ax, qmap.get(n, 0.0)); M = M @ R
                out[c] = M; changed = True
    return out

slot_joint = {int(k): v for k, v in json.loads(str(P["slot_joint"])).items()}
names = list(P["link_names"])
def compare(q_canon, links, tag):
    qmap = {slot_joint[s]: float(q_canon[6 + s]) for s in slot_joint}
    F = fk(qmap)
    errs = [np.linalg.norm(F[l][:3, 3] - links[i]) for i, l in enumerate(names) if l in F]
    return max(errs) * 1000
print("zero pose: max |sim - URDF FK| mm", round(compare(np.zeros(37), P["links0"], "zero"), 4))
worst = max(compare(P["q_end"][e], P["links_end"][e], e) for e in range(P["q_end"].shape[0]))
print("per-slot probe envs (33): worst |sim - URDF FK| mm", round(worst, 4))
worst = max(compare(P["traj_q"][t], P["traj_links"][t], t) for t in range(P["traj_q"].shape[0]))
print("sinusoid trajectory: worst |sim - URDF FK| mm", round(worst, 4))
# per probed slot: the commanded target and the reached angle
lo, hi = P["joint_lower"], P["joint_upper"]
print("slot joint                       lower   upper   target(+0.6)  reached")
for s in sorted(slot_joint):
    print(f"{s:3d}  {slot_joint[s]:26s} {np.degrees(lo[6+s]):7.1f} {np.degrees(hi[6+s]):7.1f} "
          f"{np.degrees(0.6*hi[6+s]):9.1f} {np.degrees(P['q_end'][1+s, 6+s]):9.1f}")
