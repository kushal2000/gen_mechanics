"""Kit-free: where the training population's fingers point, in the canonical palm frame (normal, width, length) at
q = 0, against the vendor hand. Per hand: each finger's direction (base joint -> fingertip pad, projected into the
palm plane) as an angle about the palm normal, the largest angular gap between neighbouring fingers (360 - span),
finger length, and how far the base joints sit from the palm centre."""
import json, sys, random
import numpy as np
import xml.etree.ElementTree as ET
sys.path.insert(0, ".")
from hand_sampler.design_space import rpy_to_mat
from isaacsimenvs.common.robots import get_robot_spec
from isaacsimenvs.common.robots.finger_layout import finger_layout

def stats(name):
    s = get_robot_spec(name)
    root = ET.parse(s.urdf_path).getroot()
    up = {}
    for j in root.findall("joint"):
        o = j.find("origin"); T = np.eye(4)
        if o is not None:
            T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            T[:3, :3] = rpy_to_mat([float(v) for v in o.get("rpy", "0 0 0").split()])
        up[j.find("child").get("link")] = (j.find("parent").get("link"), T, j.get("name"))
    def pose(link):
        M = np.eye(4)
        while link != s.palm_body_name:
            p, T, _ = up[link]; M = T @ M; link = p
        return M
    kp = np.asarray(s.palm_keypoints); e = kp[1:] - kp[:1]; ax = (e / np.linalg.norm(e, axis=1)[:, None]).T
    c = np.asarray(s.palm_center_offset)
    lay = finger_layout(s)
    child_of = {v[2]: k for k, v in up.items()}
    angs, lens, base_r = [], [], []
    for f, tip in enumerate(s.fingertip_body_names):
        Mt = pose(tip); pad = Mt[:3, :3] @ np.asarray(s.fingertip_offsets[f]) + Mt[:3, 3]
        j0 = s.hand_joint_names[lay.fingers[f][0]]
        base = pose(child_of[j0])[:3, 3]
        d = ax.T @ (pad - base); b = ax.T @ (base - c)
        angs.append(np.degrees(np.arctan2(d[1], d[2]))); lens.append(np.linalg.norm(pad - base)); base_r.append(np.linalg.norm(b[1:]))
    a = np.sort(np.mod(angs, 360)); gaps = np.diff(np.concatenate([a, a[:1] + 360]))
    ext = np.linalg.norm(e, axis=1)
    return dict(n=len(angs), max_gap=float(gaps.max()), len_mm=1000 * np.mean(lens), base_r_mm=1000 * np.mean(base_r),
                palm_wl_mm=(1000 * ext[1], 1000 * ext[2]))

m = json.load(open(sys.argv[1]))
random.seed(0)
hands = random.sample(m["hands"], min(400, len(m["hands"])))
rows = [stats(h["spec"]) for h in hands]
gap = np.array([r["max_gap"] for r in rows]); L = np.array([r["len_mm"] for r in rows])
br = np.array([r["base_r_mm"] for r in rows]); pw = np.array([r["palm_wl_mm"] for r in rows])
print(f"training population ({len(rows)} of {len(m['hands'])}):")
print(f"  largest angular gap between fingers (deg): p5 {np.percentile(gap,5):.0f} median {np.median(gap):.0f} p95 {np.percentile(gap,95):.0f}")
print(f"  hands whose fingers all lie within a 120 deg sector (gap >= 240): {np.mean(gap >= 240):.3f}")
print(f"  finger length mm: p5 {np.percentile(L,5):.0f} median {np.median(L):.0f} p95 {np.percentile(L,95):.0f}")
print(f"  finger base distance from palm centre mm: median {np.median(br):.0f}")
print(f"  palm width x length mm: median {np.median(pw[:,0]):.0f} x {np.median(pw[:,1]):.0f}, min {pw[:,0].min():.0f} x {pw[:,1].min():.0f}")
for v in sys.argv[2:]:
    r = stats(v)
    print(f"{v}: max gap {r['max_gap']:.0f} deg, finger length {r['len_mm']:.0f} mm, base r {r['base_r_mm']:.0f} mm, palm {r['palm_wl_mm'][0]:.0f} x {r['palm_wl_mm'][1]:.0f} mm")
