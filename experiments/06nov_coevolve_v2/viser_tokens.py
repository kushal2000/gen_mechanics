"""Kinematic comparison of hands as the policy sees them: each hand in its canonical palm frame (the frame its joint
tokens are expressed in -- origin at the palm centre, x the palm normal toward the camera, z toward the fingers),
side by side, with a slider per joint and every joint token drawn on the hand. No physics.

    .venv_isaacsim/bin/python experiments/06nov_coevolve_v2/viser_tokens.py --port 8101 \
        --hands <spec>[=label] <spec>[=label] ...      (a registered robot name or a grammar .spec.json)

Per joint token (observations.py): joint_origin (sphere), joint_axis (short line through it), joint_link_vec
(line to the next joint, or to the fingertip for a finger's last joint), coloured by the token's finger slot and
labelled f<finger slot>.j<joint in the finger>; fingertips are white spheres. A table per hand lists the values in
the canonical palm frame (mm), as the observation would read them at the sliders' pose. "all joints" sets every
joint to the same fraction of its range (-1 = lower limit, 0 = zero pose, +1 = upper limit): the quick way to see
which way each hand's joints turn for positive angles.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
SLOT_RGB = [(0.16, 0.42, 0.85), (0.86, 0.20, 0.18), (0.15, 0.65, 0.30), (0.85, 0.55, 0.10), (0.55, 0.25, 0.75),
            (0.10, 0.65, 0.70)]
# canonical palm frame -> world: normal x toward the viewer (-y), width y along +x, fingers z up
W = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def rot_to_wxyz(R: np.ndarray) -> tuple:
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return (w, x, y, z)


class Hand:
    def __init__(self, server, ref: str, label: str, offset: np.ndarray):
        import yourdfpy
        from viser.extras import ViserUrdf
        from hand_sampler import resolve
        from isaacsimenvs.common.robots import REGISTRY, _spec_from_file
        from isaacsimenvs.common.robots.finger_layout import finger_layout
        from isaacsimenvs.inhand_olympics.wrist import palm_frame

        self.server, self.label = server, label
        self.spec = spec = REGISTRY[ref] if ref in REGISTRY else _spec_from_file(ref)
        self.urdf = yourdfpy.URDF.load(str(resolve(spec.urdf_path)), load_meshes=True,
                                       build_collision_scene_graph=False)
        self.names = list(spec.hand_joint_names)
        self.lay = finger_layout(spec)
        self.slot = list(self.lay.slot_of_joint)
        self.jmap = {j.name: j for j in self.urdf.robot.joints}
        self.q = {n: 0.0 for n in self.urdf.actuated_joint_names}
        self.urdf.update_cfg(np.zeros(len(self.urdf.actuated_joint_names)))
        T_pb = self.urdf.get_transform(spec.palm_body_name)
        self.centre = (T_pb @ np.r_[np.asarray(spec.palm_center_offset, float), 1.0])[:3]
        self.Rc = T_pb[:3, :3] @ palm_frame(spec)                       # canonical palm frame in the root frame
        R = W @ self.Rc.T
        self.T_world_root = np.eye(4)
        self.T_world_root[:3, :3], self.T_world_root[:3, 3] = R, offset - R @ self.centre
        server.scene.add_frame(f"/{label}", wxyz=rot_to_wxyz(R), position=tuple(self.T_world_root[:3, 3]),
                               show_axes=False)
        server.scene.add_frame(f"/{label}_palm", wxyz=(1, 0, 0, 0), position=tuple(offset), axes_length=0.04,
                               axes_radius=0.002)
        self.vis = ViserUrdf(server, self.urdf, root_node_name=f"/{label}", mesh_color_override=(0.75, 0.75, 0.78, 0.55))
        self.tips = list(spec.fingertip_body_names)
        self.tip_off = [np.asarray(o, float) for o in spec.fingertip_offsets] if spec.fingertip_offsets else \
            [np.zeros(3)] * len(self.tips)
        self.labels_on = True
        self.table = None

    def finger_of_joint(self, i: int) -> tuple[int, int]:
        for f, js in enumerate(self.lay.fingers):
            if i in js:
                return f, js.index(i)
        return -1, -1

    def tokens(self):
        """Per hand joint (spec order): slot, finger, k, origin / axis / link vec in root and canonical frames."""
        cfg = np.array([self.q[n] for n in self.urdf.actuated_joint_names])
        self.urdf.update_cfg(cfg)
        self.vis.update_cfg(cfg)
        rows = []
        for i, n in enumerate(self.names):
            j = self.jmap[n]
            Tj = self.urdf.get_transform(j.parent) @ j.origin
            p, a = Tj[:3, 3], Tj[:3, :3] @ np.asarray(j.axis, float)
            f, k = self.finger_of_joint(i)
            js = self.lay.fingers[f] if f >= 0 else []
            if f >= 0 and k + 1 < len(js):
                jn = self.jmap[self.names[js[k + 1]]]
                nxt = (self.urdf.get_transform(jn.parent) @ jn.origin)[:3, 3]
            elif f >= 0 and f < len(self.tips):
                nxt = (self.urdf.get_transform(self.tips[f]) @ np.r_[self.tip_off[f], 1.0])[:3]
            else:
                nxt = p
            rows.append({"i": i, "name": n, "slot": self.slot[i], "finger": f, "k": k, "p": p, "a": a, "v": nxt - p,
                         "lim": (j.limit.lower, j.limit.upper)})
        return rows

    def draw(self):
        rows = self.tokens()
        M = self.T_world_root
        segs, cols = [], []
        for r in rows:
            fs = r["slot"] // 5 if r["slot"] < 30 else 5
            c = SLOT_RGB[fs % len(SLOT_RGB)]
            pw = (M @ np.r_[r["p"], 1.0])[:3]
            aw, vw = M[:3, :3] @ r["a"], M[:3, :3] @ r["v"]
            self.server.scene.add_icosphere(f"/tok/{self.label}/o{r['i']}", radius=0.004, color=c, position=tuple(pw))
            segs += [[pw - 0.02 * aw, pw + 0.02 * aw], [pw, pw + vw]]
            dark = tuple(0.5 * x for x in c)
            cols += [[dark, dark], [c, c]]
            if self.labels_on:
                self.server.scene.add_label(f"/tok/{self.label}/l{r['i']}", f"f{fs}.j{r['slot'] % 5}",
                                            position=tuple(pw + 0.006 * aw))
            else:
                try:
                    self.server.scene.remove_by_name(f"/tok/{self.label}/l{r['i']}")
                except Exception:
                    pass
        self.server.scene.add_line_segments(f"/tok/{self.label}/lines", points=np.asarray(segs, np.float32),
                                            colors=np.asarray(cols, np.float32), line_width=3.0)
        for t, (tip, off) in enumerate(zip(self.tips, self.tip_off)):
            tw = (M @ self.urdf.get_transform(tip) @ np.r_[off, 1.0])[:3]
            self.server.scene.add_icosphere(f"/tok/{self.label}/tip{t}", radius=0.005, color=(1, 1, 1),
                                            position=tuple(tw))
        if self.table is not None:
            Rc, c0 = self.Rc, self.centre
            lines = ["| token | joint | origin (mm) | axis | link vec (mm) | limits (deg) |", "|---|---|---|---|---|---|"]
            for r in sorted(rows, key=lambda r: r["slot"]):
                po = Rc.T @ (r["p"] - c0) * 1000
                ao, vo = Rc.T @ r["a"], Rc.T @ r["v"] * 1000
                lo, hi = (np.degrees(r["lim"][0]), np.degrees(r["lim"][1])) if r["lim"][0] is not None else (0, 0)
                lines.append(f"| f{r['slot'] // 5}.j{r['slot'] % 5} | {r['name']} | {po[0]:.0f}, {po[1]:.0f}, {po[2]:.0f} | "
                             f"{ao[0]:+.2f}, {ao[1]:+.2f}, {ao[2]:+.2f} | {vo[0]:.0f}, {vo[1]:.0f}, {vo[2]:.0f} | "
                             f"{lo:.0f} / {hi:.0f} |")
            self.table.content = "\n".join(lines)


def main() -> None:
    import viser
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--hands", nargs="+", required=True)
    ap.add_argument("--spacing", type=float, default=0.40)
    args = ap.parse_args()
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=2.0, height=2.0, plane="xy", position=(0, 0, -0.25))
    hands = []
    for h, item in enumerate(args.hands):
        ref, _, label = item.partition("=")
        label = (label or Path(ref).stem.replace(".spec", "")).replace("/", "_")
        hands.append(Hand(server, ref, label, np.array([h * args.spacing, 0.0, 0.0])))
    show = server.gui.add_checkbox("token labels", True)
    allj = server.gui.add_slider("all joints (fraction of range)", -1.0, 1.0, 0.05, 0.0)
    sliders = []
    for hand in hands:
        with server.gui.add_folder(hand.label):
            for i, n in enumerate(hand.names):
                j = hand.jmap[n]
                lo, hi = float(j.limit.lower), float(j.limit.upper)
                f, k = hand.finger_of_joint(i)
                s = server.gui.add_slider(f"f{hand.slot[i] // 5}.j{hand.slot[i] % 5} {n}", lo, hi, 0.01, 0.0)
                sliders.append((hand, n, s, lo, hi))

                def cb(_, hand=hand, n=n, s=s):
                    hand.q[n] = float(s.value)
                    hand.draw()
                s.on_update(cb)
            hand.table = server.gui.add_markdown("")
        hand.draw()

    @allj.on_update
    def _(_):
        t = float(allj.value)
        for hand, n, s, lo, hi in sliders:
            v = t * hi if t >= 0 else -t * lo
            s.value = float(np.clip(v, lo, hi))

    @show.on_update
    def _(_):
        for hand in hands:
            hand.labels_on = bool(show.value)
            hand.draw()
    print(f"[viser_tokens] {len(hands)} hands; open http://{__import__('socket').getfqdn()}:{args.port}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
