"""Any two hands at the in-hand reorientation SPAWN state, in viser: pose them, and compare what the policy sees.

Reads the JSONs capture_spawn.py writes (results/spawn/<name>.json; no simulator here).

  * Hand A and hand B: any two captures (vendor vs its replica, or any other pair), side by side or overlaid,
    each with its cube, goal (green ghost) and canonical palm frame. Each hand is drawn in ITS OWN canonical palm
    frame -- palm centre at the origin, palm axes along the world axes -- so overlaid, the two palm centres and
    frames coincide and every difference on screen is one the policy sees (its observations are in that frame).
  * Each finger's base (the first joint, the MCP flexion on Wuji v2) is marked, and its gap listed in mm.
  * Every joint's hinge is drawn as a short line through its origin along its axis (positive end marked), and
    follows the sliders -- so a joint turning about a different line, or the other way, is visible.
  * Joint sliders for each hand, starting at the captured spawn angles. "B follows A" drives B's matching joint
    (through the replica's joint map) to the same angle as A's.
  * The policy observation, term by term, for the terms you tick: per-term and combined differences over the
    entries that pair up (same label, or vendor joint -> replica slot through the joint map). As the sliders move,
    joint_pos and joint_link_bbox are RECOMPUTED (forward kinematics, the env's own formulas -- checked against the
    capture at the spawn pose); every other term stays at its spawn value (velocities, previous-step terms and
    targets have no meaning for a slider pose, and nothing else depends on the fingers).

    .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/viser_spawn.py --port 8084
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPAWN = HERE / "results/spawn"
COMMERCIAL = REPO / "assets/urdf/unified_dynamics_commercial_hands"
REPLICAS = REPO / "assets/urdf/unified_dynamics_commercial_hands_replicas"
STEM = {"wuji2": "wuji2_left", "leap": "leap_left", "allegro": "allegro_left", "midas": "midas"}
LIVE = ("joint_pos", "joint_link_bbox")
TOKEN_TERMS = {"joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets",
               "joint_link_bbox", "joint_lower", "joint_upper"}
COLORS = {"A": (0.85, 0.55, 0.2), "B": (0.2, 0.45, 0.85)}


def parse(name: str) -> tuple[str, str, str]:
    """<hand>_commercial, <hand>_replica or <hand>_replica_<fidelity level> -> (hand, kind, level)."""
    for hand in STEM:
        if name.startswith(hand + "_"):
            kind, _, level = name[len(hand) + 1:].partition("_")
            return hand, kind, level
    return "", "", ""


def urdf_of(name: str) -> pathlib.Path | None:
    hand, kind, level = parse(name)
    if not hand:
        return None
    p = (COMMERCIAL / hand / f"{STEM[hand]}.urdf" if kind == "commercial"
         else REPLICAS / hand / f"{STEM[hand]}_replica{'_' + level if level else ''}.urdf")
    return p if p.exists() else None


def load(name: str) -> dict:
    return json.loads((SPAWN / f"{name}.json").read_text())


def label_map(a_name: str, b_name: str) -> dict[str, str]:
    """A's joint names -> B's, when one is the other's replica (make_replicas.py's joint map); else {}."""
    ha, ka, la = parse(a_name)
    hb, kb, lb = parse(b_name)
    if ha != hb or not ha or {ka, kb} != {"commercial", "replica"}:
        return {}
    level = la or lb
    p = REPLICAS / ha / f"{STEM[ha]}_replica{'_' + level if level else ''}.joint_map.json"
    if not p.exists():
        return {}
    m = json.loads(p.read_text())                       # vendor -> slot
    return m if ka == "commercial" else {v: k for k, v in m.items()}


class Hand:
    """One capture: its spawn state, forward kinematics, and the live joint-dependent observation terms."""

    def __init__(self, name: str):
        import yourdfpy
        self.name, self.d = name, load(name)
        d = self.d
        self.urdf_path = urdf_of(name)
        self.fk = yourdfpy.URDF.load(str(self.urdf_path), load_meshes=False) if self.urdf_path else None
        x, y, z, qw, qx, qy, qz = d["root"]
        self.T_root = np.eye(4)
        self.T_root[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        self.T_root[:3, 3] = [x, y, z]
        w, cx, cy, cz = d["palm_frame_canonical_wxyz"]
        self.R_canon = Rot.from_quat([cx, cy, cz, w]).as_matrix()
        self.c = np.array(d["palm_center"])
        self.limits = dict(zip(d["joint_names"], (tuple(l) for l in d["joint_limits"])))
        self.spawn_q = dict(zip(d["joint_names"], d["joint_pos"]))
        self.q = dict(self.spawn_q)
        # movable = a real range (a generated hand's ghost slots carry (0, 1e-8))
        self.movable = [j for j in d["hand_joint_names"] if self.limits[j][1] - self.limits[j][0] > 1e-6]

    def body_T(self) -> dict[str, np.ndarray]:
        self.fk.update_cfg({j: self.q.get(j, 0.0) for j in self.fk.actuated_joint_names})
        return {b: self.T_root @ self.fk.get_transform(b, self.fk.base_link) for b in self.fk.link_map}

    def terms(self) -> dict:
        """d["terms"] with the LIVE terms recomputed at the current joint angles."""
        out = {k: dict(v) for k, v in self.d["terms"].items()}
        d = self.d
        if "joint_pos" in out:                     # range units: 2 (q - lo) / (hi - lo) - 1, 0 on a ghost slot
            vals = []
            for j in d["canonical_joint_names"]:
                lo, hi = self.limits[j]
                vals.append(2.0 * (self.q[j] - lo) / (hi - lo) - 1.0 if hi - lo > 1e-6 else 0.0)
            out["joint_pos"]["values"] = vals
        if "joint_link_bbox" in out and self.fk is not None and "bbox_local" in d:
            T = self.body_T()
            vals = []
            for body, corners, ok in zip(d["bbox_bodies"], d["bbox_local"], d["bbox_valid"]):
                c = np.asarray(corners)
                if not ok or body not in T:
                    vals += [0.0] * c.size
                    continue
                w = (T[body][:3, :3] @ c.T).T + T[body][:3, 3]
                vals += ((w - self.c) @ self.R_canon).ravel().tolist()
            out["joint_link_bbox"]["values"] = vals
        return out


def pair(ta: dict, tb: dict, lmap: dict[str, str]):
    """(A label, A value, B label, B value) for the entries that pair up, or None."""
    if ta["labels"] == tb["labels"]:
        return [(l, x, l, y) for l, x, y in zip(ta["labels"], ta["values"], tb["values"])]
    if not lmap:
        return None
    vb = dict(zip(tb["labels"], tb["values"]))
    out = []
    for lab, x in zip(ta["labels"], ta["values"]):
        base, _, rest = lab.partition("[")
        lb = lmap.get(base)
        lb = None if lb is None else lb + ("[" + rest if rest else "")
        if lb is None or lb not in vb:
            return None
        out.append((lab, x, lb, vb[lb]))
    return out


def compare(TA: dict, TB: dict, selected: list[str], lmap) -> str:
    rows = ["| term | dim A / B | max abs | mean abs | B-only max abs | |", "|---|---|---|---|---|---|"]
    allv = []
    for f in selected:
        ta, tb = TA.get(f), TB.get(f)
        live = "live" if f in LIVE else ""
        if ta is None or tb is None:
            rows.append(f"| `{f}` | {'-' if ta is None else len(ta['values'])} / "
                        f"{'-' if tb is None else len(tb['values'])} | | | | missing |")
            continue
        p = pair(ta, tb, lmap)
        if p is None:
            rows.append(f"| `{f}` | {len(ta['values'])} / {len(tb['values'])} | - | - | | not pairable |")
            continue
        dlt = np.array([y - x for _, x, _, y in p])
        allv.append(dlt)
        used = {l for _, _, l, _ in p}
        extra = [abs(v) for l, v in zip(tb["labels"], tb["values"]) if l not in used]
        rows.append(f"| `{f}` | {len(ta['values'])} / {len(tb['values'])} | {np.abs(dlt).max():.3f} | "
                    f"{np.abs(dlt).mean():.3f} | {f'{max(extra):.3f}' if extra else ''} | {live} |")
    if allv:
        v = np.concatenate(allv)
        head = (f"**Selected terms combined** ({v.size} paired entries): L2 **{np.linalg.norm(v):.3f}**, "
                f"max abs **{np.abs(v).max():.3f}**, mean abs **{np.abs(v).mean():.3f}**\n\n")
    else:
        head = "_no pairable term selected_\n\n"
    return head + "\n".join(rows)


def detail(TA: dict, TB: dict, f: str, lmap) -> str:
    if f not in TA or f not in TB:
        return "_not in both captures_"
    ta, tb = TA[f], TB[f]
    p = pair(ta, tb, lmap)
    fmt = lambda v: f"{v:+.3f}"
    if p is None:
        n = max(len(ta["labels"]), len(tb["labels"]))
        rows = ["| A entry | A | B entry | B |", "|---|---|---|---|"]
        for i in range(n):
            a = (ta["labels"][i], fmt(ta["values"][i])) if i < len(ta["labels"]) else ("", "")
            b = (tb["labels"][i], fmt(tb["values"][i])) if i < len(tb["labels"]) else ("", "")
            rows.append(f"| {a[0]} | {a[1]} | {b[0]} | {b[1]} |")
        return "\n".join(rows)
    rows = ["| A entry | B entry | A | B | B - A |", "|---|---|---|---|---|"]
    rows += [f"| {l1} | {l2} | {fmt(x)} | {fmt(y)} | {fmt(y - x)} |" for l1, x, l2, y in p]
    used = {l for _, _, l, _ in p}
    rows += [f"| | {l} (B only) | | {fmt(y)} | |" for l, y in zip(tb["labels"], tb["values"]) if l not in used]
    return "\n".join(rows)


def live_check(h: "Hand") -> float | None:
    """Max |live recompute - capture| at the SPAWN angles: the formulas must reproduce what the sim reported."""
    if "bbox_local" not in h.d:
        return None
    q0, h.q = h.q, dict(h.spawn_q)
    t = h.terms()
    h.q = q0
    return max(float(np.abs(np.array(t[f]["values"]) - np.array(h.d["terms"][f]["values"])).max())
               for f in LIVE if f in t)


def summary(h: "Hand", tag: str) -> str:
    d = h.d
    cube = (np.array(d["object"][:3]) - h.c) @ h.R_canon
    ext = (np.round(np.array(d["terms"]["palm_extents"]["values"]) * 1000).astype(int).tolist()
           if "palm_extents" in d["terms"] else "-")
    err = live_check(h)
    chk = "" if err is None else f"; live-term check at spawn: max err {err:.1e}"
    return (f"* **{tag}** `{h.name}`: obs {d['obs_dim']}, {len(h.movable)} movable joints; palm {ext} mm; "
            f"cube centre in palm frame {np.round(cube * 1000, 1).tolist()} mm{chk}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8084)
    args = ap.parse_args()

    import viser
    from viser.extras import ViserUrdf

    names = sorted(p.stem for p in SPAWN.glob("*.json"))
    if not names:
        raise SystemExit(f"no captures in {SPAWN}; run capture_spawn.py first")
    a0 = "wuji2_commercial" if "wuji2_commercial" in names else names[0]
    b0 = "wuji2_replica" if "wuji2_replica" in names else names[-1]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    st: dict = {"hands": {}, "robots": {}, "nodes": [], "sliders": {"A": {}, "B": {}}, "lmap": {}, "muted": False}

    with server.gui.add_folder("hands"):
        dd_a = server.gui.add_dropdown("A", tuple(names), initial_value=a0)
        dd_b = server.gui.add_dropdown("B", tuple(names), initial_value=b0)
        g_overlay = server.gui.add_checkbox("overlay (same place)", False)
        g_gap = server.gui.add_slider("side-by-side gap (m)", 0.0, 0.6, 0.01, 0.3)
        md_sum = server.gui.add_markdown("")
    terms0 = load(a0)["obs_list"]
    with server.gui.add_folder("observation terms to compare"):
        b_all = server.gui.add_button("all terms")
        b_tok = server.gui.add_button("joint tokens only")
        b_glob = server.gui.add_button("global token only")
        g_terms = {f: server.gui.add_checkbox(f + (" (live)" if f in LIVE else ""), True) for f in terms0}
    with server.gui.add_folder("difference (B - A)"):
        md_tab = server.gui.add_markdown("")
        dd_term = server.gui.add_dropdown("term detail", tuple(terms0), initial_value=terms0[0])
        md_term = server.gui.add_markdown("")
    with server.gui.add_folder("joints"):
        g_follow = server.gui.add_checkbox("B follows A (joint map)", True)
        g_axes = server.gui.add_checkbox("show joint axes", True)
        b_reset = server.gui.add_button("reset both to spawn")
        # Each hand draws its own reset noise, so at spawn matching joints sit a few degrees apart (Wuji v2's thumb
        # IP 12): this puts B at A's angles, joint for joint, to compare geometry alone.
        b_copy = server.gui.add_button("B takes A's angles")
        f_a = server.gui.add_folder("A joints")
        f_b = server.gui.add_folder("B joints")

    def selected():
        return [f for f, cb in g_terms.items() if cb.value]

    def refresh_obs(_=None):
        A, B = st["hands"].get("A"), st["hands"].get("B")
        if A is None or B is None:
            return
        TA, TB = A.terms(), B.terms()
        md_tab.content = compare(TA, TB, selected(), st["lmap"])
        md_term.content = detail(TA, TB, dd_term.value, st["lmap"])

    def pose(tag):
        h, r = st["hands"][tag], st["robots"].get(tag)
        if r is not None:
            r.update_cfg(np.array([h.q.get(j, 0.0) for j in r.get_actuated_joint_names()]))
        draw_axes(tag)

    def draw_axes(tag):
        """Each movable joint's hinge: a 30 mm line through its origin along its axis, with a dot at the + end."""
        h = st["hands"][tag]
        if h.fk is None or not g_axes.value:
            for k in ("axes", "axes_tip"):
                n = st.get(f"{k}_{tag}")
                if n is not None:
                    n.remove()
                    st[f"{k}_{tag}"] = None
            return
        T = h.body_T()
        segs, tips = [], []
        for j in h.fk.robot.joints:
            if j.type != "revolute" or j.name not in h.movable:
                continue
            Tc = T[j.child]
            u = Tc[:3, :3] @ np.asarray(j.axis, float)
            u = u / np.linalg.norm(u)
            segs.append([Tc[:3, 3] - 0.015 * u, Tc[:3, 3] + 0.015 * u])
            tips.append(Tc[:3, 3] + 0.015 * u)
        if not segs:
            return
        st[f"axes_{tag}"] = server.scene.add_line_segments(
            f"/{tag}/axes", points=np.array(segs, dtype=np.float32), colors=COLORS[tag], line_width=3.0)
        st[f"axes_tip_{tag}"] = server.scene.add_point_cloud(
            f"/{tag}/axes_tip", points=np.array(tips, dtype=np.float32),
            colors=np.tile(np.array(COLORS[tag], dtype=np.float32), (len(tips), 1)), point_size=0.004)

    def on_slider(tag, joint, s):
        def cb(_):
            if st["muted"]:
                return
            st["hands"][tag].q[joint] = s.value
            pose(tag)
            if tag == "A" and g_follow.value and joint in st["lmap"]:
                jb, hb = st["lmap"][joint], st["hands"]["B"]
                if jb in st["sliders"]["B"]:
                    v = float(np.clip(s.value, *hb.limits[jb]))
                    st["muted"] = True
                    st["sliders"]["B"][jb].value = v
                    st["muted"] = False
                    hb.q[jb] = v
                    pose("B")
            refresh_obs()
        return cb

    def build_sliders(tag, folder):
        for s in st["sliders"][tag].values():
            s.remove()
        st["sliders"][tag] = {}
        h = st["hands"][tag]
        back = {v: k for k, v in st["lmap"].items()}
        with folder:
            for j in h.movable:
                lo, hi = h.limits[j]
                label = j if tag == "A" or j not in back else f"{j} = {back[j]}"
                s = server.gui.add_slider(label, float(lo), float(hi), 0.01, float(np.clip(h.q[j], lo, hi)))
                s.on_update(on_slider(tag, j, s))
                st["sliders"][tag][j] = s

    def draw(tag, offset):
        h = st["hands"][tag]
        d = h.d
        root = f"/{tag}"
        # world = offset + R_canon^T (p - palm centre): the hand in its own canonical palm frame
        Rr = h.R_canon.T
        q = Rot.from_matrix(Rr).as_quat()
        st["nodes"].append(server.scene.add_frame(root, position=tuple(np.asarray(offset) - Rr @ h.c),
                                                  wxyz=(q[3], q[0], q[1], q[2]), show_axes=False))
        x, y, z, qw, qx, qy, qz = d["root"]
        st["nodes"].append(server.scene.add_frame(f"{root}/base", position=(x, y, z), wxyz=(qw, qx, qy, qz),
                                                  show_axes=False))
        if h.urdf_path is not None:
            st["robots"][tag] = ViserUrdf(server, h.urdf_path, root_node_name=f"{root}/base/hand",
                                          mesh_color_override=COLORS[tag] + (0.85,))
            pose(tag)
        st["nodes"].append(server.scene.add_frame(f"{root}/palm_frame", position=tuple(h.c),
                                                  wxyz=tuple(d["palm_frame_canonical_wxyz"]),
                                                  axes_length=0.05, axes_radius=0.002))
        o, g = d["object"], d["goal"]
        size = tuple(d.get("object_size", (0.05, 0.05, 0.05)))
        st["nodes"].append(server.scene.add_box(f"{root}/object", dimensions=size, position=tuple(o[:3]),
                                                wxyz=tuple(o[3:]), color=COLORS[tag], opacity=0.6))
        st["nodes"].append(server.scene.add_box(f"{root}/goal", dimensions=size, position=tuple(g[:3]),
                                                wxyz=tuple(g[3:]), color=(0.3, 0.8, 0.3), opacity=0.25))
        st["nodes"].append(server.scene.add_label(f"{root}/label", f"{tag}: {h.name}",
                                                  position=(o[0], o[1], o[2] + 0.12)))
        for j, pos in bases(h, tag).items():
            st["nodes"].append(server.scene.add_icosphere(f"{root}/base_{j}", radius=0.005, position=tuple(pos),
                                                          color=COLORS[tag]))

    def bases(h, tag) -> dict[str, np.ndarray]:
        """First joint of each finger -> its position (env-local world, as captured). For B, A's joint names are
        mapped through the joint map so both hands list the same fingers."""
        child = dict(zip(h.d["hand_joint_names"], h.d["joint_link_bodies"]))
        if tag == "A" and st["lmap"]:
            firsts = [j for j in h.movable if st["lmap"].get(j, "").endswith("_j0")]
        elif tag == "B" and st["lmap"]:
            back = {v: k for k, v in st["lmap"].items()}
            firsts = [j for j in h.movable if j.endswith("_j0") and j in back]
        else:                                    # no joint map: a replica's slots name their depth, a vendor's don't
            firsts = [j for j in h.movable if j.endswith("_j0")]
        return {j: np.array(h.d["bodies"][child[j]][:3]) for j in firsts}

    def base_table() -> str:
        A, B = st["hands"]["A"], st["hands"]["B"]
        if not st["lmap"]:
            return ""
        ba, bb = bases(A, "A"), bases(B, "B")
        rows = ["| finger base (A) | A (x, y, z) mm | B (x, y, z) mm | B - A mm | gap mm |", "|---|---|---|---|---|"]
        for j, pa in ba.items():
            jb = st["lmap"][j]
            if jb not in bb:
                continue
            a = (pa - A.c) @ A.R_canon * 1000
            b = (bb[jb] - B.c) @ B.R_canon * 1000
            f = lambda v: "(" + ", ".join(f"{x:+.1f}" for x in v) + ")"
            rows.append(f"| {j} | {f(a)} | {f(b)} | {f(b - a)} | {np.linalg.norm(b - a):.1f} |")
        return ("\n\n**Finger bases in the palm frame** (from each palm centre; x normal, y width, z length)\n\n"
                + "\n".join(rows))

    def rebuild(_=None):
        for r in st["robots"].values():
            r.remove()
        for n in st["nodes"]:              # the /A and /B roots: removing one removes everything under it
            if n.name.count("/") == 1:
                n.remove()
        st["robots"], st["nodes"] = {}, []
        st["hands"] = {"A": Hand(dd_a.value), "B": Hand(dd_b.value)}
        st["lmap"] = label_map(dd_a.value, dd_b.value)
        gap = 0.0 if g_overlay.value else g_gap.value
        draw("A", np.array([0.0, -gap / 2, 0.0]))
        draw("B", np.array([0.0, gap / 2, 0.0]))
        build_sliders("A", f_a)
        build_sliders("B", f_b)
        pairing = ("joints paired through the replica's joint map" if st["lmap"]
                   else "no joint map between these two: per-joint terms pair only by identical labels")
        md_sum.content = (summary(st["hands"]["A"], "A") + "\n" + summary(st["hands"]["B"], "B")
                          + f"\n\n_{pairing}_" + base_table())
        refresh_obs()

    def preset(which):
        def cb(_):
            for f, c in g_terms.items():
                c.value = which == "all" or ((f in TOKEN_TERMS) == (which == "tok") and f != "joint_enabled")
            refresh_obs()
        return cb

    def reset(_):
        for tag in ("A", "B"):
            h = st["hands"][tag]
            h.q = dict(h.spawn_q)
            st["muted"] = True
            for j, s in st["sliders"][tag].items():
                s.value = float(np.clip(h.q[j], *h.limits[j]))
            st["muted"] = False
            pose(tag)
        refresh_obs()

    g_axes.on_update(lambda _: (draw_axes("A"), draw_axes("B")))
    dd_a.on_update(rebuild)
    dd_b.on_update(rebuild)
    g_overlay.on_update(rebuild)
    g_gap.on_update(rebuild)
    for c in g_terms.values():
        c.on_update(refresh_obs)
    dd_term.on_update(refresh_obs)
    b_all.on_click(preset("all"))
    b_tok.on_click(preset("tok"))
    b_glob.on_click(preset("glob"))
    b_reset.on_click(reset)

    def copy_a(_):
        A, B = st["hands"]["A"], st["hands"]["B"]
        st["muted"] = True
        for ja, jb in st["lmap"].items():
            if jb in st["sliders"]["B"]:
                v = float(np.clip(A.q[ja], *B.limits[jb]))
                B.q[jb] = v
                st["sliders"]["B"][jb].value = v
        st["muted"] = False
        pose("B")
        refresh_obs()
    b_copy.on_click(copy_a)
    rebuild()
    print(f"viser on port {args.port}: {len(names)} captures: {', '.join(names)}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
