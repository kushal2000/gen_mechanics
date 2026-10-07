"""Commercial hand vs its grammar replica, at the in-hand reorientation SPAWN state, in viser.

Reads the JSONs capture_spawn.py writes (results/spawn/<name>.json; no simulator here). For the selected pair:

  * both hands drawn at their captured spawn pose -- joint positions from the sim, root at the sim's root body
    pose -- with the cube, the goal (ghost) and each palm frame, side by side or overlaid;
  * the policy observation term by term: dims, the two values (per joint where the term is per-joint), and the
    difference. Terms whose layout differs between the hands (a different joint count) are listed per hand.

    .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/viser_spawn.py --port 8084
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPAWN = HERE / "results/spawn"
URDF = {
    "wuji2_commercial": REPO / "assets/urdf/unified_dynamics_commercial_hands/wuji2/wuji2_left.urdf",
    "leap_commercial": REPO / "assets/urdf/unified_dynamics_commercial_hands/leap/leap_left.urdf",
    "allegro_commercial": REPO / "assets/urdf/unified_dynamics_commercial_hands/allegro/allegro_left.urdf",
    "wuji2_replica": REPO / "assets/urdf/unified_dynamics_commercial_hands_replicas/wuji2/wuji2_left_replica.urdf",
    "leap_replica": REPO / "assets/urdf/unified_dynamics_commercial_hands_replicas/leap/leap_left_replica.urdf",
    "allegro_replica": REPO / "assets/urdf/unified_dynamics_commercial_hands_replicas/allegro/allegro_left_replica.urdf",
    "midas_replica": REPO / "assets/urdf/unified_dynamics_commercial_hands_replicas/midas/midas_replica.urdf",
}
COLORS = {"commercial": (0.85, 0.55, 0.2), "replica": (0.2, 0.45, 0.85)}


def load(name: str) -> dict | None:
    p = SPAWN / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pairs() -> dict[str, tuple[str, str]]:
    have = {p.stem for p in SPAWN.glob("*.json")}
    out = {}
    for hand in ("wuji2", "leap", "allegro"):
        if f"{hand}_commercial" in have or f"{hand}_replica" in have:
            out[hand] = (f"{hand}_commercial", f"{hand}_replica")
    return out


def joint_map(replica_name: str) -> dict[str, str]:
    """Vendor joint -> replica slot (make_replicas.py writes it next to the replica URDF)."""
    u = URDF.get(replica_name)
    p = u.with_suffix(".joint_map.json") if u is not None else None
    return json.loads(p.read_text()) if p is not None and p.exists() else {}


def align(ta: dict, tb: dict, jmap: dict[str, str]) -> list[tuple[str, float, str, float]] | None:
    """Entries of one term paired across the hands: (A label, A value, B label, B value), or None if not pairable.

    Same labels pair by label; otherwise A's vendor joint names go through the joint map ("name" or "name[i]")."""
    if ta["labels"] == tb["labels"]:
        return [(l, x, l, y) for l, x, y in zip(ta["labels"], ta["values"], tb["values"])]
    if not jmap:
        return None
    vb = dict(zip(tb["labels"], tb["values"]))
    out = []
    for lab, x in zip(ta["labels"], ta["values"]):
        base, _, rest = lab.partition("[")
        lb = jmap.get(base, None)
        lb = None if lb is None else (lb + ("[" + rest if rest else ""))
        if lb is None or lb not in vb:
            return None
        out.append((lab, x, lb, vb[lb]))
    return out


def _fmt(v: float) -> str:
    return f"{v:+.3f}"


def obs_table(a: dict | None, b: dict | None, jmap: dict[str, str]) -> str:
    """Markdown: one row per observation term; differences over the entries that pair up (by label or joint map).
    B-only entries are the replica's ghost slots (the grammar's 5 x 6 template); their largest |value| is shown."""
    if a is None or b is None:
        return "_capture both hands to compare_"
    rows = ["| term | dim A / B | max abs diff | mean abs diff | B-only max abs |", "|---|---|---|---|---|"]
    for f in a["obs_list"]:
        ta, tb = a["terms"].get(f), b["terms"].get(f)
        if tb is None:
            rows.append(f"| `{f}` | {len(ta['values'])} / - | | only A |")
            continue
        pairs_ = align(ta, tb, jmap)
        if pairs_ is None:
            rows.append(f"| `{f}` | {len(ta['values'])} / {len(tb['values'])} | - | - | not pairable |")
            continue
        d = np.array([y - x for _, x, _, y in pairs_])
        used = {lb for _, _, lb, _ in pairs_}
        extra = [abs(v) for l, v in zip(tb["labels"], tb["values"]) if l not in used]
        rows.append(f"| `{f}` | {len(ta['values'])} / {len(tb['values'])} | {np.abs(d).max():.3f} | "
                    f"{np.abs(d).mean():.3f} | {max(extra):.3f} |" if extra else
                    f"| `{f}` | {len(ta['values'])} / {len(tb['values'])} | {np.abs(d).max():.3f} | "
                    f"{np.abs(d).mean():.3f} | |")
    return "\n".join(rows)


def term_detail(a: dict | None, b: dict | None, f: str, jmap: dict[str, str]) -> str:
    """Markdown: the selected term entry by entry, A next to B (aligned by label where both have it)."""
    out = []
    for tag, d in (("A", a), ("B", b)):
        if d is None or f not in d["terms"]:
            out.append(f"**{tag}**: not in this capture")
    if a is None or b is None or f not in a["terms"] or f not in b["terms"]:
        return "\n\n".join(out)
    ta, tb = a["terms"][f], b["terms"][f]
    la, lb = ta["labels"], tb["labels"]
    pairs_ = align(ta, tb, jmap)
    if pairs_ is not None:
        rows = ["| A entry | B entry | A | B | B - A |", "|---|---|---|---|---|"]
        for l1, x, l2, y in pairs_:
            rows.append(f"| {l1} | {l2} | {_fmt(x)} | {_fmt(y)} | {_fmt(y - x)} |")
        used = {l2 for _, _, l2, _ in pairs_}
        for l2, y in zip(lb, tb["values"]):
            if l2 not in used:
                rows.append(f"| | {l2} (B only) | | {_fmt(y)} | |")
        return "\n".join(rows)
    n = max(len(la), len(lb))
    rows = ["| A entry | A | B entry | B |", "|---|---|---|---|"]
    for i in range(n):
        ca = (la[i], _fmt(ta["values"][i])) if i < len(la) else ("", "")
        cb = (lb[i], _fmt(tb["values"][i])) if i < len(lb) else ("", "")
        rows.append(f"| {ca[0]} | {ca[1]} | {cb[0]} | {cb[1]} |")
    return "\n".join(rows)


def spawn_summary(a: dict | None, b: dict | None) -> str:
    def line(tag, d):
        if d is None:
            return f"* **{tag}**: no capture"
        o, g = np.array(d["object"][:3]), np.array(d["goal"][:3])
        pc = np.array(d["palm_center"])
        return (f"* **{tag}** `{d['name']}`: obs {d['obs_dim']}, {len(d['hand_joint_names'])} hand joints; "
                f"palm `{d['palm_body']}` root at {np.round(d['root'][:3], 3).tolist()}; "
                f"cube {np.round(o, 3).tolist()}, {np.linalg.norm(o - pc) * 100:.1f} cm from palm centre")
    return "\n".join([line("A", a), line("B", b)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8084)
    args = ap.parse_args()

    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    opts = pairs()
    if not opts:
        raise SystemExit(f"no captures in {SPAWN}; run capture_spawn.py first")
    state: dict = {"nodes": [], "robots": [], "ab": (None, None), "jmap": {}}

    with server.gui.add_folder("pair"):
        dd = server.gui.add_dropdown("hand", tuple(opts), initial_value=tuple(opts)[0])
        g_overlay = server.gui.add_checkbox("overlay (same place)", False)
        g_gap = server.gui.add_slider("side-by-side gap (m)", 0.0, 0.6, 0.01, 0.3)
        md_sum = server.gui.add_markdown("")
    with server.gui.add_folder("observations at spawn (A commercial, B replica)"):
        md_tab = server.gui.add_markdown("")
        dd_term = server.gui.add_dropdown("term", ("-",))
        md_term = server.gui.add_markdown("")

    def clear():
        for r in state["robots"]:
            r.remove()
        for n in state["nodes"]:
            n.remove()
        state["robots"], state["nodes"] = [], []

    def draw(d: dict, tag: str, offset: np.ndarray):
        kind = "commercial" if tag == "A" else "replica"
        root = f"/{kind}"
        state["nodes"].append(server.scene.add_frame(root, position=tuple(offset), show_axes=False))
        x, y, z, qw, qx, qy, qz = d["root"]
        state["nodes"].append(server.scene.add_frame(f"{root}/base", position=(x, y, z), wxyz=(qw, qx, qy, qz),
                                                     show_axes=False))
        urdf = URDF.get(d["name"])
        if urdf is not None and urdf.exists():
            r = ViserUrdf(server, urdf, root_node_name=f"{root}/base/hand",
                          mesh_color_override=COLORS[kind] + (0.85,))
            names = r.get_actuated_joint_names()
            jp = dict(zip(d["joint_names"], d["joint_pos"]))
            r.update_cfg(np.array([jp.get(n, 0.0) for n in names]))
            state["robots"].append(r)
        # every sim body's frame, so a viewer-URDF mismatch would show as frames off the meshes
        for n, p in d["bodies"].items():
            state["nodes"].append(server.scene.add_frame(f"{root}/bodies/{n}", position=tuple(p[:3]),
                                                         wxyz=tuple(p[3:]), axes_length=0.012, axes_radius=0.0008))
        pc = d["palm_center"]
        state["nodes"].append(server.scene.add_icosphere(f"{root}/palm_center", radius=0.006, position=tuple(pc),
                                                         color=(0.1, 0.1, 0.1)))
        pb = d["bodies"][d["palm_body"]]
        state["nodes"].append(server.scene.add_frame(f"{root}/palm_frame", position=tuple(pb[:3]),
                                                     wxyz=tuple(pb[3:]), axes_length=0.05, axes_radius=0.002))
        o, g = d["object"], d["goal"]
        state["nodes"].append(server.scene.add_box(f"{root}/object", dimensions=(0.05, 0.05, 0.05),
                                                   position=tuple(o[:3]), wxyz=tuple(o[3:]),
                                                   color=COLORS[kind], opacity=0.6))
        state["nodes"].append(server.scene.add_box(f"{root}/goal", dimensions=(0.05, 0.05, 0.05),
                                                   position=tuple(g[:3]), wxyz=tuple(g[3:]),
                                                   color=(0.3, 0.8, 0.3), opacity=0.25))
        state["nodes"].append(server.scene.add_label(f"{root}/label", f"{tag}: {d['name']}",
                                                     position=(o[0], o[1], o[2] + 0.12)))

    def render(_=None):
        clear()
        a_name, b_name = opts[dd.value]
        a, b = load(a_name), load(b_name)
        state["ab"] = (a, b)               # before touching dd_term: setting its value fires its callback
        gap = 0.0 if g_overlay.value else g_gap.value
        if a is not None:
            draw(a, "A", np.array([0.0, -gap / 2, 0.0]))
        if b is not None:
            draw(b, "B", np.array([0.0, gap / 2, 0.0]))
        md_sum.content = spawn_summary(a, b)
        state["jmap"] = joint_map(b_name)
        md_tab.content = obs_table(a, b, state["jmap"])
        terms = tuple((a or b)["obs_list"])
        dd_term.options = terms
        if dd_term.value not in terms:
            dd_term.value = terms[0]
        md_term.content = term_detail(a, b, dd_term.value, state["jmap"])

    dd.on_update(render)
    g_overlay.on_update(render)
    g_gap.on_update(render)
    dd_term.on_update(lambda _: setattr(md_term, "content", term_detail(*state["ab"], dd_term.value, state["jmap"])))
    render()
    print(f"viser on port {args.port}: {', '.join(opts)}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
