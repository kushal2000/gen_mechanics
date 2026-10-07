"""Inspect the Wuji v2 hand and its reduced variants in viser -- kinematics only, no simulator.

Every URDF matching assets/urdf/unified_dynamics_commercial_hands/wuji2/wuji2_left*.urdf is offered in a
dropdown: the full hand and every variant generated next to it (missing fingers today; locked joints and other
reductions as they are added). For the selected hand: one slider per movable joint within its URDF limits,
pose buttons, collision-mesh view, and a summary -- movable joints, fingertips, link mass, and what the variant
removed relative to the full hand. Optionally the full hand is drawn beside it for comparison.

    .venv_isaacsim/bin/python experiments/06oct_minimal_embodiment/viser_variants.py --port 8082
"""
from __future__ import annotations

import argparse
import json
import pathlib
import socket
import time
import xml.etree.ElementTree as ET

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
HAND_DIR = REPO / "assets/urdf/unified_dynamics_commercial_hands/wuji2"
FULL_OF = {"L": HAND_DIR / "wuji2_left.urdf", "R": HAND_DIR / "wuji2_right.urdf"}   # R: Wuji's own right hand
FULL = FULL_OF["L"]


def variants() -> dict[str, pathlib.Path]:
    """Display name -> URDF: each side's full hand first, then its variants ("L no pinky", "R only thumb index")."""
    out = {}
    for side, word in (("L", "left"), ("R", "right")):
        if not FULL_OF[side].exists():
            continue
        out[f"{side} full hand"] = FULL_OF[side]
        for p in sorted(HAND_DIR.glob(f"wuji2_{word}_*.urdf")):
            out[f"{side} " + p.stem.replace(f"wuji2_{word}_", "").replace("_", " ")] = p
    return out


def summary(urdf: pathlib.Path) -> dict:
    root = ET.parse(urdf).getroot()
    movable = [j.get("name") for j in root.findall("joint") if j.get("type") in ("revolute", "continuous", "prismatic")]
    mass = sum(float(m.get("value")) for m in root.iter("mass"))
    spec_path = urdf.with_suffix("").with_suffix(".spec.json") if urdf.suffix == ".urdf" else None
    spec_path = urdf.parent / (urdf.stem + ".spec.json")
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    return {"links": [l.get("name") for l in root.findall("link")], "movable": movable, "mass": mass,
            "tips": spec.get("fingertip_body_names", []), "removed": spec.get("missing_finger") or spec.get("reduction")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8082)
    args = ap.parse_args()

    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=args.port)
    print(f"\n[variants] viser on http://{socket.gethostname()}:{args.port}\n", flush=True)
    server.scene.set_up_direction("+z")
    opts = variants()

    with server.gui.add_folder("hand"):
        dd = server.gui.add_dropdown("variant", tuple(opts), initial_value="L full hand")
        g_full = server.gui.add_checkbox("show full hand beside it", False)
        g_coll = server.gui.add_checkbox("show collision meshes", False)
        md = server.gui.add_markdown("")
    with server.gui.add_folder("pose"):
        b_home = server.gui.add_button("home (all 0)")
        b_curl = server.gui.add_button("curl (all at 60% of upper limit)")
        b_rand = server.gui.add_button("random pose")
    sliders_folder = server.gui.add_folder("joints")

    state = {"robot": None, "ghost": None, "sliders": {}, "limits": {}}

    def apply(_=None):
        if state["robot"] is not None:
            state["robot"].update_cfg(np.array([s.value for s in state["sliders"].values()]))

    def set_all(fn):
        for name, s in state["sliders"].items():
            lo, hi = state["limits"][name]
            s.value = float(np.clip(fn(lo, hi), lo, hi))
        apply()

    b_home.on_click(lambda _: set_all(lambda lo, hi: 0.0))
    b_curl.on_click(lambda _: set_all(lambda lo, hi: 0.6 * hi))
    b_rand.on_click(lambda _: set_all(lambda lo, hi: np.random.uniform(lo, hi)))

    def load(name):
        for key in ("robot", "ghost"):
            if state[key] is not None:
                state[key].remove()
                state[key] = None
        for s in state["sliders"].values():
            s.remove()
        state["sliders"], state["limits"] = {}, {}
        urdf = opts[name]
        full = FULL_OF[name[0]]                   # the same side's full hand
        full_info = summary(full)
        state["robot"] = ViserUrdf(server, urdf, root_node_name="/hand", load_collision_meshes=True)
        state["robot"].show_collision = bool(g_coll.value)
        if g_full.value and not name.endswith("full hand"):
            server.scene.add_frame("/ghost", position=(0.0, 0.25, 0.0), show_axes=False)
            state["ghost"] = ViserUrdf(server, full, root_node_name="/ghost/hand", mesh_color_override=(0.6, 0.6, 0.6, 0.5))
            state["ghost"].update_cfg(np.zeros(len(state["ghost"].get_actuated_joint_names())))
        limits = state["robot"].get_actuated_joint_limits()
        with sliders_folder:
            for jn in state["robot"].get_actuated_joint_names():
                lo, hi = limits[jn]
                lo = -np.pi if lo is None else float(lo)
                hi = np.pi if hi is None else float(hi)
                s = server.gui.add_slider(jn, lo, hi, 0.01, float(np.clip(0.0, lo, hi)))
                s.on_update(apply)
                state["sliders"][jn], state["limits"][jn] = s, (lo, hi)
        apply()
        info = summary(urdf)
        gone_links = sorted(set(full_info["links"]) - set(info["links"]))
        gone_joints = sorted(set(full_info["movable"]) - set(info["movable"]))
        md.content = (f"**{name}** — {len(info['movable'])} movable joints, {len(info['tips'])} fingertips, "
                      f"{1000 * info['mass']:.0f} g of links\n\n"
                      f"full hand: {len(full_info['movable'])} joints, {len(full_info['tips'])} fingertips, "
                      f"{1000 * full_info['mass']:.0f} g\n\n"
                      + (f"removed joints: {', '.join(gone_joints)}\n\nremoved links: {', '.join(gone_links)}"
                         if gone_joints or gone_links else "nothing removed"))

    dd.on_update(lambda _: load(str(dd.value)))
    g_full.on_update(lambda _: load(str(dd.value)))
    g_coll.on_update(lambda _: setattr(state["robot"], "show_collision", bool(g_coll.value)) if state["robot"] else None)
    load("L full hand")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
