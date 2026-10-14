"""View primitive hands (primitive_hands.py) in viser: one at a time with per-finger curl sliders, or every built hand
side by side; make new ones from explicit bearings or at random, from the page.

No Isaac Sim: the URDFs are drawn straight from their joint angles (ViserUrdf), so it starts in seconds.

    .venv_isaacsim/bin/python experiments/14oct_design_search/viser_hands.py [--port 8093]
    # then open http://<node>:8093

The palm is drawn face up: its grasp normal (+x of the palm frame, the side the fingers curl toward) points up, so
curling the fingers closes them into a cup above the disc. A red line marks the normal.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import socket
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import primitive_hands as ph  # noqa: E402

# Palm frame -> world: +x (grasp normal) up, +z (bearing 0) along world +x.
FACE_UP = (0.5, 0.5, 0.5, 0.5)          # wxyz: the rotation taking x->z, y->x, z->y
SPACING = 0.36                           # between hands in the grid (a straight hand is 0.34 across)


def built_hands(out_dir: pathlib.Path) -> dict[str, ph.PrimitiveHand]:
    return {p.name[: -len(".design.json")]: ph.PrimitiveHand.from_json(json.loads(p.read_text()))
            for p in sorted(out_dir.glob("*.design.json"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--out", default=str(ph.OUT), help="the folder of built hands")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)

    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=3.0, height=3.0, cell_size=0.05, position=(0.0, 0.0, -0.06))
    print(f"\n[viser_hands] open http://{socket.gethostname()}:{args.port}\n", flush=True)

    hands = built_hands(out_dir)
    if not hands:
        ph.build(ph.PrimitiveHand(bearings_deg=(0.0, 90.0, 180.0, 270.0)), out_dir)
        hands = built_hands(out_dir)
    shown: dict[str, tuple] = {}         # name -> (root frame, ViserUrdf, label)
    state = {"seed": 1000, "dirty": True}

    with server.gui.add_folder("hand"):
        g_mode = server.gui.add_dropdown("show", ("one hand", "every built hand"), initial_value="one hand")
        g_hand = server.gui.add_dropdown("hand", tuple(hands), initial_value=next(iter(hands)))
        md = server.gui.add_markdown("")
    with server.gui.add_folder("curl (degrees, 0 straight .. 90 per joint)"):
        g_all = server.gui.add_slider("every joint", 0.0, 90.0, 1.0, 0.0)
        g_fing = [server.gui.add_slider(f"finger {f}", 0.0, 90.0, 1.0, 0.0) for f in range(6)]
        g_tip = server.gui.add_checkbox("curl only the last 3 joints", False)
    with server.gui.add_folder("make a hand"):
        g_bear = server.gui.add_text("bearings (deg)", "0,90,180,270")
        b_build = server.gui.add_button("build from bearings")
        g_gap = server.gui.add_number("min gap (deg)", 20.0, step=5.0)
        b_rand = server.gui.add_button("build a random hand")
        md_make = server.gui.add_markdown("")

    def show(names: list[str]) -> None:
        for name in list(shown):
            if name not in names:
                frame, urdf, label = shown.pop(name)
                urdf.remove()
                label.remove()
                frame.remove()
        cols = max(1, math.ceil(math.sqrt(len(names))))
        for i, name in enumerate(names):
            pos = (SPACING * (i % cols), -SPACING * (i // cols), 0.0) if len(names) > 1 else (0.0, 0.0, 0.0)
            if name not in shown:
                root = f"/hands/{name}"
                frame = server.scene.add_frame(root, wxyz=FACE_UP, position=pos, show_axes=False)
                urdf = ViserUrdf(server, out_dir / f"{name}.urdf", root_node_name=f"{root}/robot")
                server.scene.add_line_segments(f"{root}/normal", points=np.array([[[0, 0, 0], [0.06, 0, 0]]],
                                               dtype=np.float32), colors=(220, 40, 40), line_width=4.0)
                label = server.scene.add_label(f"/labels/{name}", name, position=(pos[0], pos[1] - 0.17, 0.0))
                shown[name] = (frame, urdf, label)
            else:
                shown[name][0].position = pos
                shown[name][2].position = (pos[0], pos[1] - 0.17, 0.0)

    def pose() -> None:
        tip_only = bool(g_tip.value)
        for name, (_, urdf, _) in shown.items():
            h = hands[name]
            cfg = {}
            for f in range(len(h.bearings_deg)):
                q = math.radians(min(90.0, max(float(g_all.value), float(g_fing[f].value) if f < len(g_fing) else 0.0)))
                for k in range(h.modules_per_finger):
                    cfg[f"f{f}_j{k}"] = 0.0 if (tip_only and k == 0) else q
            urdf.update_cfg(cfg)

    def describe() -> None:
        if g_mode.value == "one hand":
            h = hands[str(g_hand.value)]
            md.content = (f"**{h.name}**  \nbearings {', '.join(f'{b:g}' for b in h.bearings_deg)} deg · gaps "
                          f"{', '.join(f'{g:g}' for g in h.gaps_deg())} deg  \n{len(h.bearings_deg)} fingers x "
                          f"{h.modules_per_finger} modules ({h.module_length * 100:g} cm, radius "
                          f"{h.module_radius * 100:g} cm) · palm radius {h.palm_radius * 100:g} cm, thickness "
                          f"{h.palm_thickness * 100:g} cm")
        else:
            md.content = f"{len(hands)} built hands in `{out_dir.relative_to(ph.REPO)}`"

    def refresh(_=None) -> None:
        state["dirty"] = True

    def add_hand(h: ph.PrimitiveHand) -> None:
        h = h.canonical()
        ph.build(h, out_dir)
        hands.update(built_hands(out_dir))
        g_hand.options = tuple(hands)
        g_hand.value = h.name
        g_mode.value = "one hand"
        md_make.content = f"built **{h.name}**"
        refresh()

    @b_build.on_click
    def _(_) -> None:
        try:
            add_hand(ph.PrimitiveHand(bearings_deg=tuple(float(x) for x in str(g_bear.value).split(",") if x.strip())))
        except Exception as e:                           # a bad list, or fingers too close: say so on the page
            md_make.content = f"**not built:** {e}"

    @b_rand.on_click
    def _(_) -> None:
        for _ in range(50):
            state["seed"] += 1
            h = ph.sample(1, seed=state["seed"], min_gap_deg=float(g_gap.value))[0]
            if h.name not in hands:
                add_hand(h)
                return
        md_make.content = "**no new hand found** -- try a smaller min gap"

    for g in (g_mode, g_hand, g_all, g_tip, *g_fing):
        g.on_update(refresh)

    while True:
        if state["dirty"]:
            state["dirty"] = False
            show([str(g_hand.value)] if g_mode.value == "one hand" else list(hands))
            pose()
            describe()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
