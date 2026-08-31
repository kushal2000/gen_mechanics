"""Browse the minimal hand design space in the browser.

    python viewer.py
    python viewer.py --port 8081 --seed 7

Self-contained: capsules are built straight from sampler.forward_kinematics, so
unlike gen_mechanics' design_space_viewer.py this needs no URDF build, no
yourdfpy and no Isaac Sim.

One slider per ACTUATOR, not per joint -- a rigidly coupled pair shares one
slider, and passive joints have none. Passive joints instead get sliders under
"passive (external)", standing in for a contact force pushing them off their
spring rest position.

Markers: a sphere at each joint centre, coloured by the roles living there --
purple if anything is passive, grey-blue if anything is rigidly coupled,
otherwise by DOF (orange FE, blue AA, green both). The yellow sphere is the
FINGERTIP, not a joint.
"""

from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np
import trimesh
import viser

from hand_sampler.minimal import sampler as S

PALM_COLOR = (205, 209, 216)
LINK_COLOR = (168, 176, 188)
JOINT_COLOR = {
    ("FE",): (232, 122, 84),
    ("AA",): (86, 160, 214),
    ("FE", "AA"): (140, 198, 104),
}
PASSIVE_COLOR = (168, 112, 214)
RIGID_COLOR = (96, 116, 148)
TIP_COLOR = (242, 205, 70)


def location_color(finger, li):
    """Role wins over DOF composition: passive > rigid > independent."""
    roles = {S.joint_role(finger, (li, dof)) for dof in finger.dofs[li]}
    if "passive" in roles:
        return PASSIVE_COLOR
    if "rigid" in roles:
        return RIGID_COLOR
    return JOINT_COLOR[tuple(finger.dofs[li])]


def capsule_mesh(p0: np.ndarray, p1: np.ndarray, radius: float) -> trimesh.Trimesh:
    """A capsule whose TOTAL tip-to-tip extent spans p0 -> p1 exactly.

    Two things to get right, both of which were wrong before:
      * trimesh.creation.capsule is CENTRED on the origin (z from -h/2-r to
        +h/2+r), so the transform has to put the segment MIDPOINT at the origin,
        not p0. Placing p0 there shifts every link back by half its length --
        which is what put joints in the middle of capsules and pushed the base
        link through the palm.
      * the cylinder section must be shortened by 2r so the hemispherical caps
        land ON the joints rather than overhanging them. Then the tip of one link
        meets the base of the next exactly at the shared joint centre.
    """
    d = np.asarray(p1, float) - np.asarray(p0, float)
    L = float(np.linalg.norm(d))
    h = max(L - 2.0 * radius, 1e-6)
    mesh = trimesh.creation.capsule(height=h, radius=radius, count=(16, 16))

    v = d / (L + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, v)
    s, c = float(np.linalg.norm(axis)), float(z @ v)
    T = np.eye(4)
    if s < 1e-9:
        T[:3, :3] = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        a = axis / s
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        T[:3, :3] = np.eye(3) + s * K + (1 - c) * (K @ K)
    T[:3, 3] = (np.asarray(p0, float) + np.asarray(p1, float)) / 2.0
    mesh.apply_transform(T)
    return mesh


def _jname(j) -> str:
    return f"{S.JOINT_LOCATIONS[j[0]]} {j[1]}"


def describe(hand: S.Hand) -> str:
    n_act = sum(S.n_actuators(f) for f in hand.fingers)
    lines = [f"**{len(hand.fingers)} fingers &nbsp;·&nbsp; {hand.n_joints} joints "
             f"&nbsp;·&nbsp; {n_act} actuators**", ""]
    for i, f in enumerate(hand.fingers):
        lens = " + ".join(f"{L * 100:.0f}" for L in f.link_lengths)
        lines.append(
            f"- **f{i}** on **{f.face}** &nbsp; {lens} cm &nbsp;·&nbsp; "
            f"splay {math.degrees(f.splay):+.0f}&deg;"
        )
        for act in S.actuators(f):
            if act[0] == "independent":
                lines.append(f"  &nbsp;&nbsp;· {_jname(act[1])}")
            else:
                lines.append(f"  &nbsp;&nbsp;· **{_jname(act[1])} = {_jname(act[2])}** "
                             f"&nbsp;(rigid 1:1)")
        for j in f.coupling.passive:
            lines.append(f"  &nbsp;&nbsp;· _{_jname(j)} passive_ "
                         f"({S.passive_stiffness(j):.2f} N·m/rad)")
        if S.n_actuators(f) == 0:
            lines.append("  &nbsp;&nbsp;⚠ **no actuators — fully passive finger**")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    gui_seed = server.gui.add_number("seed", initial_value=args.seed, step=1)
    gui_nf = server.gui.add_dropdown("n fingers", ("random", "2", "3"),
                                     initial_value="random")
    gui_new = server.gui.add_button("resample")
    gui_markers = server.gui.add_checkbox("show joint markers", True)
    gui_flexall = server.gui.add_slider("flex all (FE)", -10.0, 120.0, 1.0, 0.0)
    gui_spreadall = server.gui.add_slider("spread all (AA)", -20.0, 20.0, 1.0, 0.0)
    gui_zero = server.gui.add_button("reset pose")
    gui_info = server.gui.add_markdown("")

    state: dict = {"hand": None, "meshes": [], "sliders": {}, "folders": []}

    def redraw() -> None:
        for h in state["meshes"]:
            h.remove()
        state["meshes"] = []
        hand: S.Hand = state["hand"]

        state["meshes"].append(
            server.scene.add_box(
                "/palm", dimensions=S.PALM_EXTENTS, color=PALM_COLOR,
                position=(0.0, 0.0, S.PALM_LENGTH / 2),
            )
        )

        for fi, finger in enumerate(hand.fingers):
            inputs = {k[2]: math.radians(sl.value)
                      for k, sl in state["sliders"].items()
                      if k[0] == fi and k[1] == "act"}
            passive = {k[2]: math.radians(sl.value)
                       for k, sl in state["sliders"].items()
                       if k[0] == fi and k[1] == "passive"}
            angles = S.expand_inputs(finger, inputs, passive)
            joints, segments = S.forward_kinematics(finger, angles)

            for si, (p0, p1, r) in enumerate(segments):
                state["meshes"].append(
                    server.scene.add_mesh_trimesh(
                        f"/f{fi}/link{si}", capsule_mesh(p0, p1, r)
                    )
                )
            if gui_markers.value:
                for ji, p in enumerate(joints):
                    tip = ji == len(joints) - 1
                    state["meshes"].append(
                        server.scene.add_icosphere(
                            f"/f{fi}/marker{ji}",
                            radius=S.CAPSULE_RADIUS * (0.75 if tip else 1.05),
                            color=TIP_COLOR if tip else location_color(finger, ji),
                            position=tuple(p),
                        )
                    )
        gui_info.content = describe(hand)

    def rebuild_sliders() -> None:
        for h in state["sliders"].values():
            h.remove()
        for f in state["folders"]:
            f.remove()
        state["sliders"], state["folders"] = {}, []

        for fi, finger in enumerate(state["hand"].fingers):
            folder = server.gui.add_folder(f"f{fi}  ({finger.face})")
            state["folders"].append(folder)
            with folder:
                for act in S.actuators(finger):
                    driver = act[1]
                    label = (_jname(driver) if act[0] == "independent"
                             else f"{_jname(act[1])} = {_jname(act[2])}")
                    lo, hi = S.joint_limit(driver)
                    sl = server.gui.add_slider(label, math.degrees(lo),
                                               math.degrees(hi), 1.0, 0.0)
                    sl.on_update(lambda _: redraw())
                    state["sliders"][(fi, "act", driver)] = sl

                if finger.coupling.passive:
                    sub = server.gui.add_folder("passive (external)")
                    state["folders"].append(sub)
                    with sub:
                        for j in finger.coupling.passive:
                            lo, hi = S.joint_limit(j)
                            # Start at the SPRING REST position, not zero. redraw()
                            # feeds these sliders to expand_inputs as explicit
                            # overrides, so a 0 here would stomp passive_home and
                            # every passive FE would sit flat instead of at +55.
                            sl = server.gui.add_slider(
                                _jname(j), math.degrees(lo), math.degrees(hi), 1.0,
                                math.degrees(S.passive_home(j))
                            )
                            sl.on_update(lambda _: redraw())
                            state["sliders"][(fi, "passive", j)] = sl

    def resample() -> None:
        rng = random.Random(int(gui_seed.value))
        nf = None if gui_nf.value == "random" else int(gui_nf.value)
        state["hand"] = S.sample_hand(rng, n_fingers=nf)
        rebuild_sliders()
        redraw()

    def write_all(dof: str, value: float) -> None:
        """Drive every actuator whose DRIVER joint is of this DOF type."""
        for (fi, kind, joint), sl in state["sliders"].items():
            if kind == "act" and joint[1] == dof:
                sl.value = value
        redraw()

    gui_new.on_click(lambda _: (setattr(gui_seed, "value", int(gui_seed.value) + 1),
                                resample()))
    gui_seed.on_update(lambda _: resample())
    gui_nf.on_update(lambda _: resample())
    gui_markers.on_update(lambda _: redraw())
    gui_flexall.on_update(lambda _: write_all("FE", gui_flexall.value))
    gui_spreadall.on_update(lambda _: write_all("AA", gui_spreadall.value))
    def reset_pose(_=None) -> None:
        """Actuators to zero, passive joints back to their spring rest."""
        for (fi, kind, joint), sl in state["sliders"].items():
            sl.value = (math.degrees(S.passive_home(joint)) if kind == "passive"
                        else 0.0)
        gui_flexall.value = 0.0
        gui_spreadall.value = 0.0
        redraw()

    gui_zero.on_click(reset_pose)

    resample()
    print(f"viser up on http://localhost:{args.port}  (Ctrl-C to stop)", flush=True)

    seen: set[int] = set()
    while True:
        ids = set(server.get_clients().keys())
        if ids - seen:
            redraw()
        seen = ids
        time.sleep(0.5)


if __name__ == "__main__":
    main()
