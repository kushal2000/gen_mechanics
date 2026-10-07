"""Play recorded policy rollouts (record_rollout.py) in viser: watch the same policy on two hands side by side.

Pick A and B from results/rollouts/*.npz, an env, then play / pause / scrub the timeline. Each hand is drawn in its
own canonical palm frame (palm centre at the origin, as the policy sees it), with its cube and the goal (green
ghost); the status line counts goals and drops so far and flags each reset.

    .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/viser_play.py --port 8085
"""
from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = pathlib.Path(__file__).resolve().parent
ROLL = HERE / "results/rollouts"
COLORS = {"A": (0.85, 0.55, 0.2), "B": (0.2, 0.45, 0.85)}


def load(name: str) -> dict:
    z = np.load(ROLL / f"{name}.npz")
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8085)
    args = ap.parse_args()

    import viser
    from viser.extras import ViserUrdf

    from viser_spawn import urdf_of

    names = sorted(p.stem for p in ROLL.glob("*.npz"))
    if not names:
        raise SystemExit(f"no rollouts in {ROLL}; run record_rollout.py first")
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    st: dict = {"data": {}, "robots": {}, "nodes": {}, "t": 0, "playing": False}

    with server.gui.add_folder("rollouts"):
        dd_a = server.gui.add_dropdown("A", tuple(names), initial_value=names[0])
        dd_b = server.gui.add_dropdown("B", tuple(names), initial_value=names[-1])
        g_env = server.gui.add_slider("env", 0, 1, 1, 0)
        g_overlay = server.gui.add_checkbox("overlay (same place)", False)
        g_gap = server.gui.add_slider("side-by-side gap (m)", 0.0, 0.6, 0.01, 0.3)
    with server.gui.add_folder("playback"):
        b_play = server.gui.add_button("play / pause")
        g_speed = server.gui.add_slider("speed (x real time)", 0.1, 4.0, 0.1, 1.0)
        g_t = server.gui.add_slider("step", 0, 1, 1, 0)
        md = server.gui.add_markdown("")

    def T_canon(d, env):
        p = d["palm"][0, env]                          # hand-only: the palm does not move
        w, x, y, z = p[3:]
        return Rot.from_quat([x, y, z, w]).as_matrix(), p[:3]

    def build(_=None):
        for r in st["robots"].values():
            r.remove()
        for n in st["nodes"].values():
            if n is not None and n.name.count("/") == 1:
                n.remove()
        st["robots"], st["nodes"] = {}, {}
        st["data"] = {"A": load(dd_a.value), "B": load(dd_b.value)}
        n_env = min(d["q"].shape[1] for d in st["data"].values())
        g_env.max = max(n_env - 1, 1)
        env = min(int(g_env.value), n_env - 1)
        steps = min(d["q"].shape[0] for d in st["data"].values())
        g_t.max = steps - 1
        gap = 0.0 if g_overlay.value else g_gap.value
        for tag, off in (("A", -gap / 2), ("B", gap / 2)):
            d = st["data"][tag]
            Rc, c = T_canon(d, env)
            q = Rot.from_matrix(Rc.T).as_quat()
            st["nodes"][f"{tag}"] = server.scene.add_frame(f"/{tag}", position=tuple(np.array([0.0, off, 0.0]) - Rc.T @ c),
                                                           wxyz=(q[3], q[0], q[1], q[2]), show_axes=False)
            st["nodes"][f"{tag}_base"] = server.scene.add_frame(f"/{tag}/base", show_axes=False)
            urdf = urdf_of(d["meta"]["name"])
            if urdf is not None:
                st["robots"][tag] = ViserUrdf(server, urdf, root_node_name=f"/{tag}/base/hand",
                                              mesh_color_override=COLORS[tag] + (0.85,))
            size = tuple(d["meta"].get("object_size", (0.05, 0.05, 0.05)))
            st["nodes"][f"{tag}_obj"] = server.scene.add_box(f"/{tag}/object", dimensions=size, color=COLORS[tag],
                                                            opacity=0.7)
            st["nodes"][f"{tag}_goal"] = server.scene.add_box(f"/{tag}/goal", dimensions=size, color=(0.3, 0.8, 0.3),
                                                             opacity=0.25)
            st["nodes"][f"{tag}_label"] = server.scene.add_label(f"/{tag}/label", f"{tag}: {d['meta']['name']}",
                                                                position=tuple(c + Rc[:, 0] * 0.18))
        show(int(g_t.value))

    def show(t):
        env = int(g_env.value)
        lines = []
        for tag in ("A", "B"):
            d = st["data"][tag]
            t = min(t, d["q"].shape[0] - 1)
            x, y, z, qw, qx, qy, qz = d["root"][t, env]
            st["nodes"][f"{tag}_base"].position = (x, y, z)
            st["nodes"][f"{tag}_base"].wxyz = (qw, qx, qy, qz)
            r = st["robots"].get(tag)
            if r is not None:
                jp = dict(zip(d["meta"]["joint_names"], d["q"][t, env]))
                r.update_cfg(np.array([jp.get(j, 0.0) for j in r.get_actuated_joint_names()]))
            for k in ("object", "goal"):
                p = d[k][t, env]
                st["nodes"][f"{tag}_{'obj' if k == 'object' else 'goal'}"].position = tuple(p[:3])
                st["nodes"][f"{tag}_{'obj' if k == 'object' else 'goal'}"].wxyz = tuple(p[3:])
            # this episode so far: since the last reset at or before t
            resets = np.nonzero(d["reset"][: t + 1, env])[0]
            start = int(resets[-1]) + 1 if len(resets) else 0
            goals = int(d["success"][start: t + 1, env].sum())
            total_g, total_d = int(d["success"][: t + 1, env].sum()), int(d["fall"][: t + 1, env].sum())
            flag = " **RESET**" if d["reset"][t, env] else ""
            lines.append(f"**{tag}** `{d['meta']['name']}`: this episode {goals} goals | so far {total_g} goals, "
                         f"{total_d} drops{flag}")
        dt = st["data"]["A"]["meta"]["dt"]
        md.content = f"t = {t * dt:.2f} s (step {t})\n\n" + "\n\n".join(lines)

    def loop():
        while True:
            if st["playing"] and st["data"]:
                dt = st["data"]["A"]["meta"]["dt"]
                nxt = int(g_t.value) + 1
                if nxt > g_t.max:
                    nxt = 0
                g_t.value = nxt
                time.sleep(dt / max(g_speed.value, 0.05))
            else:
                time.sleep(0.05)

    def toggle(_):
        st["playing"] = not st["playing"]

    dd_a.on_update(build)
    dd_b.on_update(build)
    g_env.on_update(build)
    g_overlay.on_update(build)
    g_gap.on_update(build)
    g_t.on_update(lambda _: show(int(g_t.value)))
    b_play.on_click(toggle)
    build()
    threading.Thread(target=loop, daemon=True).start()
    print(f"viser play on port {args.port}: {', '.join(names)}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
