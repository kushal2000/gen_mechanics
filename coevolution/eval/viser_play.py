"""Interactive policy playback: Kit in a child, viser and all drawing here.

Shared engine behind ``play_pose_reach.py`` and ``play_inhand.py``. The parent
owns the viser page and draws the hand itself with ViserUrdf; the child
(``_play_worker.py``) owns Isaac Sim and streams joint positions and object
poses back over a localhost Connection.

WHY THE SPLIT. viser and Kit cannot share one interpreter. Kit prepends a
``pip_prebundle`` carrying an old ``websockets`` and wins the import regardless
of sys.path order, the package's ``__path__``, or an explicit spec pin -- all
three were tried against a real checkpoint and ``websockets.http11`` still
resolved inside the Isaac Sim tree every time. dextoolbench's
eval_interactive_isaacsim.py splits for the same reason (its docstring cites a
segfault; this collision is the harder one).

It also turns out to be the better architecture. The parent draws from joint
angles rather than streaming RTX frames, so viser is a 3D scene you orbit and
inspect rather than a video player, switching designs is a URDF swap, and
nothing needs Isaac's camera.

A PLAIN subprocess, not multiprocessing: spawn re-imports this module in the
child, which would load viser's stack before Kit boots -- the segfault the
reference warns about.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import secrets
import subprocess
import sys
import tempfile
import time
from multiprocessing.connection import Listener

_WORKER = pathlib.Path(__file__).resolve().parent / "_play_worker.py"
# Green, translucent: the goal pose drawn as a ghost of the object itself.
GOAL_GHOST_RGBA = (0.25, 0.85, 0.40, 0.35)
# Keypoint radius for a 45 mm cube at keypoint_scale 1.5: the residual is a
# chordal distance, so mm -> degrees is 2*asin(d / 2r).
KP_RADIUS_M = 0.5 * 1.5 * 0.045 * (3 ** 0.5)


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--checkpoint", default="",
                   help="an rl_games .pth. OMIT IT to run a randomly initialised "
                        "policy -- the right way to check that resets and initial "
                        "poses look sane, with no trained behaviour on top. The env "
                        "then comes from the task's own config instead of a run's.")
    p.add_argument("--run-dir", default="", help="training run dir; inferred from --checkpoint if omitted")
    p.add_argument("--population", default="", help="override the run's robot_spec")
    p.add_argument("--designs", default="0",
                   help="comma-separated design indices to load, one env each, e.g. 0,7,133. "
                        "They all run the same policy on the same goals, so the dropdown is a "
                        "like-for-like comparison.")
    p.add_argument("--include", default="", help="extra population .json to append, e.g. gen-SHARPA")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--tolerance", type=float, default=0.0, help="pin the success tolerance; 0 = the run's own")
    p.add_argument("--trace", default="", help="write a per-step trace of env 0 here")
    p.add_argument("--success-steps", type=int, default=0,
                   help="how many near-goal steps count as a goal; 0 = the run's own (10). "
                        "Set 1 to score a goal the instant the residual crosses the "
                        "threshold, which is what you want when checking whether "
                        "near-misses are real rather than a hold-duration failure. "
                        "It also changes the reward: reach_goal_bonus is amortised as "
                        "bonus/success_steps per near-goal step.")
    p.add_argument("--expl-coef", type=float, default=0.0,
                   help="SAPG exploration coefficient. Training spans 50->0 across env blocks; "
                        "0 is the greedy end and the right one for watching a policy.")
    p.add_argument("--device", default="cuda:0")
    return p


def infer_run_dir(checkpoint: str) -> pathlib.Path:
    """``<run>/rank_0/<name>/nn/<ckpt>.pth`` -- the run is four levels up."""
    return pathlib.Path(checkpoint).resolve().parents[3]


def assemble_population(ref: str, args) -> tuple[str, list[dict]]:
    """One env per design: the chosen indices, plus anything from --include.

    Returns ``(population ref, labels)``. A multi-design population cannot be
    loaded into fewer envs -- the env refuses, rightly, because the rest would
    never be stepped -- so pick explicitly rather than silently truncating.
    """
    from hand_sampler import population_io, robot_spec

    bare, hand_only = robot_spec.split_handonly(ref)
    hands, labels = [], []
    if robot_spec.is_population_file(bare):
        pop = population_io.load_population(bare)
        for i in [int(x) for x in args.designs.split(",") if x.strip() != ""]:
            h = pop[i % len(pop)]
            hands.append(h)
            labels.append({"name": f"design {i % len(pop)}", "fingers": h.n_fingers, "joints": h.n_joints})
    else:
        return ref, [{"name": bare, "fingers": -1, "joints": -1}]

    for extra in [x for x in args.include.split(",") if x.strip()]:
        e_bare, _ = robot_spec.split_handonly(extra)
        for h in population_io.load_population(e_bare):
            hands.append(h)
            labels.append({"name": pathlib.Path(e_bare).stem, "fingers": h.n_fingers, "joints": h.n_joints})

    out = pathlib.Path(tempfile.mkdtemp(prefix="play_pop_")) / "watch.json"
    population_io.save_population(hands, out, name="watch", provenance={"source": bare})
    return ("handonly:" if hand_only else "") + str(out), labels


def _viewing_urdf(hand, pop_ref: str, out: pathlib.Path) -> pathlib.Path:
    """A URDF carrying every joint the trajectory animates.

    urdf_for_viewing writes the hand's 30 envelope slots and is rooted at
    ARM_TIP_LINK. With an arm that is not enough: the trajectory is
    joint_names_canonical, 37 names, and ViserUrdf raises KeyError on the first
    one the URDF lacks -- 'iiwa14_joint_1'. pose_viewer._graft_hand_onto_arm
    hangs the hand off the arm's flange and supplies exactly those 7.

    With no arm the hand URDF already IS the robot (the stub carries link_7 at
    the articulation root), so the graft is skipped -- the same branch
    pose_viewer takes.
    """
    from coevolution.pose_viewer import ARM_URDF_PATH, _graft_hand_onto_arm
    from hand_sampler import build, robot_spec

    text = build.urdf_for_viewing(hand, out).read_text(encoding="utf-8")
    if robot_spec.population_from_ref(pop_ref).spec.num_arm_joints:
        text = _graft_hand_onto_arm(text, ARM_URDF_PATH)
        # The grafted arm links keep the VENDOR urdf's relative <mesh filename>
        # entries, and ViserUrdf resolves those against the urdf's own
        # directory -- which is a temp dir. Every arm link then loads nothing
        # and the arm is invisible while its joints still animate. Point them
        # at the real files. (The hand needs none of this: it is capsules and a
        # box, no meshes at all.)
        # Inlined rather than imported from scene_utils.author_robot: that
        # package pulls in pxr, and this process has no Kit.
        import xml.etree.ElementTree as ET

        root = ET.fromstring(text)
        mesh_root = str(pathlib.Path(ARM_URDF_PATH).parent) + "/"
        for mesh in root.iter("mesh"):
            fn = mesh.get("filename", "")
            if fn and not fn.startswith("/"):
                mesh.set("filename", mesh_root + fn)
        text = ET.tostring(root, encoding="unicode")
    out.write_text(text, encoding="utf-8")
    return out


def play(task: str, args, hydra_args=None) -> None:
    """Serve the viser page; the child does the physics."""
    import numpy as np
    import viser
    from viser.extras import ViserUrdf

    from hand_sampler import build, population_io, robot_spec

    if args.checkpoint:
        run_dir = pathlib.Path(args.run_dir) if args.run_dir else infer_run_dir(args.checkpoint)
        cfg_path = run_dir / "rank_0" / ".hydra" / "config.yaml"
        if not cfg_path.is_file():
            raise SystemExit(f"no saved config under {run_dir}; pass --run-dir explicitly")

        from omegaconf import OmegaConf

        ref = args.population or str(OmegaConf.load(cfg_path).env.assets.robot_spec)
    else:
        # Random policy: there is no run to replay, so the env is built from the
        # TASK's own config, which is what you want when checking inits -- it is
        # the task as currently configured, not as some past run configured it.
        run_dir = pathlib.Path("")
        if not args.population:
            raise SystemExit("--population is required when no --checkpoint is given")
        ref = args.population
        print("[play] NO CHECKPOINT: randomly initialised policy, task-default env",
              flush=True)
    pop_ref, labels = assemble_population(ref, args)
    print(f"[play] task {task}\n[play] run  {run_dir}\n[play] watching {len(labels)} design(s): "
          + ", ".join(f"{l['name']} ({l['fingers']}f/{l['joints']}j)" for l in labels), flush=True)

    bare, _ = robot_spec.split_handonly(pop_ref)
    hands = population_io.load_population(bare)
    urdf_dir = pathlib.Path(tempfile.mkdtemp(prefix="play_urdf_"))
    urdfs = [_viewing_urdf(h, pop_ref, urdf_dir / f"d{i}.urdf") for i, h in enumerate(hands)]

    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    host, port = listener.address
    child = subprocess.Popen(
        [sys.executable, str(_WORKER), "--task", task, "--checkpoint", args.checkpoint,
         "--run-dir", ("" if not args.checkpoint else str(run_dir)), "--population", pop_ref, "--num-envs", str(len(hands)),
         "--expl-coef", str(args.expl_coef), "--tolerance", str(args.tolerance),
         "--success-steps", str(args.success_steps), "--trace", args.trace,
         "--device", args.device, "--host", host, "--port", str(port),
         "--authkey", authkey.hex()] + list(hydra_args or []),
        env={**__import__("os").environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
    print("[play] Kit is booting in the child; this takes a minute", flush=True)
    conn = listener.accept()

    server = viser.ViserServer(port=args.port)
    print(f"\n[play] viser on http://{__import__('socket').gethostname()}:{args.port}\n", flush=True)

    with server.gui.add_folder("design"):
        picker = server.gui.add_dropdown(
            "watching", tuple(f"{l['name']} ({l['fingers']}f/{l['joints']}j)" for l in labels))
    with server.gui.add_folder("playback"):
        b_pause = server.gui.add_button("pause / resume")
        b_step = server.gui.add_button("single step")
        b_reset = server.gui.add_button("reset episode")
        b_goal = server.gui.add_button("resample goal")
        g_det = server.gui.add_checkbox("deterministic action", False)
        # Overrides the policy entirely: every joint is driven to 0. Not the same
        # as a zero action, which commands the midpoint of each joint's range --
        # equal to 0 only for a hand whose limits are symmetric.
        g_zero = server.gui.add_checkbox("hold joints at 0 (ignore policy)", False)
    with server.gui.add_folder("goal probe"):
        # Walk up to the success condition by hand. The angle is the quantity the
        # metric ought to be a function of; the axis sliders are here because it is
        # not -- the four reward keypoints are two antipodal pairs, all coplanar, so
        # for a fixed angle the residual varies by sqrt(3) with the axis and the
        # threshold runs 5.74 deg on the most sensitive axis to 9.94 deg on the
        # least. Sweep "axis polar" at a fixed angle near 6 deg to watch near-goal
        # flicker on and off without the angle changing at all.
        g_manual = server.gui.add_checkbox("drive object by hand (freezes physics)", False)
        g_kp = server.gui.add_checkbox("draw reward keypoints", True)
        s_deg = server.gui.add_slider("angle from goal (deg)", 0.0, 180.0, 0.1, 10.0)
        s_pol = server.gui.add_slider("axis polar (deg)", 0.0, 180.0, 1.0, 90.0)
        s_azi = server.gui.add_slider("axis azimuth (deg)", -180.0, 180.0, 1.0, 0.0)
    with server.gui.add_folder("status"):
        md = server.gui.add_markdown("waiting for Kit...")

    # GUI callbacks run on viser's own threads while the main loop is inside
    # conn.recv(). multiprocessing.Connection is NOT thread-safe, so sending
    # from a callback races the read and the command is silently lost -- which
    # is exactly what "pause does nothing" looks like. Queue instead, and let
    # the one thread that owns the connection do every send.
    import queue

    outbox: "queue.Queue[dict]" = queue.Queue()

    # The CHILD owns the running flag. Mirroring it here and sending
    # pause/resume from the parent's copy meant two owners of one piece of
    # state: b_step flipped the parent's without telling the child, and any
    # double-fired handler toggled twice, so every other click became a no-op --
    # "pause works, resume does not".
    @b_pause.on_click
    def _(_):
        outbox.put({"cmd": "toggle"})

    @b_step.on_click
    def _(_):
        outbox.put({"cmd": "step"})

    @b_reset.on_click
    def _(_):
        outbox.put({"cmd": "reset"})

    @b_goal.on_click
    def _(_):
        outbox.put({"cmd": "resample"})

    @g_det.on_update
    def _(_):
        outbox.put({"cmd": "deterministic", "value": bool(g_det.value)})

    @g_zero.on_update
    def _(_):
        outbox.put({"cmd": "hold_zero", "value": bool(g_zero.value)})

    def _push_manual(_=None, arm=False):
        # TOUCHING A SLIDER ARMS THE PROBE. Requiring the checkbox first made every
        # slider silently inert, which is indistinguishable from the probe being
        # broken -- the worker logged "goal probe off" on every drag.
        if arm and not g_manual.value:
            g_manual.value = True        # fires this handler again, with arm False
            return
        outbox.put({"cmd": "manual", "value": bool(g_manual.value),
                    "deg": float(s_deg.value), "polar": float(s_pol.value),
                    "azim": float(s_azi.value)})

    g_manual.on_update(lambda _=None: _push_manual())
    for _w in (s_deg, s_pol, s_azi):
        _w.on_update(lambda _=None: _push_manual(arm=True))

    kp_handles: dict = {}
    viser_urdf, joint_names, shown = None, None, -1
    last_print = [0.0]
    prev_goal, prev_succ, prev_step = [None], [0], [None]
    object_urdfs: list = []
    # The robot base is not the env origin -- spec.base_pos puts it beside the
    # table -- so mount the arm under a frame the child positions.
    robot_root = server.scene.add_frame("/robot_root", show_axes=False)
    # The table the object sits on. Its pose is per-env and randomised at reset,
    # so the child sends it; the box size is the URDF's.
    # Hidden until the child actually reports a table pose. Created-but-never-
    # positioned left it sitting at the world origin, directly under the hand --
    # which is worse than not drawing it, because it looks deliberate.
    table = server.scene.add_box("/table", dimensions=(0.475, 0.4, 0.3), color=(150, 150, 155))
    table.visible = False
    # /object and /goal are FRAMES the real object URDF is mounted under, once
    # the child tells us which of the curated pool each env drew. A stand-in box
    # would misrepresent the task: the pool spans 16 mm markers to 290 mm
    # hammers, and the grasp is the whole point.
    obj = server.scene.add_frame("/object", show_axes=False)
    goal = server.scene.add_frame("/goal", show_axes=False)
    obj_urdf = goal_urdf = None
    obj_shown = -1
    try:
        while child.poll() is None:
            while True:                       # every send happens here
                try:
                    conn.send(outbox.get_nowait())
                except queue.Empty:
                    break
            if not conn.poll(timeout=0.05):
                continue
            # Drain everything pending: handle every non-frame message IN ORDER,
            # and keep only the LAST frame. The child outruns viser by orders of
            # magnitude, so rendering each frame means showing a backlog minutes
            # old. An earlier version overwrote one `msg` slot instead, which
            # silently ATE the one-shot `ready` handshake whenever frames
            # arrived in the same batch -- joint_names stayed None and the robot
            # was never drawn at all.
            batch = [conn.recv()]
            while conn.poll():
                batch.append(conn.recv())
            newest_frame = None
            for m in batch:
                if m.get("kind") == "frame":
                    newest_frame = m
                    continue
                kind = m.get("kind")
                if kind == "status":
                    md.content = m["text"]
                elif kind == "running":
                    try:
                        b_pause.label = "resume" if not m["value"] else "pause"
                    except Exception:
                        pass                      # older viser: label is read-only
                elif kind == "ready":
                    joint_names = m["joint_names"]
                    object_urdfs = m.get("object_urdfs") or []
                    robot_root.position = tuple(m.get("base_pos") or (0.0, 0.0, 0.0))
                    robot_root.wxyz = tuple(m.get("base_rot") or (1.0, 0.0, 0.0, 0.0))
                    md.content = "running"
                elif kind == "error":
                    md.content = "**worker failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                    print(m["text"], flush=True)
                    raise SystemExit(1)
            if newest_frame is None:
                continue
            msg = newest_frame

            i = list(picker.options).index(picker.value)
            if i != shown:
                if viser_urdf is not None:
                    server.scene.remove_by_name("/robot_root/robot")
                viser_urdf = ViserUrdf(server, urdfs[i], root_node_name="/robot_root/robot")
                shown = i
            if i != obj_shown and i < len(object_urdfs) and object_urdfs[i]:
                for name in ("/object/mesh", "/goal/mesh"):
                    try:
                        server.scene.remove_by_name(name)
                    except Exception:
                        pass
                obj_urdf = ViserUrdf(server, pathlib.Path(object_urdfs[i]),
                                     root_node_name="/object/mesh")
                # The goal is the SAME geometry as a translucent green ghost, so
                # the target pose reads as "put the tool here" rather than as an
                # unrelated marker -- orientation included, which a sphere or an
                # axis triad cannot show.
                goal_urdf = ViserUrdf(server, pathlib.Path(object_urdfs[i]),
                                      root_node_name="/goal/mesh",
                                      mesh_color_override=GOAL_GHOST_RGBA)
                obj_shown = i
            if joint_names and viser_urdf is not None:
                q = msg["joint_pos"][i]
                viser_urdf.update_cfg(dict(zip(joint_names, q)))
            obj.position = tuple(msg["object_pos"][i])
            obj.wxyz = tuple(msg["object_quat"][i])
            _draw_keypoints(server, kp_handles, msg, i, bool(g_kp.value))
            goal.position = tuple(msg["goal_pos"][i])
            goal.wxyz = tuple(msg["goal_quat"][i])
            if msg.get("table_pos"):
                table.position = tuple(msg["table_pos"][i])
                table.wxyz = tuple(msg["table_quat"][i])
                table.visible = True
            md.content = _status(msg["stats"], i)

            # Also to stdout, throttled: the panel is fine for watching, but a
            # printed series is what you scroll back through to see whether the
            # residual is actually falling.
            now = time.time()
            if now - last_print[0] >= 0.5:
                st = msg["stats"]
                def _v(k):
                    v = st.get(k)
                    return None if not v else (v[i] if len(v) > i else v[0])
                dist, tol = _v("_keypoints_max_dist"), _v("_current_success_tolerance")
                best, succ = _v("_closest_keypoint_max_dist"), _v("_successes")
                step = _v("episode_length_buf")
                # Flag a goal change and an episode reset, so the print shows
                # WHEN the env resampled rather than only the residual. A goal
                # that changes without a success, or a step counter that rewinds
                # without either, is the signature of a reset bug.
                gq = tuple(round(float(x), 5) for x in msg["goal_quat"][i])
                sc = int(_v("_successes") or 0)
                tag = ""
                if prev_goal[0] is not None and gq != prev_goal[0]:
                    tag = "  <== GOAL RESAMPLED" + ("" if sc != prev_succ[0] else " (no success!)")
                if prev_step[0] is not None and (step or 0) < prev_step[0]:
                    tag += "  <== EPISODE RESET"
                prev_goal[0], prev_succ[0], prev_step[0] = gq, sc, (step or 0)
                if dist is not None:
                    thr = (tol or 0.0) * 1.5          # tolerance * keypoint_scale
                    ang = 2 * math.degrees(math.asin(min(dist / (2 * KP_RADIUS_M), 1.0)))
                    print(f"[dist] step {int(step or 0):4d}  rot residual {1000*dist:7.2f} mm "
                          f"({ang:5.1f} deg)  best {1000*(best or 0):7.2f} mm  "
                          f"threshold {1000*thr:5.2f} mm  successes {int(succ or 0)}{tag}", flush=True)
                last_print[0] = now
    except (EOFError, ConnectionResetError):
        print("[play] worker closed the connection", flush=True)
    finally:
        try:
            conn.send({"cmd": "quit"})
        except Exception:
            pass
        time.sleep(1.0)
        if child.poll() is None:
            child.kill()


# Object keypoints blue, goal keypoints green -- the same green as the goal ghost.
KP_OBJ_RGB = (60, 130, 246)
KP_GOAL_RGB = (64, 217, 102)


def _draw_keypoints(server, handles: dict, msg: dict, i: int, show: bool) -> None:
    """Draw the two keypoint sets the reward compares, and the residual between them.

    Both sets share the object's centre, because that is what the metric does: it
    refers each set to its own origin before subtracting, so only orientation is
    scored. The line lengths therefore ARE the per-keypoint residual, and the
    longest line is _keypoints_max_dist -- the number the tolerance is compared to.
    """
    import numpy as np

    okp, gkp = msg.get("obj_kp"), msg.get("goal_kp")
    if not show or not okp or not gkp:
        for h in handles.values():
            try:
                h.visible = False
            except Exception:
                pass
        return
    o = np.asarray(okp[i], dtype=np.float32)
    g = np.asarray(gkp[i], dtype=np.float32)
    n = len(o)
    want = {
        "obj": (o, np.tile(np.array(KP_OBJ_RGB, dtype=np.uint8), (n, 1))),
        "goal": (g, np.tile(np.array(KP_GOAL_RGB, dtype=np.uint8), (n, 1))),
    }
    for key, (pts, cols) in want.items():
        h = handles.get(key)
        if h is None:
            handles[key] = server.scene.add_point_cloud(
                f"/keypoints_{key}", points=pts, colors=cols, point_size=0.006)
        else:
            h.points = pts
            h.visible = True
    # One segment per keypoint pair. add_line_segments is not in every viser
    # version, so fall back to leaving the clouds alone rather than dying here.
    seg = np.stack([o, g], axis=1)
    h = handles.get("res")
    try:
        if h is None:
            handles["res"] = server.scene.add_line_segments(
                "/keypoints_residual", points=seg,
                colors=np.tile(np.array((240, 90, 90), dtype=np.uint8), (n, 2, 1)),
                line_width=3.0)
        else:
            h.points = seg
            h.visible = True
    except Exception:
        handles["res"] = None


def _status(stats: dict, i: int) -> str:
    def v(key, fmt="{:.4f}"):
        s = stats.get(key)
        if not s:
            return "-"
        return fmt.format(s[i] if len(s) > i else s[0])
    # mm and degrees, because the raw metres are hard to read against a 5.85 mm
    # threshold, and degrees is the quantity anyone actually reasons about. The
    # inverse of residual = 2*r*sin(theta/2) on the most sensitive axis, so it is a
    # LOWER bound on the true angle: a residual can also come from a bigger rotation
    # about a less sensitive axis.
    def ang(key):
        s = stats.get(key)
        if not s:
            return "-"
        d = s[i] if len(s) > i else s[0]
        x = d / (2.0 * KP_RADIUS_M)
        return "{:.2f} deg".format(math.degrees(2.0 * math.asin(min(max(x, -1.0), 1.0))))

    def mm(key):
        s = stats.get(key)
        if not s:
            return "-"
        return "{:.2f} mm".format(1000.0 * (s[i] if len(s) > i else s[0]))

    return (f"**threshold** {mm('_tol_effective')} = {ang('_tol_effective')}  \n"
            f"**residual** {mm('_keypoints_max_dist')} = {ang('_keypoints_max_dist')} "
            f"(>= this angle)  \n"
            f"**is_success** {v('_is_success', '{:.0f}')}  \n"
            f"**raw tolerance** {v('_current_success_tolerance')} (x keypoint_scale)  \n"
            f"**keypoint dist** {v('_keypoints_max_dist')} m  \n"
            f"**best so far** {v('_closest_keypoint_max_dist')} m  \n"
            f"**near goal** {v('_near_goal', '{:.0f}')}  |  "
            f"**successes** {v('_successes', '{:.0f}')}  \n"
            f"**lifted** {v('_lifted_object', '{:.0f}')}  |  "
            f"**step** {v('episode_length_buf', '{:.0f}')}")
