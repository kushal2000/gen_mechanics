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


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--checkpoint", required=True, help="an rl_games .pth")
    p.add_argument("--run-dir", default="", help="training run dir; inferred from --checkpoint if omitted")
    p.add_argument("--population", default="", help="override the run's robot_spec")
    p.add_argument("--designs", default="0",
                   help="comma-separated design indices to load, one env each, e.g. 0,7,133. "
                        "They all run the same policy on the same goals, so the dropdown is a "
                        "like-for-like comparison.")
    p.add_argument("--include", default="", help="extra population .json to append, e.g. gen-SHARPA")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--tolerance", type=float, default=0.0, help="pin the success tolerance; 0 = the run's own")
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

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else infer_run_dir(args.checkpoint)
    cfg_path = run_dir / "rank_0" / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"no saved config under {run_dir}; pass --run-dir explicitly")

    from omegaconf import OmegaConf

    ref = args.population or str(OmegaConf.load(cfg_path).env.assets.robot_spec)
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
         "--run-dir", str(run_dir), "--population", pop_ref, "--num-envs", str(len(hands)),
         "--expl-coef", str(args.expl_coef), "--tolerance", str(args.tolerance),
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
    with server.gui.add_folder("status"):
        md = server.gui.add_markdown("waiting for Kit...")

    # GUI callbacks run on viser's own threads while the main loop is inside
    # conn.recv(). multiprocessing.Connection is NOT thread-safe, so sending
    # from a callback races the read and the command is silently lost -- which
    # is exactly what "pause does nothing" looks like. Queue instead, and let
    # the one thread that owns the connection do every send.
    import queue

    outbox: "queue.Queue[dict]" = queue.Queue()
    running = {"v": True}

    @b_pause.on_click
    def _(_):
        running["v"] = not running["v"]
        outbox.put({"cmd": "resume" if running["v"] else "pause"})

    @b_step.on_click
    def _(_):
        running["v"] = False
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

    viser_urdf, joint_names, shown = None, None, -1
    object_urdfs: list = []
    # The robot base is not the env origin -- spec.base_pos puts it beside the
    # table -- so mount the arm under a frame the child positions.
    robot_root = server.scene.add_frame("/robot_root", show_axes=False)
    # The table the object sits on. Its pose is per-env and randomised at reset,
    # so the child sends it; the box size is the URDF's.
    table = server.scene.add_box("/table", dimensions=(0.475, 0.4, 0.3), color=(150, 150, 155))
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
            msg = conn.recv()
            kind = msg.get("kind")
            if kind == "status":
                md.content = msg["text"]
                continue
            if kind == "error":
                md.content = "**worker failed**\n\n```\n" + msg["text"][-1500:] + "\n```"
                print(msg["text"], flush=True)
                break
            if kind == "ready":
                joint_names = msg["joint_names"]
                object_urdfs = msg.get("object_urdfs") or []
                robot_root.position = tuple(msg.get("base_pos") or (0.0, 0.0, 0.0))
                robot_root.wxyz = tuple(msg.get("base_rot") or (1.0, 0.0, 0.0, 0.0))
                md.content = "running"
                continue
            if kind != "frame":
                continue

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
            goal.position = tuple(msg["goal_pos"][i])
            goal.wxyz = tuple(msg["goal_quat"][i])
            if msg.get("table_pos"):
                table.position = tuple(msg["table_pos"][i])
                table.wxyz = tuple(msg["table_quat"][i])
            md.content = _status(msg["stats"], i)
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


def _status(stats: dict, i: int) -> str:
    def v(key, fmt="{:.4f}"):
        s = stats.get(key)
        if not s:
            return "-"
        return fmt.format(s[i] if len(s) > i else s[0])
    return (f"**tolerance** {v('_current_success_tolerance')}  \n"
            f"**keypoint dist** {v('_keypoints_max_dist')} m  \n"
            f"**best so far** {v('_closest_keypoint_max_dist')} m  \n"
            f"**near goal** {v('_near_goal', '{:.0f}')}  |  "
            f"**successes** {v('_successes', '{:.0f}')}  \n"
            f"**lifted** {v('_lifted_object', '{:.0f}')}  |  "
            f"**step** {v('episode_length_buf', '{:.0f}')}")
