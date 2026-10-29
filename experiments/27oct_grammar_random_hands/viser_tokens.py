"""What the joint transformer sees: one hand in the pen-repose env, its policy observation cut into tokens exactly as
the network cuts it, drawn over the simulated hand. CPU viewer + a Kit child (one env).

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/viser_tokens.py \
        --hand assets/urdf/27oct_grammar_random_hands/coarse_s0_000/coarse_s0_000.spec.json [--port 8096]

The child runs the olympics env with the repose settings the runs use (pen_repose, box pen, uniform goals, palm
rotation locked) and sends env 0's flat policy observation every step. The viewer cuts it with the network's own
layout (``isaacsimenvs.common.obs_layout.build_token_layout`` with the 6 wrist joints as arm joints and
task_onehot, as ``coevolution/networks/joint_transformer.py`` builds it) and decodes each field back into the world:

  hand-joint tokens   joint_link_bbox: 4 points (a corner, then the ends of its edges along the link's "distal"
                      direction, the bend direction and the joint axis), in the palm frame from the palm centre ->
                      drawn as a wireframe box (distal edge thick); joint_pos / prev_joint_pos (normalised to
                      [-1, 1] by joint_lower / joint_upper), velocities, previous targets
  global token        the 6 wrist joints' 5 fields; keypoints_rel_ee (the pen's 4 keypoints from the palm centre, palm
                      frame) and keypoints_rel_goal -> drawn as spheres on the pen and on the goal ghost; object_vel;
                      palm_extents; task_onehot

and checks every decoded quantity against the simulator's own state (sent alongside): each token's joint position
(de-normalised) against the sim joint of that name, its box's joint origin against the sim joint's child-body
origin, the keypoints against the sim pen and goal, the wrist joints. The check table is the first thing to read.

Box coverage: the token box summarises the link (hand_sampler.design_space._ordered_box: its near face at the joint
along "distal", centred on the joint along the axis); the panel shows how much of the link's real collision geometry
falls outside it (computed from the URDF, Kit-free) -- the rendered hand shows the same thing.
"""
from __future__ import annotations

import argparse
import os
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

REPOSE_OVERRIDES = dict(goal_sampling="uniform", random_first_goal=True)
FINGER_COLOURS = [(230, 80, 60), (60, 160, 230), (90, 190, 90), (230, 170, 40), (170, 90, 220), (40, 200, 190),
                  (200, 120, 160)]


# ------------------------------------------------------------------------------------------------ the Kit child

def _make_env(hand: str):
    import isaacsimenvs.inhand_olympics  # noqa: F401
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg
    cfg = InHandOlympicsEnvCfg()
    cfg.hand.robot_spec = hand
    cfg.scene.num_envs = 1
    cfg.task.names = ("pen_repose",)
    cfg.pen.shape = "box"
    cfg.pen_repose.goal_sampling = REPOSE_OVERRIDES["goal_sampling"]
    cfg.pen_repose.random_first_goal = REPOSE_OVERRIDES["random_first_goal"]
    cfg.goal.first_goal_tolerance = 0.03
    cfg.action.wrist_max_speed_ang = 0.0
    cfg.goal.seconds_per_goal = 1e6           # no timeouts while someone is looking
    cfg.reset.stagger_initial_episodes = False
    env = InHandOlympicsEnv(cfg)
    env.reset()
    return env


def worker(args) -> None:
    from viser_hand_table import _boot     # isaacsimenvs/inhand_olympics/tests
    app = _boot()
    conn = Client((args.host, args.wport), authkey=bytes.fromhex(args.authkey))
    try:
        _loop(args, conn)
    except Exception:
        import traceback
        conn.send({"kind": "error", "text": traceback.format_exc()})
    conn.close()
    app.close()
    os._exit(0)


def _loop(args, conn) -> None:
    import torch
    from isaaclab.utils.math import quat_mul

    from isaacsimenvs.common.obs_geometry import apply_local_offset
    from isaacsimenvs.inhand_olympics.goals import keypoints
    env = _make_env(args.hand)
    names = list(env.robot.joint_names)
    canon = [names[int(k)] for k in env.perm_lab_to_canon.tolist()]
    obs_list = list(env.cfg.obs.obs_list)
    conn.send({"kind": "ready", "urdf": str(env.wrist_urdf_path), "joints": names, "canon": canon,
               "obs_list": obs_list, "table_top": env.cfg.table.top_height, "table_size": list(env.cfg.table.size),
               "pen": [env.cfg.pen.length, env.cfg.pen.diameter],
               "link_bodies": list(env.robot_spec.joint_link_bodies), "joint_box": env.cfg.obs.joint_box,
               "lower": env.joint_lower[0].tolist(), "upper": env.joint_upper[0].tolist()})
    targets = env.robot.data.default_joint_pos.clone()
    lo, hi = env.joint_lower[0], env.joint_upper[0]
    paused, t_next = False, time.time()
    body_ids = env._joint_link_body_ids
    while True:
        step = not paused
        while conn.poll():
            m = conn.recv()
            if m["cmd"] == "quit":
                env.close()
                return
            if m["cmd"] == "target":
                targets[0, names.index(m["joint"])] = float(m["value"])
            elif m["cmd"] == "random_fingers":
                hand_ids = [names.index(n) for n in canon[6:]]
                u = torch.rand(len(hand_ids), device=env.device)
                targets[0, hand_ids] = lo[hand_ids] + u * (hi[hand_ids] - lo[hand_ids])
            elif m["cmd"] == "reset":
                env.reset()
                targets = env.robot.data.default_joint_pos.clone()
            elif m["cmd"] == "pause":
                paused = bool(m["value"])
            elif m["cmd"] == "step":
                step = True
        if step:
            env.step(env.action_toward(targets))
        obs = env._get_observations()["policy"][0]
        o = env.scene.env_origins[0]
        palm = env.robot.data.body_state_w[0:1, env.palm_body]
        centre = apply_local_offset(palm[:, 0:3], palm[:, 3:7], env.palm_center_offset[0:1], (1,))[0] - o
        palm_rot = quat_mul(palm[:, 3:7], env.palm_frame_quat[0:1])[0]
        pen_pos, pen_quat = env.pen_pose()
        conn.send({"kind": "frame", "obs": obs.tolist(), "q": env.robot.data.joint_pos[0].tolist(),
                   "centre": centre.tolist(), "palm_rot": palm_rot.tolist(),
                   "link_origins": (env.robot.data.body_state_w[0, body_ids, 0:3] - o).tolist(),
                   "link_quats": env.robot.data.body_state_w[0, body_ids, 3:7].tolist(),
                   "pen_pos": pen_pos[0].tolist(), "pen_quat": pen_quat[0].tolist(),
                   "goal_pos": env.goal_pos[0].tolist(), "goal_quat": env.goal_quat[0].tolist(),
                   "kp": keypoints(pen_pos, pen_quat, env.pen_offsets)[0].tolist(),
                   "kp_goal": keypoints(env.goal_pos, env.goal_quat, env.pen_offsets)[0].tolist(),
                   "successes": int(env.successes[0])})
        t_next += float(env.step_dt)
        time.sleep(max(0.0, t_next - time.time()))
        t_next = max(t_next, time.time() - 0.1)


# ------------------------------------------------------------------------------------------------ Kit-free helpers

def token_layout(hand: str, obs_list: list[str]) -> dict:
    """The network's layout (joint_transformer._build_layout with olympics' extra_arm_joints / extra_field_sizes)."""
    from coevolution.networks.joint_transformer import _WithArmJoints
    from isaacsimenvs.common.obs_layout import build_token_layout
    from isaacsimenvs.common.robots import get_robot_spec
    spec = _WithArmJoints(get_robot_spec(hand), 6)
    return build_token_layout(spec, obs_list, {"task_onehot": 8})


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def box_corners(pts: np.ndarray) -> np.ndarray:
    """(4, 3) [corner, corner + e1, corner + e2, corner + e3] -> (8, 3) corners."""
    p0, e = pts[0], pts[1:] - pts[0]
    return np.array([p0 + a * e[0] + b * e[1] + c * e[2] for a in (0, 1) for b in (0, 1) for c in (0, 1)])


BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def local_boxes(spec_path: str, mode: str) -> np.ndarray:
    """(J, 4, 3) the token boxes in each child link's frame, as the env builds them for ``mode``."""
    from hand_sampler.design_space import joint_link_boxes
    from isaacsimenvs.common.robots import get_robot_spec
    rs = get_robot_spec(spec_path)
    if mode == "ordered":
        return np.asarray(rs.joint_link_boxes, float)
    return joint_link_boxes(rs.urdf_path, rs.hand_joint_names, mode=mode, capsules=rs.replace_cylinders_with_capsules)[1]


def coverage(spec_path: str, mode: str) -> dict[str, tuple[float, float]]:
    """joint -> (mm of the link's collision geometry outside its token box, IoU), from the URDF (Kit-free)."""
    import xml.etree.ElementTree as ET

    from hand_sampler import design_space as ds
    from isaacsimenvs.common.robots import get_robot_spec
    rs = get_robot_spec(spec_path)
    urdf = REPO / rs.urdf_path
    root = ET.parse(urdf).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = {j.get("name"): j for j in root.findall("joint")}
    out = {}
    boxes = local_boxes(spec_path, mode)
    for k, jn in enumerate(rs.hand_joint_names):
        cols = links[joints[jn].find("child").get("link")].findall("collision")
        pieces = [ds._collision_vertices(c, urdf.parent) for c in cols]
        for c in cols:              # a cylinder the importer turns into a capsule: add its caps
            cyl = c.find("geometry/cylinder")
            if cyl is not None and rs.replace_cylinders_with_capsules:
                r, L, O = float(cyl.get("radius")), float(cyl.get("length")), ds._origin(c.find("origin"))
                pieces.append(np.array([[0, 0, L / 2 + r], [0, 0, -L / 2 - r]]) @ O[:3, :3].T + O[:3, 3])
        V = np.concatenate([p for p in pieces if p.size]) if pieces else np.zeros((0, 3))
        if not len(V):
            out[jn] = (float("nan"), float("nan"))
            continue
        B = boxes[k]
        p0, e = B[0], B[1:] - B[0]
        n = np.linalg.norm(e, axis=1)
        loc = (V - p0) @ (e / n[:, None]).T
        miss = float((np.maximum(0, -loc) + np.maximum(0, loc - n)).max())
        lo, hi = loc.min(0), loc.max(0)
        inter = np.prod(np.clip(np.minimum(hi, n) - np.maximum(lo, 0), 0, None))
        out[jn] = (1e3 * miss, float(inter / (np.prod(n) + np.prod(hi - lo) - inter)))
    return out


# ------------------------------------------------------------------------------------------------ the viewer

def main(args) -> None:
    import signal

    import viser
    from viser_hands import Hand, _wxyz

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    host, wport = listener.address
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "--hand", args.hand,
                             "--host", host, "--wport", str(wport), "--authkey", authkey.hex()],
                            cwd=REPO, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
    sys.path.insert(0, str(REPO / "isaacsimenvs/inhand_olympics/tests"))
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("+z")
    print(f"\n[tokens] open http://{socket.gethostname()}:{args.port}  (the sim boots in about a minute)\n", flush=True)
    status = server.gui.add_markdown("booting the sim ...")
    conn_q: "queue.Queue" = queue.Queue()
    threading.Thread(target=lambda: conn_q.put(listener.accept()), daemon=True).start()
    try:
        while True:
            try:
                conn = conn_q.get(timeout=1.0)
                break
            except queue.Empty:
                if proc.poll() is not None:
                    raise SystemExit("the sim child exited while booting")
        ready = conn.recv()
        if ready["kind"] == "error":
            raise SystemExit(ready["text"])
        _run(server, conn, ready, args, status, Hand, _wxyz)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            conn.send({"cmd": "quit"})
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def _run(server, conn, ready, args, status, Hand, _wxyz) -> None:
    layout = token_layout(args.hand, ready["obs_list"])
    names, canon = ready["joints"], ready["canon"]
    hand_joints = layout["hand_joint_names"]
    assert canon[6:] == hand_joints, (canon[6:], hand_joints)
    fs = layout["token_field_slices"]
    from isaacsimenvs.common.obs_layout import field_offsets
    from coevolution.networks.joint_transformer import _WithArmJoints
    from isaacsimenvs.common.robots import get_robot_spec
    spec = get_robot_spec(args.hand)
    offs = field_offsets(ready["obs_list"], _WithArmJoints(spec, 6), {"task_onehot": 8})
    mode = ready["joint_box"]
    cover = coverage(args.hand, mode)
    expect_local = local_boxes(args.hand, mode)
    hand = Hand(Path(ready["urdf"]), capsules=spec.replace_cylinders_with_capsules)
    top = ready["table_top"]
    finger_of = {j: (int(j[1:].split("_")[0]) if j.startswith("f") else -1) for j in hand_joints}

    # scene: table, the hand (wrist URDF, base on the table top), pen, goal ghost
    sx, sy, sz = ready["table_size"]
    server.scene.add_box("/table", dimensions=(sx, sy, sz), color=(150, 120, 90), position=(0, 0, top - sz / 2),
                         opacity=0.6)
    L, D = ready["pen"]
    pen = server.scene.add_box("/pen", dimensions=(L, D, D), color=(40, 90, 200), opacity=0.5)
    goal = server.scene.add_box("/goal", dimensions=(L, D, D), color=(40, 200, 90), opacity=0.25)
    server.scene.add_frame("/robot", position=(0, 0, top), show_axes=False)
    frames = {}
    for link, T in hand.link_poses({}).items():
        frames[link] = server.scene.add_frame(f"/robot/{link}", show_axes=False)
    for k, (link, m, O, rgb) in enumerate(hand.visuals):
        server.scene.add_mesh_simple(f"/robot/{link}/v{k}", vertices=np.asarray(m.vertices, np.float32),
                                     faces=np.asarray(m.faces, np.uint32), color=rgb, opacity=0.55,
                                     wxyz=_wxyz(O[:3, :3]), position=tuple(O[:3, 3]))

    # GUI
    g_tok = server.gui.add_dropdown("token", tuple(hand_joints), initial_value=hand_joints[0])
    g_all = server.gui.add_checkbox("show every token's box", True)
    g_pause = server.gui.add_checkbox("pause the sim", False)
    b_step, b_rand, b_reset = (server.gui.add_button("one step"), server.gui.add_button("random finger targets"),
                               server.gui.add_button("reset"))
    md_tok = server.gui.add_markdown("")
    with server.gui.add_folder("global token", expand_by_default=False):
        md_glob = server.gui.add_markdown("")
    with server.gui.add_folder("joint targets", expand_by_default=False):
        sliders = {}
        for n in canon:
            k = names.index(n)
            lo, hi = ready["lower"][k], ready["upper"][k]
            sl = server.gui.add_slider(n, lo, hi, (hi - lo) / 200 or 1e-3, min(max(0.0, lo), hi) if n != "wrist_z"
                                       else 0.15)
            sl.on_update(lambda _e, n=n, sl=sl: conn.send({"cmd": "target", "joint": n, "value": float(sl.value)}))
            sliders[n] = sl
    g_pause.on_update(lambda _e: conn.send({"cmd": "pause", "value": bool(g_pause.value)}))
    b_step.on_click(lambda _e: conn.send({"cmd": "step"}))
    b_rand.on_click(lambda _e: conn.send({"cmd": "random_fingers"}))
    b_reset.on_click(lambda _e: conn.send({"cmd": "reset"}))

    def tok_fields(obs, j):
        cols = layout["token_columns"][j]
        t = obs[cols]
        return {f: t[a:b] for f, (a, b) in fs.items()}

    def glob_field(obs, f):
        a, b = offs[f]
        return obs[a:b]

    last, printed = 0.0, [0.0]
    while True:
        m = conn.recv()
        if m["kind"] == "error":
            status.content = f"**sim error**\n```\n{m['text'][-1500:]}\n```"
            return
        if time.time() - last < 0.1:          # draw at 10 Hz
            continue
        last = time.time()
        obs = np.asarray(m["obs"])
        q = dict(zip(names, m["q"]))
        for link, T in hand.link_poses(q).items():
            frames[link].wxyz, frames[link].position = _wxyz(T[:3, :3]), tuple(T[:3, 3])
        pen.wxyz, pen.position = tuple(m["pen_quat"]), tuple(m["pen_pos"])
        goal.wxyz, goal.position = tuple(m["goal_quat"]), tuple(m["goal_pos"])
        c, R = np.asarray(m["centre"]), quat_to_R(m["palm_rot"])
        to_world = lambda p: c + np.asarray(p).reshape(-1, 3) @ R.T          # noqa: E731

        # checks: every decoded quantity against the sim
        err_q, err_origin, err_box = 0.0, 0.0, 0.0
        sel = hand_joints.index(g_tok.value)
        for j, jn in enumerate(hand_joints):
            t = tok_fields(obs, j)
            lo, hi = float(t["joint_lower"][0]), float(t["joint_upper"][0])
            q_tok = lo + (float(t["joint_pos"][0]) + 1) / 2 * (hi - lo)
            err_q = max(err_q, abs(q_tok - q[jn]))
            pts = to_world(t["joint_link_bbox"])
            link_o, link_R = np.asarray(m["link_origins"][j]), quat_to_R(m["link_quats"][j])
            # the joint's position: its own field, or (ordered boxes, no joint_origin) the box's centre line
            origin_tok = (to_world(t["joint_origin"])[0] if "joint_origin" in t else
                          pts[0] + 0.5 * ((pts[2] - pts[0]) + (pts[3] - pts[0])))
            err_origin = max(err_origin, float(np.linalg.norm(origin_tok - link_o)))
            # the box: the expected box of this mode, placed at the sim's link pose
            err_box = max(err_box, float(np.abs(pts - (expect_local[j] @ link_R.T + link_o)).max()))
            show = g_all.value or j == sel
            corners = box_corners(pts)
            col = FINGER_COLOURS[finger_of[jn] % len(FINGER_COLOURS)] if finger_of[jn] >= 0 else (120, 120, 120)
            segs = np.array([[corners[a], corners[b]] for a, b in BOX_EDGES])
            server.scene.add_line_segments(f"/tok/{jn}/box", points=segs.astype(np.float32),
                                           colors=np.array(col, np.uint8), line_width=4.0 if j == sel else 1.5,
                                           visible=show)
            server.scene.add_line_segments(f"/tok/{jn}/distal", points=np.array([[pts[0], pts[1]]], np.float32),
                                           colors=np.array((255, 255, 255), np.uint8),
                                           line_width=6.0 if j == sel else 3.0, visible=show)
            if j == sel:
                server.scene.add_label("/tok/label", jn, position=tuple(pts[1]))
        kp_tok = to_world(glob_field(obs, "keypoints_rel_ee"))
        kp_goal_tok = np.asarray(m["kp"]) - np.asarray(glob_field(obs, "keypoints_rel_goal")).reshape(-1, 3)
        err_kp = float(np.abs(kp_tok - np.asarray(m["kp"])).max())
        err_kpg = float(np.abs(kp_goal_tok - np.asarray(m["kp_goal"])).max())
        server.scene.add_point_cloud("/tok/kp", points=kp_tok.astype(np.float32), colors=(255, 220, 0),
                                     point_size=0.008)
        server.scene.add_point_cloud("/tok/kp_goal", points=kp_goal_tok.astype(np.float32), colors=(0, 255, 120),
                                     point_size=0.008)
        wrist_tok = [obs[layout["global_slices"][f][0]: layout["global_slices"][f][1]] for f in range(5)]
        q_w = np.array([q[n] for n in canon[:6]])
        lo_w = np.array([ready["lower"][names.index(n)] for n in canon[:6]])
        hi_w = np.array([ready["upper"][names.index(n)] for n in canon[:6]])
        err_w = float(np.abs(lo_w + (wrist_tok[0] + 1) / 2 * (hi_w - lo_w) - q_w).max())
        extents = glob_field(obs, "palm_extents")

        def ok(e, tol):
            return f"{e:.1e} {'ok' if e < tol else '**MISMATCH**'}"
        status.content = (
            f"**checks (decoded token vs sim)**  \n"
            f"joint positions: {ok(err_q, 1e-4)} rad  \n"
            f"joint positions in space ({'joint_origin' if 'joint_origin' in fs else 'from the box'}): "
            f"{ok(err_origin, 1e-4)} m  \n"
            f"link boxes ({mode}) vs expected at the sim link pose: {ok(err_box, 1e-4)} m  \n"
            f"pen keypoints (global): {ok(err_kp, 1e-4)} m  \n"
            f"goal keypoints (global): {ok(err_kpg, 1e-4)} m  \n"
            f"wrist joints (global): {ok(err_w, 1e-4)}  \n"
            f"tokens {layout['n_hand']} x {layout['token_dim']}, global {layout['global_dim']}, "
            f"obs {layout['obs_dim']} = {len(obs)}; successes {m['successes']}")
        if time.time() - printed[0] > 10.0:       # the checks in the log too
            print("[tokens] " + status.content.replace("  \n", " | ").replace("**", ""), flush=True)
            printed[0] = time.time()
        t = tok_fields(obs, sel)
        jn = hand_joints[sel]
        miss, iou = cover[jn]
        rows = "\n".join(f"| {f} | {np.array2string(np.asarray(v), precision=3, suppress_small=True, max_line_width=60)} |"
                         for f, v in t.items() if f != "joint_link_bbox")
        e = (np.asarray(t["joint_link_bbox"]).reshape(4, 3)[1:] - np.asarray(t["joint_link_bbox"]).reshape(4, 3)[0])
        md_tok.content = (
            f"**{jn}** (token {sel}; child body `{ready['link_bodies'][sel]}`)  \n"
            f"sim q {q[jn]:+.3f} rad  \nbox edges: distal {1e3 * np.linalg.norm(e[0]):.0f} mm, "
            f"bend {1e3 * np.linalg.norm(e[1]):.0f} mm, axis {1e3 * np.linalg.norm(e[2]):.0f} mm  \n"
            f"link geometry outside the box: {miss:.1f} mm, IoU {iou:.2f}\n\n| field | value |\n|---|---|\n{rows}")
        md_glob.content = (
            f"wrist joint_pos (norm) {np.array2string(wrist_tok[0], precision=3)}  \n"
            f"palm_extents {np.array2string(extents, precision=3)} m  \n"
            f"object_vel {np.array2string(glob_field(obs, 'object_vel'), precision=2)}  \n"
            f"task_onehot {np.array2string(glob_field(obs, 'task_onehot'), precision=0)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="assets/urdf/27oct_grammar_random_hands/coarse_s0_000/coarse_s0_000.spec.json")
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    ap.add_argument("--wport", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--authkey", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        sys.path.insert(0, str(REPO / "isaacsimenvs/inhand_olympics/tests"))
        worker(a)
    else:
        main(a)
