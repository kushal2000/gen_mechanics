"""Zero-shot viewer: any ten-hands policy on any ten-hands hand, in viser.

Pick a HAND (the robot the env builds) and a POLICY (the hand whose run the checkpoint comes
from), then press "Load new environment". Matching pairs replay a policy on its own hand;
mismatched pairs are zero-shot transfer.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/old_experiments/30sep_ten_hands/viser_zero_shot.py --port 8080

WHY IT WORKS ACROSS HANDS. The joint transformer shares every weight across joint tokens,
so a checkpoint's tensors fit any hand except sigma, (exploration blocks x joints). The
worker's --cross-hand rebuilds the network for the target's joints and resizes sigma to the
target's joint count (each block's mean), which only matters for stochastic actions.

PHYSICS FOLLOWS THE HAND, not the policy: each target hand gets the joint speed cap its own
training run used (10 rad/s for the _v10 hands, the vendor URDF's otherwise), so a transfer
number is the policy meeting the same hand its own specialist met.

Kit and viser cannot share an interpreter (see coevolution/eval/viser_play.py), so each
environment is a child process running coevolution/eval/_play_worker.py; loading a new one
stops that child and starts another (about a minute while Kit boots).
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import queue
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import types
from multiprocessing.connection import Listener

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from coevolution.eval import viser_play as vp  # noqa: E402  (stdlib-only at import)

TASK = "GenMech-InHandReorient-Direct-v0"
LOGS = REPO / "debug_outputs/train_logs/30sep_ten_hands"
WORKER = REPO / "coevolution/eval/_play_worker.py"

# hand -> (robot_spec, study whose checkpoint is "that hand's policy", joint speed cap the hand
# trained with: 0 = the vendor URDF's). The four slow-joint hands use their 10 rad/s reruns.
HANDS = {
    "sharpa":     ("sharpa_handonly", "left_sharpa", 0.0),
    "gen_sharpa": (f"handonly:{REPO}/assets/populations/sharpa_capsule.json", "left_gen_sharpa", 0.0),
    "allegro":    ("allegro_left_handonly", "left_allegro_v10", 10.0),
    "leap":       ("leap_left_handonly", "left_leap_v10", 10.0),
    "shadow":     ("shadow_left_handonly", "left_shadow_v10", 10.0),
    "tesollo":    ("tesollo_left_handonly", "left_tesollo_v10", 10.0),
    "dex3":       ("dex3_left_handonly", "left_dex3", 0.0),
    "xhand":      ("xhand_left_handonly", "left_xhand", 0.0),
    "wuji2":      ("wuji2_left_handonly", "left_wuji2", 0.0),
}


def latest_checkpoint(study: str) -> pathlib.Path | None:
    """The newest .pth any run of this study has written (runs still training included)."""
    pths = glob.glob(f"{LOGS}/0_scale_train_{study}_c01_*/rank_0/*/nn/*.pth")
    return pathlib.Path(max(pths, key=os.path.getmtime)) if pths else None


class Session:
    """One Kit child: one hand, one policy."""

    def __init__(self, hand: str, policy: str, device: str):
        spec, _, vlim = HANDS[hand]
        ckpt = latest_checkpoint(HANDS[policy][1])
        if ckpt is None:
            raise FileNotFoundError(f"no checkpoint yet for the {policy} policy ({HANDS[policy][1]})")
        self.hand, self.policy, self.ckpt = hand, policy, ckpt
        self.run_dir = ckpt.parents[3]

        # A generated hand is a population: narrow it to design 0 and author the capsule URDF
        # here, as viser_play does. A registered hand's URDF arrives in the ready handshake.
        self.pop_ref, self.urdf = spec, None
        from hand_sampler import population_io, robot_spec
        if robot_spec.is_population_ref(spec):
            self.pop_ref, _ = vp.assemble_population(spec, types.SimpleNamespace(designs="0", include=""))
            bare, _ = robot_spec.split_handonly(self.pop_ref)
            d = pathlib.Path(tempfile.mkdtemp(prefix="zs_urdf_"))
            self.urdf = vp._viewing_urdf(population_io.load_population(bare)[0], self.pop_ref, d / "hand.urdf")

        authkey = secrets.token_bytes(16)
        self.listener = Listener(("127.0.0.1", 0), authkey=authkey)
        host, port = self.listener.address
        self.child = subprocess.Popen(
            [sys.executable, str(WORKER), "--task", TASK, "--checkpoint", str(ckpt),
             "--run-dir", str(self.run_dir), "--population", self.pop_ref, "--num-envs", "1",
             "--expl-coef", "0", "--cross-hand", "--hand-velocity-limit", str(vlim),
             "--device", device, "--host", host, "--port", str(port), "--authkey", authkey.hex()],
            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, cwd=str(REPO))
        self.conn = None

    def accept(self, timeout_s: float = 600.0) -> bool:
        """Wait for the child to connect; False if it died first."""
        self.listener._listener._socket.settimeout(2.0)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.child.poll() is not None:
                return False
            try:
                self.conn = self.listener.accept()
                return True
            except (socket.timeout, TimeoutError):
                continue
        return False

    def close(self) -> None:
        try:
            if self.conn is not None:
                self.conn.send({"cmd": "quit"})
        except Exception:
            pass
        for _ in range(20):
            if self.child.poll() is not None:
                break
            time.sleep(0.25)
        if self.child.poll() is None:
            self.child.kill()
            self.child.wait()
        try:
            self.listener.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hand", default="allegro", choices=list(HANDS))
    ap.add_argument("--policy", default="sharpa", choices=list(HANDS))
    args = ap.parse_args()

    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=args.port)
    print(f"\n[zero-shot] viser on http://{socket.gethostname()}:{args.port}\n", flush=True)

    with server.gui.add_folder("environment"):
        g_hand = server.gui.add_dropdown("hand", tuple(HANDS), initial_value=args.hand)
        g_policy = server.gui.add_dropdown("policy (trained on)", tuple(HANDS), initial_value=args.policy)
        b_load = server.gui.add_button("Load new environment")
        md_env = server.gui.add_markdown("")
    with server.gui.add_folder("playback"):
        b_pause = server.gui.add_button("pause / resume")
        b_step = server.gui.add_button("single step")
        b_reset = server.gui.add_button("reset episode")
        b_goal = server.gui.add_button("resample goal")
        g_det = server.gui.add_checkbox("deterministic action", True)
        g_kp = server.gui.add_checkbox("draw reward keypoints", False)
    with server.gui.add_folder("status"):
        md = server.gui.add_markdown("pick a hand and a policy, then Load new environment")

    outbox: "queue.Queue[dict]" = queue.Queue()     # GUI threads -> the loop that owns conn
    load_req: "queue.Queue[tuple[str, str]]" = queue.Queue()
    b_pause.on_click(lambda _: outbox.put({"cmd": "toggle"}))
    b_step.on_click(lambda _: outbox.put({"cmd": "step"}))
    b_reset.on_click(lambda _: outbox.put({"cmd": "reset"}))
    b_goal.on_click(lambda _: outbox.put({"cmd": "resample"}))
    g_det.on_update(lambda _: outbox.put({"cmd": "deterministic", "value": bool(g_det.value)}))
    b_load.on_click(lambda _: load_req.put((str(g_hand.value), str(g_policy.value))))

    robot_root = server.scene.add_frame("/robot_root", show_axes=False)
    obj = server.scene.add_frame("/object", show_axes=False)
    goal = server.scene.add_frame("/goal", show_axes=False)
    kp_handles: dict = {}
    sess: Session | None = None
    state: dict = {}

    def clear_scene():
        for name in ("/robot_root/robot", "/object/mesh", "/goal/mesh"):
            try:
                server.scene.remove_by_name(name)
            except Exception:
                pass
        for h in kp_handles.values():
            try:
                h.visible = False
            except Exception:
                pass

    def load(hand: str, policy: str):
        nonlocal sess
        if sess is not None:
            md.content = "stopping the previous environment..."
            sess.close()
            sess = None
        clear_scene()
        state.clear()
        kind = "own hand" if hand == policy else "**zero-shot**"
        try:
            sess = Session(hand, policy, args.device)
        except Exception as e:
            md.content = f"**cannot load**: {e}"
            return
        md_env.content = (f"**{policy}** policy on **{hand}** ({kind})  \n"
                          f"checkpoint `{sess.ckpt.name}`  \n"
                          f"joint speed cap: {HANDS[hand][2] or 'vendor URDF'}"
                          + (" rad/s" if HANDS[hand][2] else ""))
        md.content = "Kit is booting in the child; about a minute..."
        print(f"[zero-shot] {policy} policy -> {hand} hand  ({sess.ckpt})", flush=True)
        if not sess.accept():
            md.content = "**worker died before connecting** -- see the terminal"
            sess.close()
            sess = None
            return
        sess.conn.send({"cmd": "deterministic", "value": bool(g_det.value)})

    load_req.put((args.hand, args.policy))
    try:
        while True:
            try:
                req = load_req.get_nowait()
                while not load_req.empty():            # several clicks: only the last counts
                    req = load_req.get_nowait()
                load(*req)
                while not outbox.empty():              # commands meant for the old child
                    outbox.get_nowait()
            except queue.Empty:
                pass
            if sess is None or sess.conn is None:
                time.sleep(0.1)
                continue
            if sess.child.poll() is not None:
                md.content = "**worker exited** -- see the terminal; load again to retry"
                sess.close()
                sess = None
                continue
            try:
                while True:
                    try:
                        sess.conn.send(outbox.get_nowait())
                    except queue.Empty:
                        break
                if not sess.conn.poll(timeout=0.05):
                    continue
                batch = [sess.conn.recv()]
                while sess.conn.poll():
                    batch.append(sess.conn.recv())
            except (EOFError, ConnectionResetError, BrokenPipeError, OSError):
                md.content = "**worker closed the connection** -- load again to retry"
                sess.close()
                sess = None
                continue

            frame = None
            for m in batch:
                k = m.get("kind")
                if k == "frame":
                    frame = m
                elif k == "status":
                    md.content = m["text"]
                elif k == "error":
                    md.content = "**worker failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                    print(m["text"], flush=True)
                elif k == "ready":
                    state["joint_names"] = m["joint_names"]
                    urdf = sess.urdf or pathlib.Path(m["urdf_path"])
                    robot_root.position = tuple(m.get("base_pos") or (0.0, 0.0, 0.0))
                    robot_root.wxyz = tuple(m.get("base_rot") or (1.0, 0.0, 0.0, 0.0))
                    state["robot"] = ViserUrdf(server, urdf, root_node_name="/robot_root/robot")
                    ou = (m.get("object_urdfs") or [None])[0]
                    if ou:
                        ViserUrdf(server, pathlib.Path(ou), root_node_name="/object/mesh")
                        boxes = vp._goal_boxes_from_urdf(pathlib.Path(ou))
                        if boxes:
                            for j, (dims, rgb, xyz, wxyz) in enumerate(boxes):
                                server.scene.add_box(f"/goal/mesh/v{j}", color=rgb, dimensions=dims,
                                                     opacity=vp.GOAL_OPACITY, position=xyz, wxyz=wxyz)
                        else:
                            ViserUrdf(server, pathlib.Path(ou), root_node_name="/goal/mesh",
                                      mesh_color_override=vp.GOAL_GHOST_RGBA)
            if frame is None or "robot" not in state:
                continue
            state["robot"].update_cfg(dict(zip(state["joint_names"], frame["joint_pos"][0])))
            obj.position, obj.wxyz = tuple(frame["object_pos"][0]), tuple(frame["object_quat"][0])
            goal.position, goal.wxyz = tuple(frame["goal_pos"][0]), tuple(frame["goal_quat"][0])
            vp._draw_keypoints(server, kp_handles, frame, 0, bool(g_kp.value))
            md.content = vp._status(frame["stats"], 0)
    except KeyboardInterrupt:
        pass
    finally:
        if sess is not None:
            sess.close()


if __name__ == "__main__":
    main()
