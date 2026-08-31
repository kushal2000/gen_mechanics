"""Interactive embodiment viewer: pick a design, watch it work, browse its objects.

    .venv_isaacsim/bin/python -m coevolution.population.eval_interactive \\
        --checkpoint train_dir/.../nn/0_pose_reach_sapg.pth \\
        --run_dir train_dir/.../mec_population24k_seed0_2026-08-17_15-13-28 \\
        --port 8085

Two controls, with very different costs:

* **Embodiment** -- a dropdown over the sampled population plus the fixed hands.
  Changing it restarts the simulator, because Kit cannot be torn down and
  re-created in-process. So this process owns viser and NOTHING else; Isaac Sim
  lives in a subprocess (``_worker.py``) that is killed and respawned per design.
  Expect roughly a minute.
* **Object** -- 64 environments hold the same design and 64 different objects.
  The worker streams every environment each step, so switching object is a local
  redraw with no round trip and no reload.

The object shown is a member of the training pool, but it is generally NOT the
one this design trained against: env i holds pool entry ``i % pool_size``, and a
design's own object is ``design_index % pool_size``, which for most designs is
outside the 64 loaded here. The status panel names the design's training object
so the distinction stays visible.

**Geometry.** ``visual`` is the URDF's render geometry; ``collision`` is what
PhysX resolves contacts against. They differ -- a generated hand's capsule is one
collision cylinder but a cylinder plus two spheres in visual. Judge contact on
``collision``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = str(REPO_ROOT / ".venv_isaacsim" / "bin" / "python")

TABLE_DIMS = (0.475, 0.4, 0.3)
OBJECT_COLOR = (215, 140, 50)
GOAL_COLOR = (90, 190, 120)


def discover_designs(population_seed: int, limit: int, extra: list[str]) -> list[str]:
    """Dropdown contents: named specs first, then population members.

    A manifest holds 24,576 designs and a dropdown cannot, so this takes a
    prefix. ``--designs`` names specific ones to append, which is how you reach
    an interesting design found by an offline sweep.
    """
    from isaacsimenvs.pose_reaching_6d.scene_utils.robots import REGISTRY

    names = sorted(REGISTRY)
    names.append("gen_sharpa_like")

    try:
        from hand_sampler.population import load_population
        hands = load_population(population_seed)
        names += [h.name for h in hands[:limit]]
    except Exception as exc:  # noqa: BLE001 - a missing manifest is not fatal here
        print(f"[viz] no population for seed {population_seed}: {exc}")

    for name in extra:
        if name and name not in names:
            names.append(name)
    return names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", default=None,
                   help="YAML with a top-level 'train:' key. Default: synthesised "
                        "from --run_dir")
    p.add_argument("--run_dir", default=None,
                   help="Training run to take the policy config and population "
                        "seed from")
    p.add_argument("--population_seed", type=int, default=3)
    p.add_argument("--initial_design", default=None)
    p.add_argument("--designs", default="",
                   help="Comma-separated extra design names to list")
    p.add_argument("--design_limit", type=int, default=64,
                   help="How many population members to list (default 64)")
    p.add_argument("--num_envs", type=int, default=64,
                   help="Envs = distinct objects shown (default 64)")
    p.add_argument("--author_object_usds", type=int, default=1,
                   help="1 authors object USDs (seconds); 0 converts (minutes)")
    p.add_argument("--num_assets_per_type", type=int, default=100,
                   help="100 reproduces the training pool's object identities")
    p.add_argument("--object_seed", type=int, default=42)
    p.add_argument("--goals_per_episode", type=int, default=10)
    p.add_argument("--success_tolerance", type=float, default=None,
                   help="Pins termination.eval_success_tolerance in metres "
                        "(default: the suite's 0.01, the curriculum floor)")
    p.add_argument("--dr", default="off", choices=("off", "train", "hard"))
    p.add_argument("--sapg_expl_coef", type=float, default=50.0)
    p.add_argument("--rl_device", default="cuda:0")
    p.add_argument("--port", type=int, default=8085)
    p.add_argument("--python", default=DEFAULT_PYTHON)
    p.add_argument("--hz", type=float, default=60.0,
                   help="Parent redraw rate; the worker paces the sim")
    p.add_argument("--no_autoload", action="store_true",
                   help="Start with no embodiment loaded")
    p.add_argument("--selftest", default=None,
                   help="Headless check: load the initial design, then SWITCH to "
                        "this one, then exit 0. Switching is the path that is easy "
                        "to get wrong and impossible to check by starting the "
                        "server, so it gets its own test")
    return p


class EmbodimentViewer:
    def __init__(self, args, designs: list[str], policy_config: str):
        import viser

        self.args = args
        self.designs = designs
        self.policy_config = policy_config
        self._proc = None
        self._conn = None
        self._pending_load = False
        self._ready = None
        self._snapshot = None
        self._urdf_handle = None
        self._object_handles = {}
        self._goal_handles = {}
        self._joint_order = None
        self._episodes = []

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)

        @self.server.on_client_connect
        def _(client) -> None:
            # viser's default view of a scene this size is an empty grey volume.
            client.camera.position = (0.9, -0.9, 1.15)
            client.camera.look_at = (0.0, 0.15, 0.62)

        self._build_static_scene()
        self._build_gui()

    # --- scene ------------------------------------------------------------
    def _build_static_scene(self) -> None:
        self.server.scene.add_grid("/ground", width=2.4, height=2.4, cell_size=0.1)
        # The table box is centred on its rigid-body root and the root is
        # randomized per episode, so its pose is read from the sim each frame
        # rather than fixed here.
        self.table = self.server.scene.add_box(
            "/table", dimensions=TABLE_DIMS, position=(0.0, 0.0, 0.38),
            color=(180, 130, 70))
        self.robot_frame = self.server.scene.add_frame("/robot", show_axes=False)
        self.object_frame = self.server.scene.add_frame("/object", show_axes=False)
        self.goal_frame = self.server.scene.add_frame("/goal", show_axes=False)

    def _build_gui(self) -> None:
        g = self.server.gui
        with g.add_folder("Embodiment", expand_by_default=True):
            self.dd_design = g.add_dropdown(
                "design", tuple(self.designs),
                initial_value=self.args.initial_design or self.designs[0])
            self.btn_load = g.add_button("Load embodiment  (restarts sim, ~1 min)")
            # Flag only. viser runs GUI callbacks on its OWN thread, and _load()
            # tears down the scene graph the main thread is drawing from -- it
            # nulls _urdf_handle mid-frame, and the main loop dies inside
            # ViserUrdf.update_cfg on nodes that were just removed. It also
            # blocks on listener.accept() for the whole Kit boot, freezing every
            # other control. The main loop owns all of it.
            self.btn_load.on_click(lambda _: setattr(self, "_pending_load", True))
            self.md_design = g.add_markdown("_no embodiment loaded_")

        with g.add_folder("Object", expand_by_default=True):
            self.sl_env = g.add_slider("env / object", min=0,
                                       max=max(0, self.args.num_envs - 1),
                                       step=1, initial_value=0)
            self.md_object = g.add_markdown("_--_")

        with g.add_folder("Episode", expand_by_default=True):
            self.btn_run = g.add_button("Run")
            self.btn_pause = g.add_button("Pause")
            self.btn_step = g.add_button("Step once")
            self.btn_reset = g.add_button("Reset")
            self.btn_run.on_click(lambda _: self._send("run"))
            self.btn_pause.on_click(lambda _: self._send("pause"))
            self.btn_step.on_click(lambda _: self._send("step"))
            self.btn_reset.on_click(lambda _: self._send("reset"))

        with g.add_folder("View", expand_by_default=True):
            self.dd_geom = g.add_dropdown("geometry", ("visual", "collision"),
                                          initial_value="visual")
            self.dd_geom.on_update(lambda _: self._apply_geometry())

        self.md_status = g.add_markdown("**Status:** pick a design, press Load")
        self.md_stats = g.add_markdown("**Episodes:** none yet")

    def _apply_geometry(self) -> None:
        if self._urdf_handle is None:
            return
        collision = self.dd_geom.value == "collision"
        self._urdf_handle.show_collision = collision
        self._urdf_handle.show_visual = not collision

    # --- worker lifecycle --------------------------------------------------
    def _kill_worker(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send("quit")
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.kill()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                pass
            self._proc = None

    def _load(self) -> None:
        design = self.dd_design.value
        self._kill_worker()
        self._clear_design_geometry()
        self._ready = None
        self._snapshot = None
        self._episodes = []
        self.md_status.content = (
            f"**Status:** booting Isaac Sim for `{design}` — about a minute…")

        authkey = os.urandom(16)
        listener = Listener(("127.0.0.1", 0), authkey=authkey)
        host, port = listener.address
        # The worker cannot connect until Kit has loaded; without this the
        # accept() below times out on the default socket timeout instead.
        listener._listener._socket.settimeout(600.0)

        cmd = [
            self.args.python, "-u", "-m", "coevolution.population._worker",
            "--worker-host", str(host), "--worker-port", str(port),
            "--worker-authkey", authkey.hex(),
            "--design", design,
            "--population_seed", str(self.args.population_seed),
            "--checkpoint", str(self.args.checkpoint),
            "--policy_config", str(self.policy_config),
            "--num_envs", str(self.args.num_envs),
            "--num_assets_per_type", str(self.args.num_assets_per_type),
            "--object_seed", str(self.args.object_seed),
            "--author_object_usds", str(self.args.author_object_usds),
            "--sapg_expl_coef", str(self.args.sapg_expl_coef),
            "--rl_device", self.args.rl_device,
            "--dr", self.args.dr,
            "--goals_per_episode", str(self.args.goals_per_episode),
        ]
        if self.args.run_dir:
            cmd += ["--run_dir", str(self.args.run_dir)]
        if self.args.success_tolerance is not None:
            cmd += ["--success_tolerance", str(self.args.success_tolerance)]
        env = os.environ.copy()
        env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        try:
            self._proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
            self._conn = listener.accept()
            print(f"[viz] worker pid={self._proc.pid} design={design}", flush=True)
        except Exception as exc:  # noqa: BLE001
            self.md_status.content = f"**Status:** worker launch failed — {exc}"
            print(traceback.format_exc(), flush=True)
            self._kill_worker()
        finally:
            try:
                listener.close()
            except Exception:
                pass

    def _send(self, msg: str) -> None:
        if self._conn is None or self._ready is None:
            self.md_status.content = "**Status:** load an embodiment first"
            return
        try:
            self._conn.send(msg)
        except Exception as exc:  # noqa: BLE001
            self.md_status.content = f"**Status:** worker gone — {exc}"
            self._kill_worker()

    # --- per-design geometry ----------------------------------------------
    def _clear_design_geometry(self) -> None:
        if self._urdf_handle is not None:
            try:
                self._urdf_handle.remove()
            except Exception:
                pass
            self._urdf_handle = None
        for handles in (self._object_handles, self._goal_handles):
            for h in handles.values():
                try:
                    h.remove()
                except Exception:
                    pass
            handles.clear()

    def _on_ready(self, ready: dict, snapshot) -> None:
        import numpy as np
        import trimesh
        import yourdfpy
        from viser.extras import ViserUrdf

        from hand_sampler.gates.mesh import _geometry_to_mesh

        self._ready = ready
        urdf_path = Path(ready["robot_urdf"])
        robot_urdf = yourdfpy.URDF.load(
            str(urdf_path), load_meshes=True, load_collision_meshes=True,
            build_scene_graph=True, build_collision_scene_graph=True)
        self.robot_frame.position = tuple(ready["base_pos"])
        self.robot_frame.wxyz = tuple(ready["base_rot"])
        self._urdf_handle = ViserUrdf(
            self.server, robot_urdf, root_node_name="/robot",
            load_meshes=True, load_collision_meshes=True)
        self._apply_geometry()

        # ViserUrdf wants the URDF's actuated-joint order; the sim reports Isaac
        # Lab's parser order. Map by name -- the two are not the same order and
        # assuming they are silently poses the wrong joints.
        sim_index = {n: i for i, n in enumerate(ready["joint_names"])}
        viser_joints = list(self._urdf_handle.get_actuated_joint_names())
        missing = [n for n in viser_joints if n not in sim_index]
        if missing:
            print(f"[viz] WARNING: {len(missing)} URDF joints absent from the sim "
                  f"(held at 0): {missing[:4]}", flush=True)
        self._joint_order = [sim_index.get(n, -1) for n in viser_joints]

        def asset_mesh(path: str):
            u = yourdfpy.URDF.load(path, load_meshes=False,
                                   load_collision_meshes=False,
                                   build_scene_graph=True,
                                   build_collision_scene_graph=True)
            out = None
            for name, link in u.link_map.items():
                for coll in link.collisions:
                    m = _geometry_to_mesh(coll.geometry, Path(path).parent, hull=False)
                    if m is None:
                        continue
                    m = m.copy()
                    origin = coll.origin if coll.origin is not None else np.eye(4)
                    m.apply_transform(u.get_transform(name) @ origin)
                    out = m if out is None else trimesh.util.concatenate([out, m])
            return out

        # One mesh per env, since every env holds a different object. Geometry is
        # used as written: the normalized scale on the env is a reward/observation
        # quantity, not a mesh multiplier.
        for e, path in enumerate(ready["object_urdf_paths"]):
            m = asset_mesh(path)
            if m is None:
                continue
            self._object_handles[e] = self.server.scene.add_mesh_simple(
                f"/object/env{e}", vertices=m.vertices, faces=m.faces,
                color=OBJECT_COLOR)
            self._goal_handles[e] = self.server.scene.add_mesh_simple(
                f"/goal/env{e}", vertices=m.vertices, faces=m.faces,
                color=GOAL_COLOR, opacity=0.45)

        n_distinct = len(set(ready["object_pool_index"]))
        bits = [f"### `{ready['design']}`"]
        if ready.get("n_active_fingers") is not None:
            bits.append(f"{ready['n_active_fingers']} active fingers, "
                        f"{ready['n_active_joints']} active joints")
        bits.append(f"obs {ready['obs_dim']}, pool {ready['pool_size']} objects")
        if ready.get("training_object_label"):
            bits.append(f"**trained against** pool #{ready['training_pool_index']} — "
                        f"{ready['training_object_label']}")
        self.md_design.content = "\n\n".join(bits)
        self.md_status.content = (
            f"**Status:** ready — {n_distinct} distinct objects across "
            f"{ready['num_envs']} envs. Press Run.")
        self._snapshot = snapshot
        print(f"[viz] ready: {ready['design']}, {n_distinct} distinct objects",
              flush=True)

    # --- per-frame ---------------------------------------------------------
    def _draw(self) -> None:
        import numpy as np

        # Bound locally, all of them: a load can land between any two of these
        # lines and the frame must either use one consistent set or skip.
        snap, ready, urdf = self._snapshot, self._ready, self._urdf_handle
        if snap is None or ready is None or urdf is None or self._joint_order is None:
            return
        e = int(self.sl_env.value)

        t = snap["table"][e]
        self.table.position = tuple(float(v) for v in t[:3])
        self.table.wxyz = tuple(float(v) for v in t[3:])

        q = snap["joint_pos"][e]
        urdf.update_cfg(
            np.array([q[i] if i >= 0 else 0.0 for i in self._joint_order]))

        o = snap["object"][e]
        self.object_frame.position = tuple(float(v) for v in o[:3])
        self.object_frame.wxyz = tuple(float(v) for v in o[3:])
        g = snap["goal"][e]
        self.goal_frame.position = tuple(float(v) for v in g[:3])
        self.goal_frame.wxyz = tuple(float(v) for v in g[3:])

        for idx, h in self._object_handles.items():
            h.visible = (idx == e)
        for idx, h in self._goal_handles.items():
            h.visible = (idx == e)

        pool_idx = ready["object_pool_index"][e]
        is_training = (pool_idx == ready.get("training_pool_index"))
        self.md_object.content = (
            f"env **{e}** — pool #{pool_idx}\n\n"
            f"{ready['object_labels'][e]}"
            + ("\n\n**this is the design's training object**" if is_training else ""))

    def _handle(self, msg) -> None:
        tag = msg[0]
        if tag == "ready":
            self._on_ready(msg[1], msg[2])
        elif tag == "state":
            self._snapshot = msg[1]
            step, episodes = msg[2], msg[3]
            e = int(self.sl_env.value)
            self.md_status.content = (
                f"**Status:** step {step} — env {e}, "
                f"goals {int(self._snapshot['successes'][e])}"
                f"/{self._ready['goals_per_episode']}, "
                f"lifted {bool(self._snapshot['lifted'][e])}")
            total = sum(episodes)
            if total:
                self.md_stats.content = (
                    f"**Episodes:** {total} finished across "
                    f"{self._ready['num_envs']} envs "
                    f"({min(episodes)}–{max(episodes)} per env)")
        elif tag == "error":
            self.md_status.content = f"**Status:** worker error — {str(msg[1])[:200]}"
            print(f"[viz] worker error:\n{msg[1]}", flush=True)
            self._kill_worker()

    def _poll(self) -> None:
        if self._conn is None:
            return
        conn = self._conn
        try:
            # Bound to a local: handling an "error" message calls _kill_worker(),
            # which sets self._conn to None, and re-reading it in the loop
            # condition then raises AttributeError instead of reporting the
            # worker's actual failure.
            while conn.poll(0):
                self._handle(conn.recv())
                if self._conn is not conn:
                    break
        except (EOFError, OSError) as exc:
            self.md_status.content = f"**Status:** worker exited — {exc}"
            self._kill_worker()

    def selftest(self, second_design: str, timeout_s: float = 900.0) -> int:
        """Load, switch, and confirm both reach ready. Returns a process exit code.

        The first load is exercised by simply starting the viewer. The SWITCH is
        not, and it is where the thread race lived: the button callback used to
        tear down the scene graph under the drawing thread. Reproducing that by
        hand means clicking a browser button at the wrong moment; here it is one
        command.
        """
        deadline = time.time() + timeout_s
        stages = [(self.dd_design.value, "initial"), (second_design, "switch")]
        for design, label in stages:
            self.dd_design.value = design
            self._pending_load = True
            seen = None
            while time.time() < deadline:
                if self._pending_load:
                    self._pending_load = False
                    self._load()
                self._poll()
                self._draw()          # must survive teardown mid-flight
                if self._ready is not None and self._ready["design"] == design:
                    seen = self._ready
                    break
                if self._conn is None and self._ready is None and seen is None:
                    time.sleep(0.05)
                time.sleep(0.02)
            if seen is None:
                print(f"[selftest] FAILED at {label} load of {design}", flush=True)
                self._kill_worker()
                return 1
            # Draw a few frames so a stale-handle crash has a chance to fire.
            for _ in range(120):
                self._poll()
                self._draw()
                time.sleep(1.0 / 120.0)
            print(f"[selftest] {label} OK: {design}, "
                  f"{len(set(seen['object_pool_index']))} distinct objects",
                  flush=True)
        self._kill_worker()
        print("[selftest] embodiment switch test OK", flush=True)
        return 0

    def run(self, autoload: bool = True) -> None:
        print(f"\n  Embodiment viewer  http://localhost:{self.args.port}\n")
        # Boot straight into a design. Landing on an empty scene with a Load
        # button reads as a broken viewer, and the first load costs a Kit boot
        # either way.
        if autoload:
            self._pending_load = True
        period = 1.0 / max(self.args.hz, 1e-3)
        try:
            while True:
                t0 = time.time()
                if self._pending_load:
                    self._pending_load = False
                    self._load()
                try:
                    self._poll()
                    self._draw()
                except Exception as exc:  # noqa: BLE001
                    # A viewer that dies on one bad frame takes the whole server
                    # with it and loses the loaded embodiment, which costs a Kit
                    # boot to get back. Report and keep serving.
                    self.md_status.content = f"**Status:** draw error — {exc}"
                    print(f"[viz] draw error: {traceback.format_exc()}", flush=True)
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            print("\n[viz] shutting down")
            self._kill_worker()


def main() -> int:
    args = build_parser().parse_args()

    policy_config = args.policy_config
    if policy_config is None:
        if not args.run_dir:
            raise SystemExit("pass --policy_config, or --run_dir to synthesise one")
        import tempfile

        from coevolution.population.run_config import (
            load_run_config, synthesise_policy_config,
        )

        # Into a temp dir, not --run_dir: that directory may belong to a job that
        # is still running, and a viewer should leave nothing in it.
        policy_config = synthesise_policy_config(
            load_run_config(args.run_dir),
            Path(tempfile.mkdtemp(prefix="genmech_viewer_")) / "policy_config.yaml",
            args.num_envs)
        print(f"[viz] synthesised policy config -> {policy_config}")

    extra = [d.strip() for d in args.designs.split(",") if d.strip()]
    if args.initial_design:
        extra.append(args.initial_design)
    designs = discover_designs(args.population_seed, args.design_limit, extra)
    print(f"[viz] {len(designs)} designs listed "
          f"(first {args.design_limit} of the population + fixed hands)")

    viewer = EmbodimentViewer(args, designs, policy_config)
    if args.selftest:
        return viewer.selftest(args.selftest)
    viewer.run(autoload=not args.no_autoload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
