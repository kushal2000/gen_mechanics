"""Kit-side worker for the interactive players. Stdlib-only at module level.

Boots Isaac Sim, builds the env, restores a policy, steps it, and streams joint
positions and object poses back to the viser parent over a localhost
Connection. It renders NOTHING: the parent draws the hand itself from these
numbers with ViserUrdf, which needs no Kit.

WHY A SEPARATE PROCESS. viser and Kit cannot share one interpreter. Kit
prepends a ``pip_prebundle`` carrying an old ``websockets`` and installs import
machinery that wins over sys.path order, the package's own ``__path__`` and an
explicit spec pin -- all three were tried, and ``websockets.http11`` still
resolved inside the Isaac Sim tree, so ``import viser`` died every time. Keep
this module's top-level imports stdlib-only so nothing heavy is loaded before
Kit boots.

Not launched directly; ``viser_play.py`` spawns it.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from multiprocessing.connection import Client

FRAME_PERIOD_S = 1.0 / 30.0      # viser cannot use more than this


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--run-dir", default="")
    p.add_argument("--population", required=True, help="already narrowed by the parent")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--expl-coef", type=float, default=0.0)
    p.add_argument("--tolerance", type=float, default=0.0)
    p.add_argument("--success-steps", type=int, default=0)
    p.add_argument("--trace", default="", help="per-step CSV-ish trace for env 0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cross-hand", action="store_true",
                   help="zero-shot: run the checkpoint on --population even though it was trained on "
                        "another hand. The network is rebuilt for the target hand's layout (every "
                        "weight is shared across joint tokens) and sigma, the one per-joint tensor, "
                        "is resized to the target's joint count.")
    p.add_argument("--hand-velocity-limit", type=float, default=-1.0,
                   help="physics.hand_velocity_limit for the TARGET hand; <0 keeps the run's value")
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--authkey", required=True)
    return p.parse_known_args()


def main() -> None:
    args, hydra_args = _args()
    # The parent passes the key as hex; it creates its Listener with the RAW
    # bytes, so decode rather than encode or the digests never match.
    conn = Client((args.host, args.port), authkey=bytes.fromhex(args.authkey))

    def send(**kw):
        try:
            conn.send(kw)
        except Exception:
            pass

    try:
        from isaaclab.app import AppLauncher

        ap = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(ap)
        app_args, _ = ap.parse_known_args([])
        app_args.headless = True
        app_args.device = args.device
        app = AppLauncher(app_args).app
        send(kind="status", text="Kit booted; building the scene")

        import gymnasium as gym
        import torch
        from omegaconf import OmegaConf

        import isaacsimenvs  # noqa: F401
        from coevolution.eval.rl_player import RlPlayer
        from coevolution.utils.hydra_utils import hydra_task_config_with_yaml

        run_dir = args.run_dir
        sys.argv = [sys.argv[0]] + list(hydra_args)

        @hydra_task_config_with_yaml(args.task, "rl_games_joint_transformer_cfg_entry_point")
        def _run(env_cfg, agent_cfg):
            saved = None
            if run_dir:
                saved = OmegaConf.load(f"{run_dir}/rank_0/.hydra/config.yaml")
                env_d = OmegaConf.to_container(saved.env, resolve=True)
                # Derived at scene build, and stored as STRINGS in the saved config.
                for k in ("observation_space", "state_space", "action_space"):
                    env_d.pop(k, None)
                env_cfg.from_dict(env_d)
            else:
                # No run to replay: hydra has already applied the task YAML, so
                # env_cfg IS the task as configured. Only the symmetric obs list
                # has to be matched by hand -- run_rank.sh sets it per run, not
                # in the YAML, and the policy is built from this width.
                env_cfg.obs.obs_list = tuple(env_cfg.obs.state_list)
            env_cfg.scene.num_envs = args.num_envs
            env_cfg.assets.robot_spec = args.population
            if args.hand_velocity_limit >= 0:
                env_cfg.physics.hand_velocity_limit = float(args.hand_velocity_limit)
            if args.tolerance > 0:
                env_cfg.termination.eval_success_tolerance = args.tolerance
            if args.success_steps > 0:
                # 1 scores a goal the instant the residual crosses the
                # threshold, instead of requiring it held for 10 steps.
                env_cfg.termination.success_steps = args.success_steps
                print(f"[worker] success_steps -> {args.success_steps}", flush=True)

            env = gym.make(args.task, cfg=env_cfg)
            inner = env.unwrapped
            obs, _ = env.reset()

            import tempfile

            import yaml

            if saved is not None:
                if args.cross_hand:
                    # The network reads its spec through ${env.assets.robot_spec}; point that
                    # at the target hand so the layout is built for ITS joints.
                    saved.env.assets.robot_spec = args.population
                acfg = OmegaConf.to_container(saved, resolve=True)["agent"]
            else:
                acfg = OmegaConf.to_container(OmegaConf.create(agent_cfg), resolve=True)
                # The train YAML interpolates the network's spec and field list
                # from env.assets.robot_spec / env.obs.obs_list, and hydra
                # resolved those WHEN IT COMPOSED -- before the env config was
                # pointed at this population. Left alone the network builds for
                # sharpa_iiwa14 against the 17-field actor list and dies with
                # "layout mismatch ... 789-d, but rl_games says 1032-d".
                # env_cfg.py:49 warns about exactly this: one knob, because the
                # agent YAML reads the network's spec from robot_spec.
                net = acfg["params"]["network"]
                net["robot_spec"] = args.population
                net["obs_list"] = list(env_cfg.obs.obs_list)
                # Training runs with no separate critic; match it.
                acfg["params"]["config"]["central_value_config"] = None
            acfg["params"]["config"]["multi_gpu"] = False
            f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
            yaml.safe_dump({"train": acfg}, f)
            f.close()
            # checkpoint None -> rl_games builds the network and never restores,
            # i.e. a randomly initialised policy.
            ckpt = args.checkpoint or None
            if ckpt and args.cross_hand:
                ckpt = _resize_sigma(ckpt, int(inner.action_space.shape[-1]))
            player = RlPlayer(obs["policy"].shape[-1], inner.action_space.shape[-1], f.name,
                              ckpt, device=args.device,
                              sapg_expl_coef=args.expl_coef, num_envs=inner.num_envs)

            spec = inner.scene_record.robot_spec
            # The real object per env, not a stand-in cube: each env draws one
            # of the curated pool, and which one is per-env bookkeeping only the
            # scene_record knows.
            from coevolution.pose_viewer import object_urdf_for_env

            obj_urdfs = []
            for e in range(int(inner.num_envs)):
                try:
                    obj_urdfs.append(str(object_urdf_for_env(inner, e)[1]))
                except Exception:
                    obj_urdfs.append(None)
            _lim_canon = inner.robot.data.joint_pos_limits[0][inner._perm_lab_to_canon]
            # The parent cannot resolve a spec itself: importing the registry pulls in
            # scene_utils/__init__ -> assembly -> pxr, which only exists inside Kit.
            # So the side that already has the spec reports the file to draw.
            import hand_sampler as _hs

            send(kind="ready",
                 urdf_path=str(_hs.resolve(spec.urdf_path)),
                 # CANONICAL order, matching joint_names: the parent labels its
                 # sliders from these, and Lab order would put each joint's range
                 # on another joint's slider.
                 joint_lower=[float(v) for v in _lim_canon[:, 0]],
                 joint_upper=[float(v) for v in _lim_canon[:, 1]],
                 joint_names=list(spec.joint_names_canonical),
                 num_envs=int(inner.num_envs),
                 num_arm_joints=int(spec.num_arm_joints),
                 object_urdfs=obj_urdfs,
                 base_pos=[float(v) for v in spec.base_pos],
                 base_rot=[float(v) for v in spec.base_rot])

            # "Hold every joint at 0" for inspecting geometry without the
            # policy. Not simply a zero ACTION: the hand action is an ABSOLUTE map
            # of [-1, 1] onto [lower, upper] (obs_utils/actions.py:87), so action 0
            # commands the MIDPOINT of each joint's range. For SHARPA those limits
            # are symmetric and the midpoint IS 0, so the two coincide -- but that
            # is a property of one hand, not of the mapping, and a generated design
            # with asymmetric limits would sit somewhere else entirely.
            #
            # Drive _cur_targets directly through the replay hook instead
            # (actions.py:38), which bypasses the delay queue, the [-1,1]
            # mapping and the moving average, so the pose is exactly 0 rather
            # than easing toward it. Clamped to the authored limits: a joint
            # whose range excludes 0 goes to its nearest limit, and how many do
            # is reported once so a strange-looking pose is explained.
            lim = inner.robot.data.joint_pos_limits           # (N, J, 2), Lab order
            zero_lab = torch.zeros_like(inner.robot.data.joint_pos)
            zero_lab = zero_lab.clamp(lim[..., 0], lim[..., 1])
            n_clamped = int((zero_lab[0].abs() > 1e-9).sum())
            if n_clamped:
                print(f"[worker] hold-zero: {n_clamped} of {zero_lab.shape[1]} joints "
                      f"cannot reach 0 and will sit at their nearest limit", flush=True)

            running, pending, det, hold_zero = True, 0, False, False
            # GOAL PROBE. Place the object at a chosen angle about a chosen axis
            # AWAY FROM THE GOAL, with physics frozen, so the success condition can
            # be walked up to by hand. The angle is what the metric should be a
            # function of; the axis is there because it is not -- the four reward
            # keypoints are two antipodal pairs and coplanar, so the residual for a
            # given angle varies by sqrt(3) with the axis, and the threshold runs
            # 5.7 deg on the most sensitive axis to 9.9 deg on the least.
            manual = None            # None = the policy drives
            from isaacsimenvs.pose_reaching_6d.obs_utils.observations import (
                compute_intermediate_values,
            )
            from isaaclab.utils.math import (
                quat_apply, quat_from_angle_axis, quat_mul,
            )

            # The 24 rotations of a cube, as quaternions: every signed permutation
            # matrix with det +1. Used only to report how far the object is from the
            # NEAREST orientation that looks identical, which for a plain box is the
            # distance a human would judge by. The reward does not use them.
            def _cube_symmetry_quats():
                import itertools

                import numpy as np
                from scipy.spatial.transform import Rotation as _R

                mats = []
                for perm in itertools.permutations(range(3)):
                    for sx in (1, -1):
                        for sy in (1, -1):
                            for sz in (1, -1):
                                M = np.zeros((3, 3))
                                for row, col in enumerate(perm):
                                    M[row, col] = (sx, sy, sz)[row]
                                if abs(np.linalg.det(M) - 1.0) < 1e-9:
                                    mats.append(M)
                q = _R.from_matrix(np.stack(mats)).as_quat()      # xyzw
                return torch.tensor(
                    np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]]),
                    device=inner.device, dtype=torch.float32)     # wxyz, (24, 4)

            SYM = _cube_symmetry_quats()
            print(f"[worker] cube symmetry group: {SYM.shape[0]} rotations", flush=True)

            def _axis_world(name):
                """The probe axis, in world, for a named direction.

                Named in the CUBE's frame and rotated out to world, because that is
                the frame the residual's axis-sensitivity lives in -- a world-frame
                polar/azimuth pair interacts with whatever random orientation the
                goal happens to have, which is why it was impossible to steer by
                hand. "most"/"least" are computed from the goal-oriented keypoints
                themselves, so they hit the extremes exactly instead of by search.
                """
                gq = inner.goal_viz.data.root_quat_w
                nE = gq.shape[0]
                kp = (inner._keypoint_offsets_fixed
                      if inner.cfg.reward.fixed_size_keypoint_reward
                      else inner._keypoint_offsets)
                o = kp.reshape(-1, kp.shape[-2], 3)[:, :2, :].expand(nE, 2, 3)
                v0 = quat_apply(gq, o[:, 0, :])
                v1 = quat_apply(gq, o[:, 1, :])
                if name == "most sensitive":
                    # Perpendicular to both distinct keypoints: each one's full
                    # radius counts, so the residual for a given angle is maximal.
                    a = torch.cross(v0, v1, dim=-1)
                elif name == "least sensitive":
                    # Halfway between them: both keypoints keep only sin(35.26 deg)
                    # of their radius perpendicular to the axis, which minimises the
                    # MAX over keypoints.
                    a = v0 + v1
                else:
                    local = {"cube face": (1.0, 0.0, 0.0),
                             "cube edge": (1.0, 1.0, 0.0),
                             "cube corner": (1.0, 1.0, 1.0),
                             "world X": None, "world Y": None, "world Z": None}
                    if name.startswith("world"):
                        w = {"world X": (1.0, 0.0, 0.0), "world Y": (0.0, 1.0, 0.0),
                             "world Z": (0.0, 0.0, 1.0)}[name]
                        a = torch.tensor(w, device=inner.device).expand(nE, 3)
                    else:
                        lv = torch.tensor(local.get(name) or (1.0, 0.0, 0.0),
                                          device=inner.device).expand(nE, 3)
                        a = quat_apply(gq, lv)
                return a / a.norm(dim=-1, keepdim=True).clamp(min=1e-9)

            def apply_manual(spec):
                import math as _m

                gq = inner.goal_viz.data.root_quat_w
                nE = gq.shape[0]
                if spec.get("mode") == "free":
                    # Intrinsic roll-pitch-yaw about the CUBE's own axes, starting
                    # from the goal, so all three at 0 means "sitting on the goal".
                    d = torch.zeros((nE, 4), device=inner.device)
                    d[:, 0] = 1.0
                    for ang_deg, lv in ((spec.get("roll", 0.0), (1.0, 0.0, 0.0)),
                                        (spec.get("pitch", 0.0), (0.0, 1.0, 0.0)),
                                        (spec.get("yaw", 0.0), (0.0, 0.0, 1.0))):
                        ax = torch.tensor(lv, device=inner.device).expand(nE, 3).contiguous()
                        a = torch.full((nE,), _m.radians(float(ang_deg)), device=inner.device)
                        d = quat_mul(d, quat_from_angle_axis(a, ax))
                    q = quat_mul(gq, d)          # post-multiply: cube-local axes
                else:
                    ax = _axis_world(spec.get("axis", "most sensitive"))
                    a = torch.full((nE,), _m.radians(float(spec.get("deg", 0.0))),
                                   device=inner.device)
                    q = quat_mul(quat_from_angle_axis(a, ax.contiguous()), gq)
                inner.object.write_root_pose_to_sim(
                    torch.cat([inner.object.data.root_pos_w, q], dim=-1))
                inner.object.write_root_velocity_to_sim(
                    torch.zeros_like(inner.object.data.root_vel_w))
                # Refresh the residual, _near_goal and _is_success against the pose
                # just written, without stepping physics.
                compute_intermediate_values(inner)

            def probe_angles():
                """(angle from the goal, angle to the nearest identical-looking pose)."""
                gq = inner.goal_viz.data.root_quat_w
                oq = inner.object.data.root_quat_w
                gi = gq * torch.tensor([1.0, -1.0, -1.0, -1.0], device=inner.device)
                rel = quat_mul(gi, oq)                                  # (n, 4)
                nominal = 2.0 * torch.acos(rel[:, 0].abs().clamp(max=1.0))
                # |dot| against each cube symmetry: the largest dot is the closest
                # symmetric image, so 2*acos of it is the smallest visual error.
                dots = (rel.unsqueeze(1) * SYM.unsqueeze(0)).sum(-1).abs()
                nearest = 2.0 * torch.acos(dots.max(dim=1).values.clamp(max=1.0))
                return nominal * 180.0 / 3.141592653589793, nearest * 180.0 / 3.141592653589793
            last_frame = [0.0]
            trace = None
            if args.trace:
                trace = open(args.trace, "w", buffering=1)
                trace.write("# step  keypoint_dist  tol  near is  succ   closest   goal_quat(wxyz)\n")
                print(f"[worker] per-step trace -> {args.trace}", flush=True)
            while app.is_running():
                while conn.poll():
                    m = conn.recv()
                    c = m.get("cmd")
                    if c == "quit":
                        env.close()
                        return
                    if c == "toggle":
                        running = not running
                        send(kind="running", value=bool(running))
                        print(f"[worker] running -> {running}", flush=True)
                    elif c == "pause":
                        running = False
                        send(kind="running", value=False)
                    elif c == "resume":
                        running = True
                        send(kind="running", value=True)
                    elif c == "step":
                        running, pending = False, pending + 1
                    elif c == "deterministic":
                        det = bool(m.get("value"))
                    elif c == "manual":
                        if m.get("value"):
                            manual = dict(m)
                            running = False
                            send(kind="running", value=False)
                            apply_manual(manual)
                        else:
                            manual = None
                            print("[worker] goal probe off; policy back in control",
                                  flush=True)
                    elif c == "joints":
                        vals = m.get("values")
                        if not vals:
                            hold_zero = False
                            inner._replay_target_lab_order = None
                            print("[worker] joint override off; policy back in control",
                                  flush=True)
                        else:
                            # The sliders are in CANONICAL order and _cur_targets is
                            # in Isaac Lab parser order, which interleaves the hand
                            # joints -- the same permutation the action pipeline
                            # applies (obs_utils/actions.py:51). Skipping it sends
                            # every joint another joint's angle, which is exactly the
                            # bug that once made the pinky look frozen.
                            t = torch.tensor(vals, device=inner.device,
                                             dtype=torch.float32).unsqueeze(0).expand(
                                                 inner.num_envs, -1)
                            tgt = t[:, inner._perm_canon_to_lab].contiguous()
                            inner._replay_target_lab_order = tgt.clamp(
                                lim[..., 0], lim[..., 1])
                            # PHYSICS KEEPS RUNNING, deliberately. These are position
                            # TARGETS: the drives have to be stepped for the joints to
                            # reach them, and the object has to be stepped to react.
                            # Pausing here stored the target and moved nothing, which
                            # is what "the sliders do nothing" was.
                            if not running:
                                running = True
                                send(kind="running", value=True)
                            # The goal probe freezes the loop to hold the object
                            # still, which would also stop the drives -- the two
                            # overrides cannot both be live.
                            manual = None
                            print(f"[worker] joint override: {len(vals)} targets, "
                                  f"max |angle| {max(abs(v) for v in vals):.3f} rad",
                                  flush=True)
                    elif c == "hold_zero":
                        hold_zero = bool(m.get("value"))
                        # Clearing the hook hands control back to the policy;
                        # leaving it set would pin the hand forever.
                        inner._replay_target_lab_order = zero_lab if hold_zero else None
                        print(f"[worker] hold_zero -> {hold_zero}", flush=True)
                    elif c == "reset":
                        obs, _ = env.reset()
                    elif c == "resample":
                        from isaacsimenvs.pose_reaching_6d.reset_utils.reset import (
                            reset_goal_trackers,
                        )

                        reset_goal_trackers(
                            inner, torch.arange(inner.num_envs, device=inner.device))

                if manual is not None:
                    # Physics frozen: re-assert the pose every iteration so nothing
                    # (gravity, a contact, a residual velocity) drifts it while the
                    # sliders are being read.
                    apply_manual(manual)
                elif running or pending:
                    with torch.no_grad():
                        act = player.get_normalized_action(obs["policy"], deterministic_actions=det)
                    obs, _, _, _, _ = env.step(act)
                    pending = max(0, pending - 1)

                # Per-STEP trace for env 0, unthrottled, to a file. The viser
                # print samples twice a second, which cannot distinguish "the
                # goal resampled without a success" from "a success happened
                # between two samples and zeroed episode_length_buf". This logs
                # every step, so a resample either lines up with is_success or
                # it does not.
                if trace is not None:
                    gq = inner.goal_viz.data.root_quat_w[0]
                    row = (f"{int(inner.episode_length_buf[0]):6d} "
                           f"{float(inner._keypoints_max_dist[0]):9.5f} "
                           f"{float(inner._current_success_tolerance):8.5f} "
                           f"{int(bool(inner._near_goal[0])):3d} "
                           f"{int(bool(inner._is_success[0])):3d} "
                           f"{int(inner._successes[0]):5d} "
                           f"{float(inner._closest_keypoint_max_dist[0]):9.5f} "
                           f"{' '.join(f'{float(v):+.4f}' for v in gq)}\n")
                    trace.write(row)
                    if int(inner.episode_length_buf[0]) % 200 == 0:
                        trace.flush()

                # Cap the frame rate. Uncapped, the child outruns viser by
                # orders of magnitude and the connection becomes a growing
                # backlog -- commands still arrive on time but the VIEW lags
                # minutes behind, so pause looks broken.
                now = time.time()
                if now - last_frame[0] < FRAME_PERIOD_S:
                    if not running and not pending:
                        time.sleep(0.005)      # paused: yield instead of spinning
                    continue
                last_frame[0] = now

                origins = inner.scene.env_origins
                # THE TWO KEYPOINT SETS THE METRIC COMPARES, so the residual can be
                # seen rather than read off a number. Under orientation_only_goal the
                # comparison is between sets CENTRED on their own origins
                # (observations.py:250), so the goal set is sent translated onto the
                # object's centre: the distance between paired points is then exactly
                # what _keypoints_max_dist maxes over. Drawing the goal keypoints at
                # the ghost instead would show a position error the reward ignores.
                kp_off = (inner._keypoint_offsets_fixed
                          if inner.cfg.reward.fixed_size_keypoint_reward
                          else inner._keypoint_offsets)
                _op, _oq = inner.object.data.root_pos_w, inner.object.data.root_quat_w
                _gq = inner.goal_viz.data.root_quat_w
                _n, _k = _op.shape[0], kp_off.shape[-2]
                _off = kp_off.expand(_n, _k, 3).reshape(-1, 3)
                _obj_kp = _op.unsqueeze(1) + quat_apply(
                    _oq.unsqueeze(1).expand(_n, _k, 4).reshape(-1, 4), _off).reshape(_n, _k, 3)
                _goal_kp = _op.unsqueeze(1) + quat_apply(
                    _gq.unsqueeze(1).expand(_n, _k, 4).reshape(-1, 4), _off).reshape(_n, _k, 3)
                send(kind="frame",
                     obj_kp=(_obj_kp - origins.unsqueeze(1)).cpu().numpy().tolist(),
                     goal_kp=(_goal_kp - origins.unsqueeze(1)).cpu().numpy().tolist(),
                     # CANONICAL order. robot.data.joint_pos is in Isaac Lab's
                     # parser order, which INTERLEAVES the hand joints relative
                     # to canonical (SHARPA lands as 7, 12, 17, 22, 27, 8, ...).
                     # The parent zips these against joint_names_canonical, so
                     # sending Lab order gives every joint another joint's angle
                     # -- the whole hand animates wrongly, and a finger whose
                     # stand-in value barely moves looks frozen.
                     joint_pos=inner.robot.data.joint_pos[:, inner._perm_lab_to_canon]
                                    .detach().cpu().numpy().tolist(),
                     object_pos=(inner.object.data.root_pos_w - origins).cpu().numpy().tolist(),
                     object_quat=inner.object.data.root_quat_w.cpu().numpy().tolist(),
                     goal_pos=(inner.goal_viz.data.root_pos_w - origins).cpu().numpy().tolist(),
                     goal_quat=inner.goal_viz.data.root_quat_w.cpu().numpy().tolist(),
                     # The in-hand task parks the table 1.35 m below the palm --
                     # the ground plane is the real catch floor -- so drawing it
                     # is pure clutter there.
                     table_pos=((inner.table.data.root_pos_w - origins).cpu().numpy().tolist()
                                if _show_table(inner) else None),
                     table_quat=(inner.table.data.root_quat_w.cpu().numpy().tolist()
                                 if _show_table(inner) else None),
                     stats=_stats(inner, probe_angles()))
            env.close()

        _run()
    except Exception:
        send(kind="error", text=traceback.format_exc())
    finally:
        try:
            conn.close()
        except Exception:
            pass
        import os

        os._exit(0)          # Kit does not tear down cleanly


def _resize_sigma(path: str, n_actions: int) -> str:
    """A copy of the checkpoint whose sigma fits a hand with n_actions joints.

    sigma is (exploration blocks, joints) -- the only tensor in the joint transformer
    whose shape depends on the hand. Each block's joints get that block's mean log-std,
    so a stochastic action explores as widely as it did in training; the greedy action
    (deterministic, or expl-coef 0) never reads sigma at all.
    """
    import tempfile

    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    inner = ck[0] if 0 in ck else ck
    sig = inner["model"].get("a2c_network.sigma")
    if sig is None or sig.shape[-1] == n_actions:
        return path
    new = sig.mean(dim=-1, keepdim=True).expand(*sig.shape[:-1], n_actions).clone()
    inner["model"]["a2c_network.sigma"] = new
    out = tempfile.NamedTemporaryFile(suffix=".pth", delete=False).name
    torch.save(ck, out)
    print(f"[worker] cross-hand: sigma {tuple(sig.shape)} -> {tuple(new.shape)}", flush=True)
    return out


def _show_table(inner) -> bool:
    """Only when the task actually uses it."""
    return (getattr(inner, "table", None) is not None
            and not getattr(inner.cfg.reset, "object_in_hand", False))


def _stats(inner, angles=None) -> dict:
    def g(name):
        v = getattr(inner, name, None)
        if v is None:
            return None
        try:
            return [float(x) for x in (v if hasattr(v, "__len__") else [v])]
        except Exception:
            return None
    out = {k: g(k) for k in ("_current_success_tolerance", "_keypoints_max_dist",
                             "_closest_keypoint_max_dist", "_near_goal", "_successes",
                             "_lifted_object", "episode_length_buf", "_is_success")}
    # The tolerance the comparison ACTUALLY uses. _current_success_tolerance is
    # not it: observations.py:269 multiplies by keypoint_scale before comparing,
    # so the raw number in the config is 1.5x smaller than the threshold in the
    # same units as _keypoints_max_dist. Showing both stops that being re-derived
    # by hand every time someone reads the panel.
    try:
        out["_tol_effective"] = [float(inner._current_success_tolerance
                                       * inner.cfg.reward.keypoint_scale)]
    except Exception:
        pass
    if angles is not None:
        out["_angle_deg"] = [float(x) for x in angles[0]]
        out["_angle_sym_deg"] = [float(x) for x in angles[1]]
    return out


if __name__ == "__main__":
    main()
