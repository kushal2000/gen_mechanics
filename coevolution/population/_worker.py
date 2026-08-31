"""Isaac Sim side of the interactive embodiment viewer.

Launched as a subprocess by ``eval_interactive.py``, one per embodiment. It owns
Kit and the policy; the parent owns viser and never imports Isaac Sim. State
flows one way over ``multiprocessing.connection``: the worker sends a snapshot of
every environment each step, the parent sends commands.

**Why a subprocess at all.** Kit cannot be torn down and re-created in-process,
so an in-process viewer is stuck with whatever embodiment it booted. Switching
designs therefore means switching processes, and the only thing that can survive
that is a parent that never loaded Kit. That is the whole architecture.

Every environment is streamed each step, not just the selected one, so the
parent's object toggle is a local redraw with no round trip.

Not meant to be run by hand; see ``eval_interactive.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from multiprocessing.connection import Client

CONTROL_HZ = 60.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--worker-host", required=True)
    p.add_argument("--worker-port", type=int, required=True)
    p.add_argument("--worker-authkey", required=True)
    p.add_argument("--design", required=True,
                   help="A population design name (gen_SSSS_NNNNN) or a registered "
                        "robot spec (sharpa_iiwa14, gen_sharpa_like, ...)")
    p.add_argument("--population_seed", type=int, default=3)
    p.add_argument("--run_dir", default=None,
                   help="Training run whose asset/actuation/observation config to "
                        "reproduce. Without it the obs layout comes from cfg "
                        "defaults and a checkpoint will refuse to load")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", required=True)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_assets_per_type", type=int, default=100)
    p.add_argument("--object_seed", type=int, default=42)
    p.add_argument("--author_object_usds", type=int, default=1,
                   help="1 authors object USDs directly (seconds); 0 converts them "
                        "from URDF, which is minutes at a 1200-entry pool")
    p.add_argument("--sapg_expl_coef", type=float, default=50.0)
    p.add_argument("--rl_device", default="cuda:0")
    p.add_argument("--dr", default="off", choices=("off", "train", "hard"))
    p.add_argument("--goals_per_episode", type=int, default=10)
    p.add_argument("--success_tolerance", type=float, default=None,
                   help="Pins termination.eval_success_tolerance in metres")
    return p


def _pose7(pos, quat):
    """(N, 7) env-local pose: xyz then wxyz, the order viser wants."""
    import numpy as np
    return np.concatenate([pos, quat], axis=1).astype(np.float32)


def run(conn, args) -> None:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = False
    app = AppLauncher(launcher_args).app

    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from coevolution.population.object_pool import (
        expected_pool_size, pool_index_for_env, reconstruct_pool,
    )
    from coevolution.population.run_config import (
        apply_run_fields, eval_protocol, load_run_config, run_env_dict, set_by_path,
    )
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.pose_reaching_6d.scene_utils.robots import REGISTRY
    from isaacsimenvs.pose_reaching_6d.env import PoseReachEnv
    from isaacsimenvs.pose_reaching_6d.env_cfg import PoseReachEnvCfg
    from isaacsimenvs.pose_reaching_6d.env_multi import PoseReachMultiEnv
    from isaacsimenvs.pose_reaching_6d.env_multi_cfg import PoseReachMultiEnvCfg
    from isaacsimenvs.pose_reaching_6d.obs_utils.morphology import population_descriptors
    from hand_sampler.paths import resolve as resolve_repo_path

    is_population_design = args.design not in REGISTRY and args.design.startswith("gen_") \
        and args.design.count("_") == 2 and args.design.split("_")[-1].isdigit() \
        and len(args.design.split("_")[-1]) == 5

    # A population member cannot be reached through robot_spec: synth_spec's name
    # regex is 3-digit and population names are 5-digit, and widening it would
    # replay the sampler rather than read the manifest -- producing a DIFFERENT
    # hand under the right name. Materialise from the manifest instead.
    if is_population_design:
        from hand_sampler.population import load_population
        from hand_sampler.synth_spec import synth_spec

        hands = load_population(args.population_seed)
        by_name = {h.name: h for h in hands}
        if args.design not in by_name:
            raise KeyError(
                f"{args.design!r} is not in seed {args.population_seed}'s manifest "
                f"({len(hands)} designs)")
        hand = by_name[args.design]
        design_spec = synth_spec(hand)

        class _SingleDesignEnv(PoseReachMultiEnv):
            """The multi-embodiment env holding ONE design in every env.

            Subclassing the multi env rather than the single one is what carries
            the morphology descriptor: a population policy conditions on it, and
            PoseReachEnv neither builds nor exposes it. The population here is a
            one-element list injected before _setup_scene reads it, which is why
            an arbitrary design index costs one design rather than a prefix of
            index+1 designs.
            """

            def _resolve_spec(self, cfg):
                self._robot_population_specs = [design_spec]
                for field in ("obs_list", "state_list"):
                    current = tuple(getattr(cfg.obs, field))
                    if "morphology" not in current:
                        setattr(cfg.obs, field, current + ("morphology",))
                return design_spec

            def _build_morphology_obs(self) -> None:
                table = torch.as_tensor(
                    population_descriptors([hand]), device=self.device,
                    dtype=torch.float32)
                self._morphology_per_env = table[self._robot_design_index_per_env]

        cfg = PoseReachMultiEnvCfg()
        cfg.assets.robot_population_seed = args.population_seed
        env_class = _SingleDesignEnv
        design_meta = {
            "n_active_fingers": hand.n_active_fingers,
            "n_active_joints": hand.n_active_joints,
        }
    else:
        cfg = PoseReachEnvCfg()
        cfg.assets.robot_spec = args.design
        env_class = PoseReachEnv
        design_meta = {}

    # The run's config first: the observation LAYOUT lives there, and cfg
    # defaults disagree with it by 22 dims -- enough that the checkpoint's
    # state_dict will not load at all.
    if args.run_dir:
        apply_run_fields(cfg, run_env_dict(load_run_config(args.run_dir)))

    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.assets.object_seed = args.object_seed
    # Default ON, unlike the env cfg: a 1200-entry pool takes minutes to convert
    # from URDF and seconds to author, and a viewer that appears to hang for
    # minutes on every embodiment switch is not usable. The population runs
    # author too, so this also matches what trained.
    cfg.assets.author_object_usds = bool(args.author_object_usds)

    if is_population_design:
        # Two things the run config must NOT decide for a single injected design.
        #
        # robot_population_count: apply_run_fields copies the run's 24576, but the
        # injected population is one spec, and _author_population_usds compares
        # the two lengths and dies.
        #
        # author_robot_usds: this is the trap. The authored path re-derives the
        # hand list from the config -- load_population(seed)[:count] -- and calls
        # author_robot_prims(hands[idx], specs[idx]). Geometry comes from `hands`,
        # metadata from `specs`. With count=1 the length check PASSES and it
        # silently authors design 0's geometry under design 49's spec. The
        # convert path builds its USD from `specs` alone, so it cannot disagree
        # with itself, and at one design it costs one conversion -- the k*n
        # blow-up that motivated authoring does not exist at k=1.
        cfg.assets.robot_population_count = 1
        cfg.assets.author_robot_usds = False

    protocol = eval_protocol(args.dr, args.goals_per_episode, args.success_tolerance)
    for key, value in protocol.items():
        set_by_path(cfg, key, value)

    # Envs take pool entry i % pool_size, so a pool smaller than the env count
    # silently repeats objects -- which looks like a working viewer showing 64
    # distinct envs, half of them duplicates.
    pool_size = expected_pool_size(cfg.assets.handle_head_types,
                                   args.num_assets_per_type)
    if pool_size < args.num_envs:
        raise ValueError(
            f"pool of {pool_size} objects < {args.num_envs} envs, so objects would "
            f"repeat. Raise --num_assets_per_type to at least "
            f"{-(-args.num_envs // (pool_size // args.num_assets_per_type))}.")

    env = env_class(cfg=cfg)
    env._replay_target_lab_order = None
    spec = env.robot_spec
    n_act = int(cfg.action_space)

    player = RlPlayer(
        num_observations=int(cfg.observation_space), num_actions=n_act,
        config_path=args.policy_config, checkpoint_path=args.checkpoint,
        device=args.rl_device,
        sapg_expl_coef=args.sapg_expl_coef if args.sapg_expl_coef >= 0 else None,
        num_envs=args.num_envs,
    )
    player.player.init_rnn()

    obs, _ = env.reset()
    # RigidObject root poses are unpopulated until the sim has run, so the first
    # snapshot would put the table 230 mm below where it is.
    obs, _, _, _, _ = env.step(torch.zeros((args.num_envs, n_act), device=env.device))

    pool = reconstruct_pool(
        handle_head_types=tuple(cfg.assets.handle_head_types),
        num_assets_per_type=int(cfg.assets.num_assets_per_type),
        object_seed=int(cfg.assets.object_seed),
        shuffle=bool(cfg.assets.shuffle_assets),
        density_scale=float(cfg.assets.object_density_scale),
    )
    live_pool_index = env._object_asset_index_per_env.cpu().numpy().tolist()
    for e, idx in enumerate(live_pool_index):
        if idx != pool_index_for_env(e, len(pool)):
            raise RuntimeError(
                f"env {e} holds pool entry {idx}, expected "
                f"{pool_index_for_env(e, len(pool))}; object labels would be wrong")

    training_pool_index = None
    if is_population_design:
        # Which object THIS design actually trained against, under the 1200-entry
        # training pool. Usually not one of the 64 shown here; the parent says so
        # rather than letting the viewer imply otherwise.
        design_index = int(args.design.split("_")[-1])
        training_pool_index = design_index % len(pool)

    ready = {
        "design": args.design,
        "is_population_design": is_population_design,
        "robot_urdf": str(resolve_repo_path(spec.urdf_path)),
        "base_pos": [float(v) for v in spec.base_pos],
        "base_rot": [float(v) for v in spec.base_rot],
        "joint_names": list(env.robot.data.joint_names),
        "num_envs": args.num_envs,
        "num_joints": spec.num_joints,
        "obs_dim": int(cfg.observation_space),
        "pool_size": len(pool),
        "object_urdf_paths": [env._object_urdf_paths[i] for i in live_pool_index],
        "object_pool_index": live_pool_index,
        "object_labels": [pool[i].label() for i in live_pool_index],
        "object_types": [pool[i].type for i in live_pool_index],
        "training_pool_index": training_pool_index,
        "training_object_label": (pool[training_pool_index].label()
                                  if training_pool_index is not None else None),
        "goals_per_episode": args.goals_per_episode,
        "eval_success_tolerance": protocol["termination.eval_success_tolerance"],
        **design_meta,
    }

    # ViserUrdf wants URDF actuated-joint order and the sim reports parser order;
    # the parent maps by name using joint_names above.
    def snapshot() -> dict:
        origin = env.scene.env_origins.cpu().numpy()
        return {
            "joint_pos": env.robot.data.joint_pos.cpu().numpy().astype(np.float32),
            "object": _pose7(env.object.data.root_pos_w.cpu().numpy() - origin,
                             env.object.data.root_quat_w.cpu().numpy()),
            "goal": _pose7(env.goal_viz.data.root_pos_w.cpu().numpy() - origin,
                           env.goal_viz.data.root_quat_w.cpu().numpy()),
            "table": _pose7(env.table.data.root_pos_w.cpu().numpy() - origin,
                            env.table.data.root_quat_w.cpu().numpy()),
            "successes": env._successes.cpu().numpy().astype(np.int32),
            "lifted": env._lifted_object.cpu().numpy().astype(bool),
        }

    conn.send(("ready", ready, snapshot()))

    state = {"running": False, "once": False, "step": 0,
             "episodes": [0] * args.num_envs, "goals": [[] for _ in range(args.num_envs)]}

    while True:
        # Drain commands, then act. Never read sim state from anywhere but this
        # loop: PhysX refuses a fetch issued while a step is in flight.
        while conn.poll(0 if (state["running"] or state["once"]) else 0.05):
            cmd = conn.recv()
            if cmd == "run":
                state["running"] = True
            elif cmd == "pause":
                state["running"] = False
            elif cmd == "step":
                state["once"] = True
            elif cmd == "reset":
                obs, _ = env.reset()
                player.player.init_rnn()
                state["step"] = 0
                conn.send(("state", snapshot(), 0, state["episodes"]))
            elif cmd == "quit":
                return

        if not (state["running"] or state["once"]):
            continue
        state["once"] = False

        policy_obs = obs["policy"].to(args.rl_device)
        action = player.get_normalized_action(policy_obs, deterministic_actions=True)
        obs, _rew, terminated, truncated, _ = env.step(action.to(env.device))
        state["step"] += 1

        done = (terminated | truncated)
        if bool(done.any()):
            final = env._prev_episode_successes.cpu().numpy()
            for e in done.nonzero(as_tuple=True)[0].cpu().numpy().tolist():
                state["episodes"][e] += 1
                state["goals"][e].append(int(final[e]))

        conn.send(("state", snapshot(), state["step"],
                   [len(g) for g in state["goals"]]))


def main() -> int:
    args = build_parser().parse_args()
    conn = Client((args.worker_host, args.worker_port),
                  authkey=bytes.fromhex(args.worker_authkey))
    try:
        run(conn, args)
    except Exception as exc:  # noqa: BLE001 - the parent must see the traceback
        try:
            conn.send(("error", f"{exc}\n{traceback.format_exc()}"))
        except Exception:
            pass
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
    # Kit's shutdown handler calls os._exit while unwinding, which skips flushes;
    # the explicit ones above are what make a worker traceback survive.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
