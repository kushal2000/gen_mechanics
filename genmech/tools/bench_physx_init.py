"""What makes ``sim.reset()`` slow, and which knobs actually move it?

Scene construction at 24,576 envs costs ~440 s, and ``sim.reset()`` is 77% of
that. Everything measured so far has attacked the other 23%: USD instancing did
nothing (69.2s vs 68.9s), and deactivating every material, shader and visual
prim halved the stage but bought only 6.5%. Prim volume is not the bottleneck.

``sim.reset()`` is where PhysX builds the simulation: articulations, links,
joints, collision shapes, filter pairs, and the GPU buffers sized by PhysxCfg.
This isolates the candidates one at a time, all at a fixed env count:

  GPU BUFFERS      Training sets gpu_max_rigid_contact_count=16.7M and
                   gpu_max_rigid_patch_count=8.4M -- sized so 24,576-env
                   close-contact grasping does not overflow the patch buffer.
                   Those are allocated during reset. If allocation dominates,
                   init time is nearly independent of the robot and the fix is
                   sizing, not geometry.

  SELF-COLLISION   enabled_self_collisions=True makes PhysX consider intra-
                   articulation shape pairs, and the robot pipeline authors
                   55-71 FilteredPairsAPI relationships per robot on top. At
                   24,576 envs that is ~1.7M filter relationships to build.

  COLLISION SHAPES The arm contributes 8 collision MESHES to every env in every
                   configuration measured. SHARPA (34 meshes) costs 25.5 ms/env
                   against 16.8 for the capsule hand, so mesh colliders are the
                   one category already shown to be expensive.

Reports scene construction and reset separately, because only the second is the
target here.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.bench_physx_init --num_envs 2048 --variant baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

VARIANTS = ("baseline", "training_buffers", "small_buffers", "no_self_collision",
            "no_arm_collision", "replicate_physics")


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--robot", default="gen_sharpa_like")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--env_spacing", type=float, default=1.3)
    parser.add_argument("--out", default="bench_physx_init.json")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    print(f"[physx] variant={args.variant} robot={args.robot} envs={args.num_envs}")
    app = AppLauncher(args).app

    import tempfile

    import torch
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
    from isaaclab.utils import configclass
    from pxr import Usd, UsdPhysics

    from genmech.robots import get_robot_spec
    from genmech.tasks.pose_reach.utils.scene_utils import (
        _apply_self_collision_filters,
        _bake_usd,
        _convert_urdf_to_usd,
        _robot_joint_drive_cfg,
    )
    from genmech.tools.build_hand_urdf import ARM_LINKS
    from genmech.tools.multi_embodiment_demo import _ROBOT_BAKE_PROPS, _articulation_cfg

    work = Path(tempfile.mkdtemp(prefix="genmech_physxinit_"))
    (work / "usd").mkdir(parents=True, exist_ok=True)

    spec = get_robot_spec(args.robot)
    self_collision = args.variant != "no_self_collision"

    t0 = time.perf_counter()
    raw = _convert_urdf_to_usd(
        spec.urdf_path, work / "usd", fix_base=True, self_collision=self_collision,
        joint_drive=_robot_joint_drive_cfg(),
        replace_cylinders_with_capsules=spec.replace_cylinders_with_capsules)
    if self_collision:
        _apply_self_collision_filters(raw, spec.adjacent_links)
    props = dict(_ROBOT_BAKE_PROPS)
    if not self_collision:
        props["enabled_self_collisions"] = False
    usd = _bake_usd(raw, work / "baked", "robot", props=props,
                    apply_physx_articulation=True)

    n_disabled = 0
    if args.variant == "no_arm_collision":
        # The arm is identical in every env and every configuration, and it
        # carries 8 collision MESHES. Disable just those, leaving the hand's
        # colliders untouched, to price the arm separately from the hand.
        stage = Usd.Stage.Open(usd)
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.CollisionAPI) and not prim.HasAPI(
                    UsdPhysics.CollisionAPI):
                continue
            if any(link in str(prim.GetPath()) for link in ARM_LINKS):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                n_disabled += 1
        stage.GetRootLayer().Save()
        print(f"[physx] disabled {n_disabled} arm collision shapes")
    convert_s = time.perf_counter() - t0

    # PhysX buffer sizing. The default variant deliberately uses Isaac Lab's
    # defaults, which is what every earlier measurement in this repo used --
    # training uses far larger buffers, so "baseline" is not the training
    # configuration and the two are reported separately rather than conflated.
    physx = PhysxCfg(solver_type=1,
                     min_position_iteration_count=8, max_position_iteration_count=8,
                     min_velocity_iteration_count=0, max_velocity_iteration_count=0)
    if args.variant == "training_buffers":
        physx.gpu_max_rigid_contact_count = 16777216
        physx.gpu_max_rigid_patch_count = 8388608
    elif args.variant == "small_buffers":
        physx.gpu_max_rigid_contact_count = 1048576
        physx.gpu_max_rigid_patch_count = 524288

    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device,
                                          physx=physx))
    art_cfg = _articulation_cfg(spec, "/World/envs/env_.*/Robot", [usd])
    Cfg = configclass(type("PhysxInitSceneCfg", (InteractiveSceneCfg,),
                           {"__annotations__": {"robot": type(art_cfg)},
                            "robot": art_cfg}))
    # replicate_physics=True lets PhysX parse ONE env and replicate it, instead
    # of authoring and parsing all N independently. It forbids per-env distinct
    # USDs -- which is the only reason this repo turned it off -- so this variant
    # prices what heterogeneity costs at INIT, as opposed to the 7.6% it costs
    # per step.
    replicate = args.variant == "replicate_physics"
    t0 = time.perf_counter()
    scene = InteractiveScene(Cfg(num_envs=args.num_envs,
                                 env_spacing=args.env_spacing,
                                 replicate_physics=replicate,
                                 clone_in_fabric=False))
    scene_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    sim.reset()
    reset_s = time.perf_counter() - t0

    art = scene["robot"]
    q0 = art.data.default_joint_pos.clone()
    art.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    art.set_joint_position_target(q0)
    for _ in range(20):          # a scene that builds fast but cannot step is
        art.write_data_to_sim()  # not an optimisation
        sim.step()
        scene.update(1 / 120.0)

    result = dict(variant=args.variant, robot=args.robot, num_envs=args.num_envs,
                  joints=int(art.num_joints), convert_s=convert_s,
                  scene_s=scene_s, reset_s=reset_s,
                  reset_ms_per_env=reset_s * 1000.0 / args.num_envs,
                  total_ms_per_env=(scene_s + reset_s) * 1000.0 / args.num_envs,
                  arm_shapes_disabled=n_disabled)
    print(f"[physx] scene={scene_s:.1f}s RESET={reset_s:.1f}s "
          f"({result['reset_ms_per_env']:.2f} ms/env reset, "
          f"{result['total_ms_per_env']:.2f} total)")

    out = Path(args.out)
    rows = []
    if out.exists():
        try:
            rows = [r for r in json.loads(out.read_text(encoding="utf-8"))
                    if not (r["variant"] == args.variant
                            and r["num_envs"] == args.num_envs)]
        except json.JSONDecodeError:
            pass
    rows.append(result)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
