"""What is actually on the stage after cloning N envs?

Scene construction costs ~17 ms per env and ``sim.reset()`` is 77% of it, so the
question is what PhysX and USD are being asked to chew through. With
``replicate_physics=False`` every env is authored as its own subtree, and
anything the robot USD carries is duplicated N times -- including materials and
shaders, which a headless training run never renders.

This counts prims by type at two env counts and reports which types scale with
env count and which do not. A type that scales is a candidate for removal or
sharing; a type that does not is already being shared and is not worth touching.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.probe_stage_census --num_envs 256

Run it twice at different --num_envs and compare: per-env counts that stay
constant across the two runs are genuinely per-env, and multiplying them by
24,576 says what the real scene is carrying.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from pathlib import Path


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="gen_sharpa_like")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--strip_materials", action="store_true",
                        help="delete Material/Shader prims from the baked USD "
                             "before cloning, to measure what they cost")
    parser.add_argument("--env_spacing", type=float, default=1.3)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    print(f"[census] robot={args.robot} envs={args.num_envs} "
          f"strip_materials={args.strip_materials}")
    app = AppLauncher(args).app

    import tempfile

    import omni.usd
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass
    from pxr import Usd

    from genmech.robots import get_robot_spec
    from genmech.tasks.pose_reach.utils.scene_utils import (
        _apply_self_collision_filters,
        _bake_usd,
        _convert_urdf_to_usd,
        _robot_joint_drive_cfg,
    )
    from genmech.tools.multi_embodiment_demo import _ROBOT_BAKE_PROPS, _articulation_cfg

    work = Path(tempfile.mkdtemp(prefix="genmech_census_"))
    (work / "usd").mkdir(parents=True, exist_ok=True)

    spec = get_robot_spec(args.robot)
    raw = _convert_urdf_to_usd(
        spec.urdf_path, work / "usd", fix_base=True, self_collision=True,
        joint_drive=_robot_joint_drive_cfg(),
        replace_cylinders_with_capsules=spec.replace_cylinders_with_capsules)
    _apply_self_collision_filters(raw, spec.adjacent_links)
    usd = _bake_usd(raw, work / "baked", "robot", props=dict(_ROBOT_BAKE_PROPS),
                    apply_physx_articulation=True)

    # Count what ONE robot carries, before any cloning.
    src = Usd.Stage.Open(usd)
    per_robot = collections.Counter(p.GetTypeName() for p in src.Traverse())
    print(f"[census] one robot USD: {sum(per_robot.values())} prims")
    for t, n in per_robot.most_common(12):
        print(f"[census]    {str(t) or '<untyped>':<28} {n:>6}")

    if args.strip_materials:
        # Materials and shaders are never rendered headless. Removing them from
        # the source means they are not duplicated into every env.
        # Deactivate rather than RemovePrim: these prims are DEFINED in the
        # referenced base/physics layers, so removing them from the composed
        # stage is a silent no-op. Deactivation composes and actually prunes the
        # subtree. Visual geometry goes too -- headless training renders nothing,
        # and the hand carries 45 visual prims against 15 colliders.
        stage = Usd.Stage.Open(usd)
        VISUAL = ("Material", "Shader", "NodeGraph")
        n = 0
        for prim in stage.Traverse():
            if prim.GetTypeName() in VISUAL or prim.GetName() == "visuals":
                prim.SetActive(False)
                n += 1
        stage.GetRootLayer().Save()
        print(f"[census] deactivated {n} material/shader/visual prims")

    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
    art_cfg = _articulation_cfg(spec, "/World/envs/env_.*/Robot", [usd])
    Cfg = configclass(type("CensusSceneCfg", (InteractiveSceneCfg,),
                           {"__annotations__": {"robot": type(art_cfg)},
                            "robot": art_cfg}))
    t0 = time.perf_counter()
    scene = InteractiveScene(Cfg(num_envs=args.num_envs,
                                 env_spacing=args.env_spacing,
                                 replicate_physics=False, clone_in_fabric=False))
    scene_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    sim.reset()
    reset_s = time.perf_counter() - t0

    stage = omni.usd.get_context().get_stage()
    census = collections.Counter(p.GetTypeName() for p in stage.Traverse())
    total = sum(census.values())

    print(f"\n[census] scene={scene_s:.1f}s reset={reset_s:.1f}s "
          f"({(scene_s + reset_s) * 1000 / args.num_envs:.2f} ms/env)")
    print(f"[census] {total:,} prims on the stage for {args.num_envs} envs "
          f"({total / args.num_envs:.1f} per env)")
    print(f"[census] {'type':<28} {'count':>9} {'per env':>9}")
    for t, n in census.most_common(14):
        print(f"[census] {str(t) or '<untyped>':<28} {n:>9,} {n / args.num_envs:>9.1f}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
