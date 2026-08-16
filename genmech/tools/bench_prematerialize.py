"""Does pre-materialising env prims before spawning scale better?

Two code paths reach the same scene, and every scene-build number measured so far
used only the first:

  A. SCENE-CFG path (what the benchmarks used). The ArticulationCfg lives in the
     InteractiveSceneCfg, so InteractiveScene clones env_0 and spawns the robot
     as part of cloning. Cloning and spawning are interleaved, and each spawn
     resolves its regex prim path against a stage that is still growing.

  B. PRE-MATERIALISE path (what genmech's training env actually does). The scene
     cfg is BARE -- num_envs and spacing only -- so InteractiveScene clones empty
     envs. scene_utils then calls _materialize_env_prims() to define every
     /World/envs/env_N Xform up front, and only afterwards constructs the
     Articulation against a stage whose env structure is already complete.

If the distinct-USD cost is superlinear because each spawn scans or recomposes a
growing stage, B should scale far better: the stage stops growing before any
spawning starts. If both scale the same, that hypothesis is wrong and the cost
is somewhere else.

Everything else is held identical -- same USDs, same env count, same physics
settings, same replicate_physics=False.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.bench_prematerialize --num_envs 8192 --n_usds 64 --path B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=("A", "B"), default="B",
                        help="A = ArticulationCfg inside the scene cfg; "
                             "B = bare scene cfg, materialise env prims, then spawn")
    parser.add_argument("--robot", default="gen_sharpa_like")
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--n_usds", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_spacing", type=float, default=1.3)
    parser.add_argument("--out", default="bench_prematerialize.json")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    print(f"[pre] path={args.path} robot={args.robot} envs={args.num_envs} "
          f"usds={args.n_usds}")
    app = AppLauncher(args).app

    import tempfile

    import torch
    from isaaclab.assets import Articulation
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass
    from isaacsim.core.utils.stage import get_current_stage

    from genmech.robots import get_robot_spec
    from genmech.robots.generated.population import load_population
    from genmech.robots.generated.synth_spec import synth_spec
    from genmech.tools.multi_embodiment_demo import _articulation_cfg, _prepare_robot_usd

    work = Path(tempfile.mkdtemp(prefix="genmech_premat_"))
    (work / "usd").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.n_usds <= 1:
        spec = get_robot_spec(args.robot)
        usds = [_prepare_robot_usd(spec, work, "robot")]
    else:
        hands = load_population(args.seed)[:args.n_usds]
        specs = [synth_spec(h) for h in hands]
        usds = [_prepare_robot_usd(s, work, s.name) for s in specs]
        spec = specs[0]
    convert_s = time.perf_counter() - t0
    print(f"[pre] converted {len(usds)} USDs in {convert_s:.1f}s")

    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
    art_cfg = _articulation_cfg(spec, "/World/envs/env_.*/Robot", usds)

    materialize_s = 0.0
    if args.path == "A":
        Cfg = configclass(type("SceneCfgA", (InteractiveSceneCfg,),
                               {"__annotations__": {"robot": type(art_cfg)},
                                "robot": art_cfg}))
        t0 = time.perf_counter()
        scene = InteractiveScene(Cfg(num_envs=args.num_envs,
                                     env_spacing=args.env_spacing,
                                     replicate_physics=False, clone_in_fabric=False))
        clone_s = time.perf_counter() - t0
        spawn_s = 0.0
    else:
        # Bare scene cfg: clone empty envs only.
        Cfg = configclass(type("SceneCfgB", (InteractiveSceneCfg,), {}))
        t0 = time.perf_counter()
        scene = InteractiveScene(Cfg(num_envs=args.num_envs,
                                     env_spacing=args.env_spacing,
                                     replicate_physics=False, clone_in_fabric=False))
        clone_s = time.perf_counter() - t0

        # Define every env Xform up front, so the stage structure is complete
        # before a single robot is spawned.
        t0 = time.perf_counter()
        stage = get_current_stage()
        for env_path in scene.env_prim_paths:
            if not stage.GetPrimAtPath(env_path).IsValid():
                stage.DefinePrim(env_path, "Xform")
        materialize_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        robot = Articulation(art_cfg)
        scene.articulations["robot"] = robot
        spawn_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    sim.reset()
    reset_s = time.perf_counter() - t0

    art = scene["robot"]
    q0 = art.data.default_joint_pos.clone()
    art.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    art.set_joint_position_target(q0)
    for _ in range(20):
        art.write_data_to_sim()
        sim.step()
        scene.update(1 / 120.0)

    build_s = clone_s + materialize_s + spawn_s + reset_s
    result = dict(path=args.path, robot=args.robot, num_envs=args.num_envs,
                  n_usds=len(set(usds)), joints=int(art.num_joints),
                  instances=int(art.num_instances), convert_s=convert_s,
                  clone_s=clone_s, materialize_s=materialize_s, spawn_s=spawn_s,
                  reset_s=reset_s, build_s=build_s,
                  ms_per_env=build_s * 1000.0 / args.num_envs)
    print(f"[pre] clone={clone_s:.1f}s materialize={materialize_s:.1f}s "
          f"spawn={spawn_s:.1f}s reset={reset_s:.1f}s "
          f"TOTAL={build_s:.1f}s ({result['ms_per_env']:.2f} ms/env)")
    print(f"[pre] instances={result['instances']} (must equal num_envs)")

    out = Path(args.out)
    rows = []
    if out.exists():
        try:
            rows = [r for r in json.loads(out.read_text(encoding="utf-8"))
                    if not (r["path"] == args.path and r["num_envs"] == args.num_envs
                            and r["n_usds"] == result["n_usds"])]
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
