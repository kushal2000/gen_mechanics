"""Where does scene construction time actually go?

At 24,576 envs the step-rate benchmark spends 6-10 MINUTES building the scene
and ~8 seconds timing physics. With 64 distinct hands it spent over two hours.
Scene construction, not stepping, is what makes a morphology sweep expensive, so
it is worth knowing which part of it is slow before optimising anything.

Four candidate causes, each isolated by a flag here:

  ENV COUNT      Sweep --num_envs. If build time is linear in envs, the cost is
                 per-env work (prim instantiation, PhysX parsing) and the fix is
                 to make each env cheaper. If it is superlinear, something is
                 scanning the whole stage per env.

  COLLIDER TYPE  SHARPA (34 convex-hull meshes) built in 634s where the capsule
                 hands took ~390-420s at identical env count. Compare
                 --robot sharpa_iiwa14 against a generated hand.

  DISTINCT USDs  --n_usds. One USD replicated is the cheap case; N distinct USDs
                 go through MultiUsdFileCfg, which resolves prim paths per env
                 against the USD stage. This is the case that took 2h.

  INSTANCING     --instanceable puts geometry in a shared prototype each env
                 references rather than deep-copying it. Usually the single
                 biggest win for large env counts -- but the robot pipeline
                 authors FilteredPairsAPI per link after conversion, and
                 instance proxies cannot be edited in place, so this reports
                 whether the self-collision filters SURVIVED as well as whether
                 it was faster. A faster scene with the filters silently
                 dropped would be a regression, not an optimisation.

  LOG VOLUME     --quiet suppresses Kit warnings. The 64-USD run wrote 2.3 GB of
                 "Unresolved reference" warnings (one per geometry-less link per
                 env), so some of that wall clock may be log I/O rather than work.

Build time is split into InteractiveScene construction and sim.reset(), because
they fail for different reasons and lumping them hides which one is at fault.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.bench_scene_build --num_envs 4096 --n_usds 8
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
    parser.add_argument("--robot", default="gen_sharpa_like",
                        help="spec name; sharpa_iiwa14 to measure mesh colliders")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--n_usds", type=int, default=1,
                        help="distinct hand USDs cycled across envs (>1 uses "
                             "MultiUsdFileCfg and the cached population)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instanceable", action="store_true")
    parser.add_argument("--quiet", action="store_true",
                        help="drop Kit's log level to error")
    parser.add_argument("--env_spacing", type=float, default=1.3)
    parser.add_argument("--steps", type=int, default=0,
                        help="timed steps after warmup; >0 also reports steps/s "
                             "so build cost and step cost can be read off the "
                             "same run at the same k")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--out", default="bench_scene_build.json")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    print(f"[build] robot={args.robot} envs={args.num_envs} usds={args.n_usds} "
          f"instanceable={args.instanceable} quiet={args.quiet}")
    app = AppLauncher(args).app

    import tempfile

    import torch
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass

    if args.quiet:
        import carb
        carb.settings.get_settings().set("/log/level", "error")
        carb.settings.get_settings().set("/log/fileLogLevel", "error")

    from genmech.robots import get_robot_spec
    from genmech.robots.generated.population import load_population
    from genmech.robots.generated.synth_spec import synth_spec
    from genmech.tasks.pose_reach.utils.scene_utils import (
        _apply_self_collision_filters,
        _bake_usd,
        _convert_urdf_to_usd,
        _robot_joint_drive_cfg,
    )
    from genmech.tools.multi_embodiment_demo import _ROBOT_BAKE_PROPS, _articulation_cfg

    work = Path(tempfile.mkdtemp(prefix="genmech_scenebuild_"))
    (work / "usd").mkdir(parents=True, exist_ok=True)

    def prepare(spec, tag):
        raw = _convert_urdf_to_usd(
            spec.urdf_path, work / "usd", fix_base=True, self_collision=True,
            joint_drive=_robot_joint_drive_cfg(),
            replace_cylinders_with_capsules=spec.replace_cylinders_with_capsules,
            make_instanceable=args.instanceable,
        )
        # Whether this SUCCEEDS is the point when instancing is on: instance
        # proxies are not editable, so the filters may be silently skipped.
        filters_ok = True
        try:
            _apply_self_collision_filters(raw, spec.adjacent_links)
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            filters_ok = False
            print(f"[build] self-collision filters FAILED on {tag}: "
                  f"{type(exc).__name__}: {exc}")
        return _bake_usd(raw, work / "baked", tag, props=dict(_ROBOT_BAKE_PROPS),
                         apply_physx_articulation=True), filters_ok

    t0 = time.perf_counter()
    if args.n_usds <= 1:
        spec = get_robot_spec(args.robot)
        usd, filters_ok = prepare(spec, "robot")
        usds = [usd]
    else:
        pool = load_population(args.seed)
        if args.n_usds <= len(pool):
            hands = pool[:args.n_usds]
        else:
            # Not enough sampled designs, so repeat the pool under DISTINCT
            # names. The cost being measured here is USD composition and prim
            # resolution per distinct FILE -- the spawner does not know or care
            # that two files describe the same geometry -- so this isolates the
            # k dimension without waiting hours on the 94%-rejection sampler.
            from dataclasses import replace as _replace
            hands = [_replace(pool[i % len(pool)], name=f"{pool[i % len(pool)].name}_k{i:04d}")
                     for i in range(args.n_usds)]
            print(f"[build] {args.n_usds} distinct USDs from {len(pool)} designs "
                  f"(geometry repeats; file identity does not)")
        specs = [synth_spec(h) for h in hands]
        pairs = [prepare(s, s.name) for s in specs]
        usds = [p[0] for p in pairs]
        filters_ok = all(p[1] for p in pairs)
        spec = specs[0]
    convert_s = time.perf_counter() - t0

    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
    art_cfg = _articulation_cfg(spec, "/World/envs/env_.*/Robot", usds)
    Cfg = configclass(type("BuildSceneCfg", (InteractiveSceneCfg,),
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

    art = scene["robot"]
    # A few steps, purely to confirm the scene is usable -- a build that is fast
    # because it produced a broken articulation is not an optimisation.
    q0 = art.data.default_joint_pos.clone()
    art.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    art.set_joint_position_target(q0)
    for _ in range(max(20, args.warmup if args.steps else 20)):
        art.write_data_to_sim()
        sim.step()
        scene.update(1 / 120.0)

    sps = env_sps = 0.0
    if args.steps:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            art.write_data_to_sim()
            sim.step()
            scene.update(1 / 120.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        sps = args.steps / dt
        env_sps = args.steps * args.num_envs / dt
        print(f"[build] {sps:.1f} steps/s, {env_sps:,.0f} env-steps/s")

    result = dict(
        robot=args.robot, num_envs=args.num_envs, n_usds=len(set(usds)),
        instanceable=args.instanceable, quiet=args.quiet,
        joints=int(art.num_joints), instances=int(art.num_instances),
        convert_s=convert_s, scene_s=scene_s, reset_s=reset_s,
        build_s=scene_s + reset_s, self_collision_filters_ok=filters_ok,
        per_env_ms=(scene_s + reset_s) * 1000.0 / args.num_envs,
        sps=sps, env_sps=env_sps,
    )
    print(f"[build] scene={scene_s:.1f}s reset={reset_s:.1f}s "
          f"total={result['build_s']:.1f}s "
          f"({result['per_env_ms']:.2f} ms/env), filters_ok={filters_ok}")

    out = Path(args.out)
    rows = []
    if out.exists():
        try:
            rows = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    rows.append(result)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[build] wrote {out}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
