"""Inside spawn_multi_asset: which loop actually costs the time?

Reading the source says the spawner is O(k) + O(n):

    for asset_cfg in cfg.assets_cfg:          # k times: load each USD once
        asset_cfg.func(f"{template}/Asset_{i:04d}", ...)
    with Sdf.ChangeBlock():
        for prim_path in prim_paths:          # n times: one CopySpec per env
            Sdf.CopySpec(layer, proto_path, layer, prim_path)

But measurement says otherwise: at 24,576 envs the cost per distinct design is
3.4 s, while at 4,096 envs the same slope is 0.79 s -- so per-design cost scales
with ENV COUNT, which neither loop should do. Predicted k*n cost matched to 3%
at k=256, so the effect is real and the structural reading is missing something.

Three places the hidden n-dependence could live, and this separates them by
timing each directly rather than inferring from totals:

  TEMPLATE SPAWN   asset_cfg.func loads a USD into /World/Template. If each load
                   recomposes or re-parses a stage that already holds n env
                   subtrees, this is O(k*n) wearing an O(k) shape.

  COPYSPEC         Sdf.CopySpec per env. If copy cost grows as the layer fills,
                   the O(n) loop is really O(n^2) -- and with k protos the
                   layer is k robots bigger before copying even starts.

  PATH RESOLUTION  find_matching_prim_paths(root_path) scans the stage once per
                   spawn call to resolve the regex.

Run at several k with n FIXED. If template-spawn time grows with k while
copy time stays flat, the cost is asset loading; if copy time grows, it is the
layer. Either answer points at a different fix.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.probe_spawn_profile --num_envs 4096 --n_usds 64
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
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--n_usds", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_spacing", type=float, default=1.3)
    parser.add_argument("--out", default="spawn_profile.json")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    print(f"[spawn] envs={args.num_envs} usds={args.n_usds}")
    app = AppLauncher(args).app

    import tempfile

    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass
    from pxr import Sdf

    from genmech.robots.generated.population import load_population
    from genmech.robots.generated.synth_spec import synth_spec
    from genmech.tools.multi_embodiment_demo import _articulation_cfg, _prepare_robot_usd

    # ---- Instrument the three suspects ------------------------------------
    # Patched on the MODULE the spawner resolves them through, so the spawner's
    # own calls are the ones being timed -- not a copy of the logic re-implemented
    # here, which could differ from what actually runs.
    prof = {"copyspec_s": 0.0, "copyspec_n": 0,
            "find_paths_s": 0.0, "find_paths_n": 0,
            "template_spawn_s": 0.0, "template_spawn_n": 0}

    orig_copy = Sdf.CopySpec
    # Per-copy timings, to see whether the Nth copy costs more than the first.
    copy_samples: list[float] = []

    def timed_copy(*a, **kw):
        t0 = time.perf_counter()
        r = orig_copy(*a, **kw)
        dt = time.perf_counter() - t0
        prof["copyspec_s"] += dt
        prof["copyspec_n"] += 1
        copy_samples.append(dt)
        return r

    Sdf.CopySpec = timed_copy

    # Patch on isaaclab.sim.utils: the spawner resolves this through
    # `sim_utils.find_matching_prim_paths` at CALL time, so replacing the module
    # attribute is enough. The wrappers submodule itself is not importable by
    # name (its package __init__ re-exports the functions, not the module).
    import isaaclab.sim.utils as isaaclab_sim_utils

    orig_find = isaaclab_sim_utils.find_matching_prim_paths

    def timed_find(*a, **kw):
        t0 = time.perf_counter()
        r = orig_find(*a, **kw)
        prof["find_paths_s"] += time.perf_counter() - t0
        prof["find_paths_n"] += 1
        return r

    isaaclab_sim_utils.find_matching_prim_paths = timed_find

    work = Path(tempfile.mkdtemp(prefix="genmech_spawnprof_"))
    (work / "usd").mkdir(parents=True, exist_ok=True)

    pool = load_population(args.seed)
    if args.n_usds > len(pool):
        from dataclasses import replace as _replace
        hands = [_replace(pool[i % len(pool)], name=f"{pool[i % len(pool)].name}_k{i:04d}")
                 for i in range(args.n_usds)]
    else:
        hands = pool[:args.n_usds]
    specs = [synth_spec(h) for h in hands]
    t0 = time.perf_counter()
    usds = [_prepare_robot_usd(s, work, s.name) for s in specs]
    convert_s = time.perf_counter() - t0
    print(f"[spawn] converted {len(usds)} USDs in {convert_s:.1f}s")

    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
    art_cfg = _articulation_cfg(specs[0], "/World/envs/env_.*/Robot", usds)

    # Wrap each asset cfg's own `func`. Patching the module that DEFINES
    # spawn_from_usd would not work: UsdFileCfg captures the function object as a
    # field default when the cfg is constructed, so the already-built cfgs would
    # keep calling the original.
    def _wrap(cfg_obj):
        inner = cfg_obj.func

        def timed(*a, **kw):
            t0 = time.perf_counter()
            r = inner(*a, **kw)
            prof["template_spawn_s"] += time.perf_counter() - t0
            prof["template_spawn_n"] += 1
            return r

        cfg_obj.func = timed

    for sub in getattr(art_cfg.spawn, "assets_cfg", []):
        _wrap(sub)
    Cfg = configclass(type("SpawnProfCfg", (InteractiveSceneCfg,),
                           {"__annotations__": {"robot": type(art_cfg)},
                            "robot": art_cfg}))
    t0 = time.perf_counter()
    InteractiveScene(Cfg(num_envs=args.num_envs, env_spacing=args.env_spacing,
                         replicate_physics=False, clone_in_fabric=False))
    scene_s = time.perf_counter() - t0

    n = len(copy_samples)
    first = sum(copy_samples[: max(1, n // 10)]) / max(1, n // 10)
    last = sum(copy_samples[-max(1, n // 10):]) / max(1, n // 10)

    result = dict(num_envs=args.num_envs, n_usds=args.n_usds, convert_s=convert_s,
                  scene_s=scene_s, **prof,
                  copy_first_decile_ms=first * 1000, copy_last_decile_ms=last * 1000,
                  unaccounted_s=scene_s - prof["copyspec_s"] - prof["find_paths_s"]
                  - prof["template_spawn_s"])
    print(f"\n[spawn] scene construction {scene_s:.1f}s, of which:")
    print(f"[spawn]   template spawn (k={prof['template_spawn_n']} loads) "
          f"{prof['template_spawn_s']:>8.1f}s")
    print(f"[spawn]   Sdf.CopySpec   (n={prof['copyspec_n']} copies) "
          f"{prof['copyspec_s']:>8.1f}s")
    print(f"[spawn]   find_matching_prim_paths (x{prof['find_paths_n']}) "
          f"{prof['find_paths_s']:>8.1f}s")
    print(f"[spawn]   unaccounted                        "
          f"{result['unaccounted_s']:>8.1f}s")
    print(f"[spawn] copy cost first decile {first * 1000:.3f} ms -> "
          f"last decile {last * 1000:.3f} ms "
          f"({'GROWS' if last > 1.5 * first else 'flat'})")

    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[spawn] wrote {out}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
