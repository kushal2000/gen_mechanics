"""What does morphological diversity cost per simulation step?

The morphology search needs many designs evaluated per outer iteration. Whether
that is affordable turns on one number: how much slower is a scene of N
DIFFERENT hands than a scene of N copies of one hand? If diversity is nearly
free, designs can be evaluated in parallel inside a single rollout. If it is
expensive, the search must batch by design and pay a scene rebuild per batch.

Three configurations, same env count, same physics settings, same arm:

  1. sharpa            -- one SHARPA hand, replicated into every env (29 joints)
  2. gen_sharpa_like   -- one GENERATED hand, replicated (37 joints, capsules)
  3. gen_population    -- a DIFFERENT sampled hand in every env (37 joints)

1 vs 2 isolates the generated hand's own shape: more joints (37 vs 29, ghosting
pads the template) but analytic capsules instead of convex-hull meshes. 2 vs 3
isolates diversity alone -- identical topology and joint count, differing only
in that each env holds distinct geometry.

All three run with replicate_physics=False and clone_in_fabric=False. That is
not a handicap invented here: the training env already runs that way
(env_cfg.py) because per-env distinct OBJECT USDs require it. Enabling it for
configs 1 and 2 would measure a setting the real task cannot use, and would
flatter them against config 3, which cannot use it by construction.

ONE CONFIG PER PROCESS, by necessity. Timing several in a single process is
wrong: ``SimulationContext.clear_instance()`` drops the context but leaves the
previous configuration's prims on the USD stage, so the second scene is built on
top of the first and the timing describes a robot population that is not the one
named. Use ``--report`` to print the comparison once the runs have finished.

    for c in sharpa gen_sharpa_like gen_population; do
        OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
            -m genmech.tools.bench_embodiment_sps --config $c --out bench.json
    done
    .venv_isaacsim/bin/python -m genmech.tools.bench_embodiment_sps \\
        --report --out bench.json

Reports steps/s and env-steps/s. Physics only -- no policy, observations or
reward -- so these are upper bounds on what a training loop sees, and the RATIOS
between configurations are the meaningful part, not the absolute values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

CONFIGS = ("sharpa", "gen_sharpa_like", "gen_pop_single", "gen_population")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIGS, default="sharpa")
    parser.add_argument("--out", default="bench_embodiment_sps.json",
                        help="results accumulate here, one entry per config")
    parser.add_argument("--report", action="store_true",
                        help="print the comparison from --out and exit; needs "
                             "no Isaac Sim")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=120,
                        help="untimed steps first; the opening steps include "
                             "PhysX warm-up and are not representative")
    parser.add_argument("--seed", type=int, default=0,
                        help="cached population to draw the distinct hands from")
    parser.add_argument("--env_spacing", type=float, default=1.3)

    if "--report" not in sys.argv:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def _report(path: Path) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_name = {r["config"]: r for r in rows}
    envs = {r["num_envs"] for r in rows}

    print(f"\n{'=' * 96}")
    print(f"[bench] {sorted(envs)} envs, physics only, "
          f"replicate_physics=False (as in training)")
    print(f"{'=' * 96}")
    print(f"{'config':<18} {'what':<32} {'joints':>6} {'USDs':>5} "
          f"{'steps/s':>9} {'env-steps/s':>12} {'convert':>9} {'build':>8}")
    for name in CONFIGS:
        r = by_name.get(name)
        if r is None:
            continue
        print(f"{r['config']:<18} {r['label']:<32} {r['joints']:>6} "
              f"{r['n_usds']:>5} {r['sps']:>9.1f} {r['env_sps']:>12,.0f} "
              f"{r['convert_s']:>8.1f}s {r['build_s']:>7.1f}s")

    # Diversity must be measured against a replicated hand from the SAME
    # population; gen_sharpa_like is a different (heavier) hand, so that
    # comparison would report geometry differences as a diversity cost.
    if "gen_pop_single" in by_name and "gen_population" in by_name:
        one, many = by_name["gen_pop_single"], by_name["gen_population"]
        ratio = many["sps"] / one["sps"]
        print(f"\n[bench] cost of DIVERSITY (same population, 1 hand replicated "
              f"vs 64 distinct): {ratio:.3f}x step rate ({(ratio - 1) * 100:+.1f}%)")
        print(f"[bench]   scene build {one['build_s']:.1f}s -> "
              f"{many['build_s']:.1f}s, conversion {one['convert_s']:.1f}s -> "
              f"{many['convert_s']:.1f}s (both one-off, not per rollout)")
    if "sharpa" in by_name and "gen_sharpa_like" in by_name:
        s, g = by_name["sharpa"], by_name["gen_sharpa_like"]
        print(f"[bench] cost of the GENERATED hand ({g['joints']} joints, "
              f"capsules) vs SHARPA ({s['joints']} joints, hull meshes): "
              f"{g['sps'] / s['sps']:.3f}x step rate")


def main() -> None:
    args = _parse_args()
    out_path = Path(args.out)

    if args.report:
        _report(out_path)
        return

    from isaaclab.app import AppLauncher

    print(f"[bench] config={args.config} envs={args.num_envs} "
          f"steps={args.steps} (+{args.warmup} warmup)")
    app = AppLauncher(args).app

    import tempfile

    import torch
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass

    from genmech.robots import get_robot_spec
    from genmech.robots.generated.population import load_population
    from genmech.robots.generated.synth_spec import synth_spec
    from genmech.tools.multi_embodiment_demo import (
        _articulation_cfg,
        _prepare_robot_usd,
    )

    work = Path(tempfile.mkdtemp(prefix="genmech_bench_"))
    (work / "usd").mkdir(parents=True, exist_ok=True)

    # ---- Assets for THIS config only --------------------------------------
    # Conversion is reported separately from step cost: it is a one-off per
    # population, whereas step cost is paid every rollout, and conflating them
    # would hide which one actually dominates.
    t0 = time.perf_counter()
    if args.config == "sharpa":
        spec = get_robot_spec("sharpa_iiwa14")
        usds = [_prepare_robot_usd(spec, work, "sharpa")]
        label = "1 SHARPA hand x N envs"
    elif args.config == "gen_sharpa_like":
        spec = get_robot_spec("gen_sharpa_like")
        usds = [_prepare_robot_usd(spec, work, "gen_like")]
        label = "1 generated hand x N envs"
    elif args.config == "gen_pop_single":
        # The control for gen_population. Comparing 64 distinct hands against
        # gen_sharpa_like confounds diversity with geometry: gen_sharpa_like has
        # all five fingers carrying collision geometry, while sampled hands
        # average fewer active fingers and ghosted fingers carry NO collision
        # shapes at all. Same joint count, very different contact-pair counts.
        # Replicating one hand FROM the population holds that distribution fixed
        # and leaves distinct-USDs as the only difference.
        hand = load_population(args.seed)[0]
        spec = synth_spec(hand)
        usds = [_prepare_robot_usd(spec, work, spec.name)]
        label = f"1 population hand ({hand.name}) x N"
    else:
        hands = load_population(args.seed)
        if len(hands) < args.num_envs:
            print(f"[bench] cached population has {len(hands)} hands for "
                  f"{args.num_envs} envs; hands will repeat. Build more with "
                  f"build_hand_population --count {args.num_envs}")
        # Convert each DISTINCT hand once. MultiUsdFileCfg cycles a short list
        # across envs by itself, so expanding to one entry per env converts the
        # same 64 designs 24,576 times -- ~7 hours of pure conversion at 24k
        # envs, which is what made this config look like a scaling wall.
        specs = [synth_spec(h) for h in hands]
        usds = [_prepare_robot_usd(s, work, s.name) for s in specs]
        spec = specs[0]
        label = f"{len(usds)} distinct hands, cycled over envs"
    convert_s = time.perf_counter() - t0

    # ---- Scene ------------------------------------------------------------
    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
    art_cfg = _articulation_cfg(spec, "/World/envs/env_.*/Robot", usds)
    Cfg = configclass(type("BenchSceneCfg", (InteractiveSceneCfg,),
                           {"__annotations__": {"robot": type(art_cfg)},
                            "robot": art_cfg}))
    t0 = time.perf_counter()
    scene = InteractiveScene(Cfg(num_envs=args.num_envs,
                                 env_spacing=args.env_spacing,
                                 replicate_physics=False, clone_in_fabric=False))
    sim.reset()
    build_s = time.perf_counter() - t0

    art = scene["robot"]
    q0 = art.data.default_joint_pos.clone()
    art.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    art.set_joint_position_target(q0)

    for _ in range(args.warmup):
        art.write_data_to_sim()
        sim.step()
        scene.update(1 / 120.0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # else the timer measures queue depth, not work
    t0 = time.perf_counter()
    for _ in range(args.steps):
        art.write_data_to_sim()
        sim.step()
        scene.update(1 / 120.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    result = dict(
        config=args.config, label=label, num_envs=args.num_envs,
        joints=int(art.num_joints), n_usds=len(set(usds)),
        convert_s=convert_s, build_s=build_s,
        steps=args.steps, seconds=dt,
        sps=args.steps / dt, env_sps=args.steps * args.num_envs / dt,
    )
    print(f"[bench] {args.config}: {result['sps']:.1f} steps/s, "
          f"{result['env_sps']:,.0f} env-steps/s "
          f"({result['joints']} joints, {result['n_usds']} USDs)")

    rows = []
    if out_path.exists():
        try:
            rows = [r for r in json.loads(out_path.read_text(encoding="utf-8"))
                    if r["config"] != args.config]
        except json.JSONDecodeError:
            pass
    rows.append(result)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[bench] wrote {out_path}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
