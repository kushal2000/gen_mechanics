#!/usr/bin/env python3
"""Grammar-solo comparison, step 1: build one variant's viable population.

Thin wrapper around ``isaacsimenvs.inhand_reorient.evolution.viable``'s
``build_viable_generation`` (the same machinery as the module's own CLI,
``python -m isaacsimenvs.inhand_reorient.evolution.viable``). The only
reason this script exists instead of calling that CLI directly: the CLI's
``main()`` builds its grasp-search ``args.train_override`` as a hardcoded
empty list, so it has no flag to change the grasp-search physics (e.g.
``env.anyrotate.grasp_canonical_profile=opposition``). This wrapper adds
``--search-override`` to thread such overrides into the same
``driver.run_grasp_search`` call the CLI already uses, without touching
``isaacsimenvs/`` itself.

Usage (run under the isaacsim venv, booted lazily -- CPU driving logic,
Kit only inside the ``grasp_cache_gen`` subprocess launches)::

    .venv_isaacsim/bin/python3 experiments/grammar_solo/build_viable_population.py \\
        --variant G_V1 --designs 16 --seed 0 --out-dir RUN_DIR/viable/G_V1 \\
        --max-batches 10 \\
        --search-override env.anyrotate.grasp_canonical_profile=opposition \\
        --search-override env.anyrotate.grasp_per_design=1000 \\
        --search-override env.anyrotate.grasp_gen_max_rounds=60 \\
        --search-override env.anyrotate.grasp_curl_frac=0.0 \\
        --search-override env.anyrotate.grasp_tip_place_frac=1.0 \\
        --search-override env.anyrotate.grasp_max_joint_speed=5.0

Writes ``<out-dir>/population.json`` (sha256-provenanced, schema matches
``population_file.POPULATION_SCHEMA``), ``<out-dir>/grasp_cache.npz`` and
``<out-dir>/viability.json`` (candidates drawn, CPU pre-filter rejects,
grasp-search pass rate, batches, seconds) -- same file layout as the
upstream CLI, so any downstream tool that reads a viable.py population
directory works unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from isaacsimenvs.inhand_reorient.evolution import archive as arch
from isaacsimenvs.inhand_reorient.evolution import driver as drv
from isaacsimenvs.inhand_reorient.evolution import viable
from isaacsimenvs.inhand_reorient.scene import population_file as pf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="single variant name, e.g. G_V1 (not comma-separated: "
                    "this script is one variant per invocation, so viability.json's rates are per-variant)")
    ap.add_argument("--probes", default="", help="comma-separated manifest hand ids (forced, searched first); "
                    "empty by default so --designs are all drawn from --variant")
    ap.add_argument("--designs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task-profile", default="hora", choices=("hora", "anyrotate"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--search-envs", type=int, default=4096)
    ap.add_argument("--max-batches", type=int, default=6)
    ap.add_argument("--search-rounds", type=int, default=40, help="default grasp_gen_max_rounds; overridden if "
                    "env.anyrotate.grasp_gen_max_rounds is also in --search-override")
    ap.add_argument("--grasps-per-design", type=int, default=64, help="default grasp_per_design; overridden if "
                    "env.anyrotate.grasp_per_design is also in --search-override")
    ap.add_argument("--search-timeout-s", type=int, default=1500)
    ap.add_argument("--max-joint-speed", type=float, default=5.0, help="grasp_max_joint_speed (rad/s; -1 off); "
                    "overridden if env.anyrotate.grasp_max_joint_speed is also in --search-override")
    ap.add_argument("--search-override", action="append", default=[], metavar="env.anyrotate.KEY=VALUE",
                    help="extra Hydra override(s) passed to every grasp_cache_gen launch (repeatable)")
    ap.add_argument("--train-python", default=str(drv.REPO_ROOT / ".venv_isaacsim" / "bin" / "python3"))
    a = ap.parse_args(argv)

    bad = [o for o in a.search_override if not o.startswith("env.")]
    if bad:
        ap.error(f"--search-override must start with 'env.' (grasp_search_cmd only forwards env.* overrides): {bad}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        train_python=a.train_python, task_profile=a.task_profile, viable_search_envs=a.search_envs,
        viable_search_timeout_s=a.search_timeout_s, viable_grasps_per_design=a.grasps_per_design,
        viable_search_rounds=a.search_rounds, train_override=list(a.search_override),
        grasp_cache_path=str(out / "grasp_cache.npz"), viable_max_joint_speed=a.max_joint_speed)
    counter = {"i": 0}

    def search(batch):
        counter["i"] += 1
        return drv.run_grasp_search(batch, args=args, run_dir=out, gen_dir=out, index=counter["i"])

    dist = {a.variant: drv.resolve_variant(a.variant)}
    probes = [h for h in a.probes.split(",") if h]
    plan, report = viable.build_viable_generation(
        0, a.designs, probes, dist, arch.Archive(), np.random.default_rng(a.seed),
        drv.IdMinter(), known={}, search=search, batch_size=a.batch_size, max_batches=a.max_batches)
    doc = pf.write_population(out / "population.json", plan.entries)
    report["population_sha256"] = doc["population_sha256"]
    report["variant"] = a.variant
    report["seed"] = a.seed
    report["search_override"] = list(a.search_override)
    report["designs"] = [{"source": m.source, "role": m.role, "digit_count": m.digit_count,
                          "joint_count": m.joint_count} for m in plan.metas]
    (out / "viability.json").write_text(json.dumps(report, indent=1))
    print(f"[build_viable_population] variant={a.variant} seed={a.seed}: {len(plan.entries)} viable design(s) -> "
          f"{out / 'population.json'}; searched {report['candidates_searched']} in "
          f"{report['grasp_search_s']:.0f} s over {report['batches']} launch(es); "
          f"founder viability {report['founder_viability_rate']}; short_by {report['short_by']}", flush=True)
    return 0 if report["short_by"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
