"""Record which object each embodiment was trained against, as JSON.

A population training run pairs every environment with a distinct design *and* a
distinct object, and writes neither pairing down. This recovers both from the
run's own ``.hydra/config.yaml`` and the population manifest, and writes them to
``coevolution/population/assignments/<run_name>.json``.

    .venv_isaacsim/bin/python -m coevolution.population.dump_assignment \\
        --run_dir debug_outputs/deprecated/gen_mechanics/multi_embodiment_control/mec_population24k_seed0_2026-08-17_15-13-28

No Isaac Sim needed -- this reads a config and replays a numpy RNG.

**The finding this file exists to make legible: each design saw exactly ONE
object for the whole run.** The USD bound to an environment prim is fixed at
scene build and cannot be swapped at runtime, so environment ``i`` held pool
entry ``i % pool_size`` from the first step to the last. At the usual operating
point (population count == num_envs) design ``i`` therefore has exactly one
object, and designs ``i``, ``i + pool_size``, ``i + 2*pool_size``, ... share an
object exactly. ``shared_object_groups`` in the output is that grouping, which
is what you need to compare designs without the object confounding the result.

Verify against the simulator before trusting it for anything that matters --
``genmech/tools/check_object_identity.py`` reads the live
``_object_asset_index_per_env``. An env->pool assignment bug once gave 510 of 512
environments the wrong asset while every offline check passed
(``docs/multi_embodiment.md`` section 4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from coevolution.population.object_pool import (
    design_index_for_env,
    pool_index_for_env,
    reconstruct_pool,
    type_counts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSIGNMENT_DIR = REPO_ROOT / "genmech" / "eval" / "embodiments" / "assignments"

SCHEMA_VERSION = 1


def _get(cfg: dict, dotted: str) -> Any:
    """Read a dotted path out of a raw hydra config, with a legible failure."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"{dotted!r} missing from config (stopped at {part!r})")
        node = node[part]
    return node


def load_run_config(run_dir: Path) -> dict:
    """The resolved config a run was launched with.

    Parsed with ``yaml.safe_load`` rather than OmegaConf on purpose: the file
    contains interpolations like ``${....env.scene.num_envs}`` that only resolve
    inside the full training config, and every field needed here is a literal.
    """
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"no {config_path}; is {run_dir} a hydra run directory? "
            "Runs that died during scene setup often have no .hydra at all."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_assignment(run_dir: Path, sanity_check: bool = True) -> dict:
    """The full assignment record for a population run."""
    cfg = load_run_config(run_dir)

    population_seed = _get(cfg, "env.assets.robot_population_seed")
    if population_seed is None:
        raise ValueError(
            f"{run_dir.name} has robot_population_seed=None, so it is a "
            "single-embodiment run: every env held the same robot "
            f"({_get(cfg, 'env.assets.robot_spec')!r}). Nothing to assign."
        )

    handle_head_types = list(_get(cfg, "env.assets.handle_head_types"))
    num_assets_per_type = int(_get(cfg, "env.assets.num_assets_per_type"))
    object_seed = int(_get(cfg, "env.assets.object_seed"))
    shuffle_assets = bool(_get(cfg, "env.assets.shuffle_assets"))
    density_scale = float(_get(cfg, "env.assets.object_density_scale"))
    num_envs = int(_get(cfg, "env.scene.num_envs"))
    population_count = int(_get(cfg, "env.assets.robot_population_count"))

    pool = reconstruct_pool(
        handle_head_types=handle_head_types,
        num_assets_per_type=num_assets_per_type,
        object_seed=object_seed,
        shuffle=shuffle_assets,
        density_scale=density_scale,
    )
    pool_size = len(pool)

    # Imported lazily: it reads a 24k-entry manifest, and the errors above should
    # fire before paying for that.
    from hand_sampler.population import load_population

    hands = load_population(int(population_seed))
    # 0 means "the whole cached population" -- scene_utils._resolve_robot_population.
    resolved_count = len(hands) if population_count == 0 else population_count
    if resolved_count > len(hands):
        raise ValueError(
            f"robot_population_count={population_count} exceeds the "
            f"{len(hands)} designs in seed {population_seed}'s manifest."
        )
    hands = hands[:resolved_count]

    assignment = []
    shared: dict[int, list[int]] = {}
    for env_id in range(num_envs):
        design_index = design_index_for_env(env_id, resolved_count)
        pool_index = pool_index_for_env(env_id, pool_size)
        hand = hands[design_index]
        assignment.append({
            "env": env_id,
            "design_index": design_index,
            "design": hand.name,
            "pool_index": pool_index,
            "n_active_fingers": hand.n_active_fingers,
            "n_active_joints": hand.n_active_joints,
        })
        shared.setdefault(pool_index, []).append(design_index)

    if sanity_check:
        _sanity_check(assignment, num_envs, pool_size, resolved_count)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "population": {
            "seed": int(population_seed),
            "count": resolved_count,
            "count_config": population_count,
            "num_envs": num_envs,
            "one_design_per_env": resolved_count == num_envs,
        },
        "object_pool": {
            "handle_head_types": handle_head_types,
            "num_assets_per_type": num_assets_per_type,
            "object_seed": object_seed,
            "shuffle_assets": shuffle_assets,
            "object_density_scale": density_scale,
            "object_restitution": _get(cfg, "env.assets.object_restitution"),
            "object_friction": _get(cfg, "env.assets.object_friction"),
            "pool_size": pool_size,
            "type_counts": type_counts(pool),
            "objects_per_design": 1,
            "note": (
                "One object per design for the entire run: the USD bound to an "
                "env prim is fixed at scene build. Designs sharing a pool_index "
                "held an identical object -- see shared_object_groups."
            ),
        },
        "pool": [entry.to_json() for entry in pool],
        "assignment": assignment,
        # Keyed by str because JSON object keys are strings; reload with int().
        "shared_object_groups": {str(k): v for k, v in sorted(shared.items())},
    }


def _sanity_check(assignment, num_envs: int, pool_size: int, count: int) -> None:
    """Fail loudly if the assignment rule ever stops being ``env % n``.

    This is the guard against the failure mode that actually happens: the rule
    changes in scene_utils, this file keeps emitting the old one, and the record
    is confidently wrong rather than obviously broken.
    """
    if len(assignment) != num_envs:
        raise AssertionError(f"expected {num_envs} rows, built {len(assignment)}")
    for row in assignment:
        expected_pool = row["env"] % pool_size
        expected_design = row["env"] % count
        if row["pool_index"] != expected_pool:
            raise AssertionError(
                f"env {row['env']}: pool_index {row['pool_index']} != {expected_pool}"
            )
        if row["design_index"] != expected_design:
            raise AssertionError(
                f"env {row['env']}: design_index {row['design_index']} != {expected_design}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run_dir", required=True,
                        help="A hydra run directory containing .hydra/config.yaml")
    parser.add_argument("--out", default=None,
                        help="Output JSON (default: assignments/<run_name>.json)")
    parser.add_argument("--no_sanity_check", action="store_true",
                        help="Skip the env%%n assertions (not recommended)")
    parser.add_argument("--indent", type=int, default=2,
                        help="Pretty-print indent (default 2)")
    parser.add_argument("--compact", action="store_true",
                        help="Write one long line instead; ~25%% smaller, unreadable")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    record = build_assignment(run_dir, sanity_check=not args.no_sanity_check)

    out = Path(args.out) if args.out else DEFAULT_ASSIGNMENT_DIR / f"{run_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=None if args.compact else args.indent)

    pop, pool_meta = record["population"], record["object_pool"]
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size_mb:.1f} MB)")
    print(f"  population seed {pop['seed']}, {pop['count']} designs, "
          f"{pop['num_envs']} envs, one_design_per_env={pop['one_design_per_env']}")
    print(f"  object pool {pool_meta['pool_size']} entries "
          f"(seed {pool_meta['object_seed']}, {pool_meta['num_assets_per_type']}/type)")
    print(f"  type counts: {pool_meta['type_counts']}")
    print(f"  {len(record['shared_object_groups'])} shared-object groups, "
          f"~{pop['count'] / max(1, len(record['shared_object_groups'])):.0f} designs each")
    first = record["assignment"][0]
    print(f"  e.g. {first['design']} -> pool[{first['pool_index']}] = "
          f"{record['pool'][first['pool_index']]['type']}")


if __name__ == "__main__":
    main()
