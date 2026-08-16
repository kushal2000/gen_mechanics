"""Convert each robot URDF to USD once, keyed by CONTENT.

URDF->USD conversion costs ~1.1 s per design and every run redoes it into a
fresh temp directory: 70 s for 64 designs, ~9 minutes for 512, paid again on
every benchmark, demo and training launch. Nothing about it depends on the run.

The cache key is a hash of the URDF TEXT plus the converter settings, not the
robot's name. Keying on name is how the population cache went wrong earlier: a
rebuilt design kept its name, the on-disk artefact was left stale, and the
simulator ran a robot that no check was looking at. A content hash cannot go
stale -- change the URDF by one character and the key changes, so the old entry
is simply never consulted again.

The settings are part of the key because the same URDF converts differently
under different flags: replace_cylinders_with_capsules turns cylinders into
capsules, self_collision changes the authored filters, fix_base changes the
articulation root. Two artefacts that differ in physics must not share a key.

Entries are never invalidated, only added. Stale ones are dead weight on disk,
not a correctness problem, and `prune` clears them when that matters.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from genmech.utils.paths import resolve as resolve_repo_path

# Beside the generated URDFs rather than in /tmp: a sweep that reruns tomorrow
# should still hit the cache, and /tmp is cleared between allocations.
USD_CACHE_DIR = "assets/usd/generated/_cache"

# Bump when the BAKE step changes in a way older artefacts do not satisfy --
# the key covers conversion inputs, not what _bake_usd does afterwards.
CACHE_VERSION = 1


def cache_root() -> Path:
    return resolve_repo_path(USD_CACHE_DIR)


def cache_key(urdf_path: Path, **settings) -> str:
    """Content hash of the URDF plus the settings that change the output."""

    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}\n".encode())
    h.update(json.dumps(settings, sort_keys=True).encode())
    h.update(Path(urdf_path).read_bytes())
    return h.hexdigest()[:16]


def cached_convert_and_bake(spec, *, bake_props: dict, apply_filters: bool = True,
                            self_collision: bool = True, fix_base: bool = True,
                            make_instanceable: bool = False,
                            verbose: bool = False) -> tuple[str, bool]:
    """Return a baked USD for ``spec``, converting only on a cache miss.

    Returns ``(usd_path, was_cached)`` so callers can report the hit rate --
    a cache that silently stops hitting is otherwise invisible, and looks only
    like everything having become slow again.
    """

    from genmech.tasks.pose_reach.utils.scene_utils import (
        _apply_self_collision_filters,
        _bake_usd,
        _convert_urdf_to_usd,
        _robot_joint_drive_cfg,
    )

    urdf = resolve_repo_path(spec.urdf_path)
    key = cache_key(
        urdf,
        fix_base=fix_base,
        self_collision=self_collision,
        make_instanceable=make_instanceable,
        capsules=bool(spec.replace_cylinders_with_capsules),
        filters=apply_filters,
        # Adjacency is authored INTO the USD, so a spec whose adjacency changed
        # must not reuse an artefact built from the old map.
        adjacency=sorted((k, tuple(sorted(v))) for k, v in spec.adjacent_links.items()),
        bake=sorted(bake_props.items()),
    )
    entry = cache_root() / key
    stamp = entry / "READY"
    baked = entry / "baked.usd.txt"

    if stamp.exists() and baked.exists():
        path = baked.read_text(encoding="utf-8").strip()
        if Path(path).exists():
            if verbose:
                print(f"[usd_cache] hit  {spec.name} -> {key}")
            return path, True

    t0 = time.perf_counter()
    entry.mkdir(parents=True, exist_ok=True)
    raw = _convert_urdf_to_usd(
        spec.urdf_path, entry / "usd", fix_base=fix_base,
        self_collision=self_collision,
        joint_drive=_robot_joint_drive_cfg(),
        replace_cylinders_with_capsules=spec.replace_cylinders_with_capsules,
        make_instanceable=make_instanceable,
    )
    if apply_filters:
        _apply_self_collision_filters(raw, spec.adjacent_links)
    usd = _bake_usd(raw, entry / "baked", spec.name, props=dict(bake_props),
                    apply_physx_articulation=True)

    # Write the READY stamp LAST. A run killed mid-conversion leaves an entry
    # with no stamp, which the next run treats as a miss and rebuilds, rather
    # than loading a half-written USD.
    baked.write_text(usd, encoding="utf-8")
    stamp.write_text(f"{spec.name}\n{time.time()}\n", encoding="utf-8")
    if verbose:
        print(f"[usd_cache] miss {spec.name} -> {key} "
              f"({time.perf_counter() - t0:.2f}s)")
    return usd, False


def stats() -> dict:
    root = cache_root()
    if not root.exists():
        return {"entries": 0, "bytes": 0}
    entries = [p for p in root.iterdir() if (p / "READY").exists()]
    total = sum(f.stat().st_size for p in entries for f in p.rglob("*") if f.is_file())
    return {"entries": len(entries), "bytes": total}


def prune() -> int:
    root = cache_root()
    if not root.exists():
        return 0
    n = 0
    for p in list(root.iterdir()):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            n += 1
    return n


__all__ = ["USD_CACHE_DIR", "cache_key", "cached_convert_and_bake", "stats", "prune"]
