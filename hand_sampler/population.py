"""A pre-generated, on-disk population of hands.

Sampling collision-free hands is expensive and was being redone from scratch on
every run: the geometric gate rejects ~93% of draws, and each rejected draw
costs a URDF write, a yourdfpy load and a pairwise penetration check. Producing
ten hands took ~145 draws and minutes of wall clock, and none of it survived the
process. This module does that work once and writes the result down.

It also fixes a naming collision that the replay-from-seed scheme has. There are
two samplers with different acceptance rules:

  ``params.sample_valid``           accepts on validate() -- mount separation
                                    and reach, which are proxies
  ``check_self_collision.sample_collision_free``
                                    additionally rejects hands whose geometry
                                    actually overlaps, which is the real gate

Both name their output ``gen_<seed>_<index>``, so the same name denotes
different hands depending on which produced it, and
``synth_spec.params_for_name`` -- which replays ``sample_valid`` -- rebuilds the
wrong hand for any name that came from the collision-free path. Reading the
parameters back from a manifest removes the ambiguity: a name resolves by
lookup, and the population is fixed at the moment it was written rather than
being a function of whatever the acceptance rule happens to be today.

The manifest stores each hand's full parameter vector, so specs and URDFs can be
rebuilt without resampling, and it records the rejection statistics because that
rate is itself a reading on how much of the design space is usable.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.build_hand_population --seed 0 --count 64
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from hand_sampler import params
from hand_sampler import resolve as resolve_repo_path

POPULATION_DIR = "assets/urdf/generated/population"
MANIFEST_NAME = "manifest.json"

# Bumped when the on-disk format changes in a way older manifests cannot satisfy.
MANIFEST_VERSION = 1


def population_dir(seed: int) -> Path:
    return resolve_repo_path(POPULATION_DIR) / f"seed_{seed:04d}"


def manifest_path(seed: int) -> Path:
    return population_dir(seed) / MANIFEST_NAME


def _segment_to_json(seg: params.Segment) -> dict:
    return {"xyz": list(seg.xyz), "rpy": list(seg.rpy)}


def _segment_from_json(d: dict) -> params.Segment:
    return params.Segment(xyz=tuple(d["xyz"]), rpy=tuple(d["rpy"]))


def hand_to_json(hand: params.HandParams) -> dict:
    """Serialize a parameter vector losslessly.

    Written field by field rather than via ``dataclasses.asdict`` so that a
    field added to ``HandParams`` later fails loudly here instead of being
    silently dropped from every cached population.
    """
    fingers = []
    for f in hand.fingers:
        fingers.append({
            "name": f.name,
            "active": f.active,
            "enabled": list(f.enabled),
            "mount": _segment_to_json(f.mount),
            "cmc": _segment_to_json(f.cmc),
            "mc": _segment_to_json(f.mc),
            "mcp": _segment_to_json(f.mcp),
            "pp_length": f.pp_length,
            "mp_length": f.mp_length,
            "dp_length": f.dp_length,
            "limits": [list(pair) for pair in f.limits],
            "radius_scale": f.radius_scale,
            "mount_params": list(f.mount_params) if f.mount_params is not None else None,
        })
    return {
        "name": hand.name,
        "fingers": fingers,
        "palm_extents": list(hand.palm_extents),
        "notes": hand.notes,
    }


def hand_from_json(d: dict) -> params.HandParams:
    fingers = tuple(
        params.FingerParams(
            name=f["name"],
            active=f["active"],
            enabled=tuple(f["enabled"]),
            mount=_segment_from_json(f["mount"]),
            cmc=_segment_from_json(f["cmc"]),
            mc=_segment_from_json(f["mc"]),
            mcp=_segment_from_json(f["mcp"]),
            pp_length=f["pp_length"],
            mp_length=f["mp_length"],
            dp_length=f["dp_length"],
            limits=tuple(tuple(pair) for pair in f["limits"]),
            radius_scale=f["radius_scale"],
            mount_params=(tuple(f["mount_params"])
                          if f["mount_params"] is not None else None),
        )
        for f in d["fingers"]
    )
    return params.HandParams(
        name=d["name"],
        fingers=fingers,
        palm_extents=tuple(d["palm_extents"]),
        notes=d.get("notes", ""),
    )


def _roundtrip_ok(hand: params.HandParams) -> bool:
    """Does the serialized form rebuild the same parameter vector?

    Checked at write time on every hand. A cache that silently loses a field is
    worse than no cache: everything downstream would keep working while
    simulating a different robot than the one that was validated.
    """
    return hand_from_json(hand_to_json(hand)) == hand


def _hits(hand: params.HandParams, gate: str) -> bool:
    """Does this hand self-collide, under the requested gate?"""
    if gate == "analytic":
        from hand_sampler.self_collision import analytic_hand_hits

        return bool(analytic_hand_hits(hand))
    import tempfile

    from hand_sampler.self_collision import generated_hand_hits

    with tempfile.TemporaryDirectory(prefix="genmech_align_") as t:
        return bool(generated_hand_hits(hand, Path(t), hand.name))


def build_population(
    seed: int,
    count: int,
    *,
    max_tries_per_hand: int = 4000,
    force: bool = False,
    align_flexion: bool = True,
    gate: str = "analytic",
    shard: int | None = None,
    num_shards: int = 1,
    **sample_kwargs,
) -> list[params.HandParams]:
    """Sample ``count`` collision-free hands and cache them with their URDFs.

    Returns the population. Re-run with ``force=True`` to regenerate; otherwise
    an existing manifest of at least ``count`` hands is reused.
    """
    import tempfile

    from hand_sampler.urdf import write_urdf
    from hand_sampler.self_collision import sample_collision_free

    out_dir = population_dir(seed)
    # A shard builds its own slice and writes a partial manifest; the cache check reads the MERGED...
    sharded = shard is not None
    if sharded:
        if not 0 <= shard < num_shards:
            raise ValueError(f"shard {shard} outside 0..{num_shards - 1}")
        lo = (count * shard) // num_shards
        hi = (count * (shard + 1)) // num_shards
        count, name_offset = hi - lo, lo
        print(f"[population] shard {shard}/{num_shards}: hands "
              f"{lo}..{hi - 1} ({count})")
    else:
        name_offset = 0
    if not force and not sharded:
        try:
            cached = load_population(seed)
            if len(cached) >= count:
                print(f"[population] reusing {len(cached)} cached hands in {out_dir}")
                return cached[:count]
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    if gate == "analytic":
        # Closed-form capsule/box distances.
        from hand_sampler.self_collision import sample_collision_free_fast

        hands = sample_collision_free_fast(
            seed, count, max_tries_per_hand=max_tries_per_hand,
            stream_key=None if not sharded else f"shard{shard:03d}",
            name_offset=name_offset, **sample_kwargs)
    elif gate == "mesh":
        with tempfile.TemporaryDirectory(prefix="genmech_population_") as tmp:
            hands = sample_collision_free(seed, count, Path(tmp),
                                          max_tries_per_hand=max_tries_per_hand,
                                          **sample_kwargs)
    else:
        raise ValueError(f"gate must be 'analytic' or 'mesh', got {gate!r}")

    from hand_sampler.urdf import urdf_path_for
    from hand_sampler.self_collision import generated_hand_hits

    entries, realigned, reverted = [], 0, 0
    for hand in hands:
        # Write the CANONICAL path -- the one synth_spec resolves and therefore the one that actually...
        urdf = urdf_path_for(hand)
        urdf.parent.mkdir(parents=True, exist_ok=True)
        write_urdf(hand, urdf)

        if align_flexion:
            aligned = align_flexion_downward(hand, urdf_path=urdf)
            if aligned is not hand:
                write_urdf(aligned, urdf)
                # Re-rolling moves the fingers, so the collision-free guarantee established during sampling no...
                if _hits(aligned, gate):
                    write_urdf(hand, urdf)
                    reverted += 1
                else:
                    hand, realigned = aligned, realigned + 1

        if not _roundtrip_ok(hand):
            raise RuntimeError(
                f"{hand.name}: parameters do not survive a JSON round-trip; "
                "hand_to_json is missing a field that HandParams carries"
            )
        entries.append({"name": hand.name, "urdf": str(urdf),
                        "params": hand_to_json(hand)})

    if align_flexion:
        print(f"[population] flexion aligned toward the workspace on "
              f"{realigned}/{len(hands)} hands; {reverted} reverted because the "
              f"new roll self-collided")

    manifest = {
        "version": MANIFEST_VERSION,
        "seed": seed,
        "count": len(entries),
        # The acceptance rule this population was built under.
        "gate": "check_self_collision.sample_collision_free",
        "align_flexion": align_flexion,
        "max_tries_per_hand": max_tries_per_hand,
        "sample_kwargs": {k: list(v) if isinstance(v, tuple) else v
                          for k, v in sample_kwargs.items()},
        "hands": entries,
    }
    manifest["shard"] = shard
    manifest["num_shards"] = num_shards
    out = (manifest_path(seed) if not sharded
           else out_dir / f"shard_{shard:03d}.json")
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[population] wrote {len(entries)} hands + URDFs -> {out}")
    return hands


def merge_population_shards(seed: int, num_shards: int) -> int:
    """Concatenate shard_*.json into the manifest load_population() reads.

    Shards are joined IN SHARD ORDER, which is the order their name indices were
    assigned, so hand i of the merged manifest is gen_<seed>_<i:05d>. env i holds
    design i % k, and _build_robot_design_tensor checks that assignment against
    per-design joint limits, so a mis-ordered merge would surface there -- but
    getting it right here is cheaper than debugging it there.
    """
    out_dir = population_dir(seed)
    entries, missing = [], []
    for i in range(num_shards):
        f = out_dir / f"shard_{i:03d}.json"
        if not f.exists():
            missing.append(i)
            continue
        entries.extend(json.loads(f.read_text(encoding="utf-8"))["hands"])
    if missing:
        raise FileNotFoundError(
            f"population seed {seed}: shards {missing} did not produce output; "
            f"refusing to merge a population with holes in it")
    names = [e["name"] for e in entries]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate hand names across shards")
    expect = [f"gen_{seed:04d}_{i:05d}" for i in range(len(entries))]
    if names != expect:
        bad = next(i for i, (a, b) in enumerate(zip(names, expect)) if a != b)
        raise RuntimeError(
            f"merged hands are not contiguously named from 0: index {bad} is "
            f"{names[bad]}, expected {expect[bad]}")
    first = json.loads((out_dir / "shard_000.json").read_text(encoding="utf-8"))
    manifest = {**first, "count": len(entries), "hands": entries,
                "shard": None, "num_shards": num_shards}
    manifest_path(seed).write_text(json.dumps(manifest, indent=2),
                                   encoding="utf-8")
    print(f"[population] merged {num_shards} shards -> {len(entries)} hands")
    return len(entries)


def load_population(seed: int) -> list[params.HandParams]:
    """Read a cached population. Raises FileNotFoundError if absent."""

    path = manifest_path(seed)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION:
        raise KeyError(
            f"{path}: manifest version {manifest.get('version')} != "
            f"{MANIFEST_VERSION}; rebuild with build_hand_population --force")
    return [hand_from_json(e["params"]) for e in manifest["hands"]]


def load_population_at(path) -> list[params.HandParams]:
    """Read any manifest.json, wherever it lives.

    Populations that were SAMPLED are identified by a seed, and load_population
    maps that seed to seed_<NNNN>/manifest.json. A population produced by
    MUTATION has no seed -- it is a function of (parent population, operator,
    alpha, rng) -- so filing it under a seed directory would misstate its
    provenance. Those are addressed by path instead, and this is the reader.
    """
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION:
        raise KeyError(f"{path}: manifest version {manifest.get('version')} != "
                       f"{MANIFEST_VERSION}")
    return [hand_from_json(e["params"]) for e in manifest["hands"]]


def load_population_any(seed=None, path=None) -> list[params.HandParams]:
    """Path wins over seed. One resolver, so the load sites cannot disagree."""
    if path:
        from hand_sampler import resolve as _r
        return load_population_at(_r(str(path)))
    if seed is None:
        raise ValueError("neither robot_population_path nor robot_population_seed is set")
    return load_population(int(seed))


def population_specs(seed: int, count: int | None = None) -> list:
    """Specs for a cached population, building URDFs only if they are missing."""

    from hand_sampler.synth_spec import synth_spec

    from hand_sampler.urdf import urdf_path_for, write_urdf

    hands = load_population(seed)
    if count is not None:
        hands = hands[:count]
    # Rewrite unconditionally.
    for h in hands:
        write_urdf(h, urdf_path_for(h))
    return [synth_spec(h) for h in hands]


__all__ = [
    "POPULATION_DIR",
    "MANIFEST_VERSION",
    "build_population",
    "load_population",
    "population_specs",
    "population_dir",
    "manifest_path",
    "hand_to_json",
    "hand_from_json",
]


# --- flexion alignment: point every finger at the workspace ---

import math
from dataclasses import replace

import numpy as np

from hand_sampler import params

# The palm's +x points down at the table when the arm is in its home pose (the -x face is the...
CURL_TARGET = np.array([1.0, 0.0, 0.0])

# Flexion joints.
_FLEX_SUFFIXES = ("FE", "PIP", "DIP")


def _finger_axis(mount: params.Segment) -> np.ndarray:
    """The finger's own axis (mount-frame +x) in palm coordinates."""

    r, p, y = mount.rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    # First column of Rz(y) Ry(p) Rx(r) -- the image of local +x.
    return np.array([cy * cp, sy * cp, -sp])


def curl_directions(hand: params.HandParams, urdf_path=None) -> dict[int, np.ndarray]:
    """Unit curl direction per active finger index, in the palm frame."""

    import yourdfpy

    from hand_sampler.urdf import urdf_path_for, write_urdf
    from hand_sampler import resolve as resolve_repo_path

    if urdf_path is None:
        urdf_path = urdf_path_for(hand)
        if not urdf_path.exists():
            write_urdf(hand, urdf_path)
    urdf = yourdfpy.URDF.load(str(resolve_repo_path(urdf_path)), load_meshes=False,
                              load_collision_meshes=False, build_scene_graph=True)

    out: dict[int, np.ndarray] = {}
    palm_inv = None
    for i, finger in enumerate(hand.fingers):
        if not finger.active:
            continue
        tip = f"gen_f{i}_DP"
        if tip not in urdf.link_map:
            continue
        flex = [j for j in urdf.joint_map
                if j.startswith(f"gen_f{i}_")
                and j.rsplit("_", 1)[-1] in _FLEX_SUFFIXES]
        if not flex:
            continue

        def tip_in_palm(angle: float) -> np.ndarray:
            urdf.update_cfg({j: angle for j in flex})
            palm = urdf.get_transform("gen_palm")
            return (np.linalg.inv(palm) @ urdf.get_transform(tip))[:3, 3]

        # A finite flexion rather than a derivative: the whole point is where the fingertip ends up...
        delta = tip_in_palm(0.6) - tip_in_palm(0.0)
        norm = float(np.linalg.norm(delta))
        if norm < 1e-9:
            continue
        out[i] = delta / norm
    return out


def optimal_roll_offset(curl: np.ndarray, axis: np.ndarray,
                        target: np.ndarray = CURL_TARGET) -> float:
    """Extra roll that best aligns ``curl`` with ``target``, about ``axis``."""

    k = axis / (np.linalg.norm(axis) + 1e-12)
    A = float(np.dot(target, curl) - np.dot(target, k) * np.dot(k, curl))
    B = float(np.dot(target, np.cross(k, curl)))
    return math.atan2(B, A)


def align_flexion_downward(hand: params.HandParams, urdf_path=None) -> params.HandParams:
    """Re-roll every face-mounted finger so it flexes toward the workspace.

    Fingers whose mount did not come from :func:`params.mount_on_face` are left
    alone: their transforms are measured values (SHARPA's, for the reference
    hand) and re-rolling them would silently stop reproducing the robot they
    were taken from.
    """

    curls = curl_directions(hand, urdf_path=urdf_path)
    if not curls:
        return hand

    fingers = list(hand.fingers)
    changed = False
    for i, finger in enumerate(fingers):
        if i not in curls or not finger.mount_params:
            continue
        face, u_frac, v_frac, roll, tilt, tilt_azimuth = finger.mount_params
        d_roll = optimal_roll_offset(curls[i], _finger_axis(finger.mount))
        new_roll = roll + d_roll
        fingers[i] = replace(
            finger,
            mount=params.mount_on_face(face, u_frac, v_frac, new_roll, tilt,
                                  tilt_azimuth, hand.palm_extents),
            mount_params=(face, u_frac, v_frac, new_roll, tilt, tilt_azimuth),
        )
        changed = True

    if not changed:
        return hand
    return replace(hand, fingers=tuple(fingers))


def report(hand: params.HandParams) -> list[tuple[int, str, float]]:
    """(finger index, face, curl component toward the table) per active finger."""

    rows = []
    for i, c in curl_directions(hand).items():
        f = hand.fingers[i]
        face = f.mount_params[0] if f.mount_params else "-"
        rows.append((i, face, float(np.dot(c, CURL_TARGET))))
    return rows


__all__ = ["CURL_TARGET", "align_flexion_downward", "curl_directions",
           "optimal_roll_offset", "report"]
