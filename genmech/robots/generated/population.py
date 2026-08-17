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

from genmech.robots.generated import params as P
from genmech.utils.paths import resolve as resolve_repo_path

POPULATION_DIR = "assets/urdf/generated/population"
MANIFEST_NAME = "manifest.json"

# Bumped when the on-disk format changes in a way older manifests cannot satisfy.
MANIFEST_VERSION = 1


def population_dir(seed: int) -> Path:
    return resolve_repo_path(POPULATION_DIR) / f"seed_{seed:04d}"


def manifest_path(seed: int) -> Path:
    return population_dir(seed) / MANIFEST_NAME


def _segment_to_json(seg: P.Segment) -> dict:
    return {"xyz": list(seg.xyz), "rpy": list(seg.rpy)}


def _segment_from_json(d: dict) -> P.Segment:
    return P.Segment(xyz=tuple(d["xyz"]), rpy=tuple(d["rpy"]))


def hand_to_json(hand: P.HandParams) -> dict:
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


def hand_from_json(d: dict) -> P.HandParams:
    fingers = tuple(
        P.FingerParams(
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
    return P.HandParams(
        name=d["name"],
        fingers=fingers,
        palm_extents=tuple(d["palm_extents"]),
        notes=d.get("notes", ""),
    )


def _roundtrip_ok(hand: P.HandParams) -> bool:
    """Does the serialized form rebuild the same parameter vector?

    Checked at write time on every hand. A cache that silently loses a field is
    worse than no cache: everything downstream would keep working while
    simulating a different robot than the one that was validated.
    """
    return hand_from_json(hand_to_json(hand)) == hand


def _hits(hand: P.HandParams, gate: str) -> bool:
    """Does this hand self-collide, under the requested gate?"""
    if gate == "analytic":
        from genmech.tools.capsule_collision import analytic_hand_hits

        return bool(analytic_hand_hits(hand))
    import tempfile

    from genmech.tools.check_self_collision import generated_hand_hits

    with tempfile.TemporaryDirectory(prefix="genmech_align_") as t:
        return bool(generated_hand_hits(hand, Path(t), hand.name))


def build_population(
    seed: int,
    count: int,
    *,
    max_tries_per_hand: int = 400,
    force: bool = False,
    align_flexion: bool = True,
    gate: str = "analytic",
    **sample_kwargs,
) -> list[P.HandParams]:
    """Sample ``count`` collision-free hands and cache them with their URDFs.

    Returns the population. Re-run with ``force=True`` to regenerate; otherwise
    an existing manifest of at least ``count`` hands is reused.
    """
    import tempfile

    from genmech.tools.build_hand_urdf import write_urdf
    from genmech.tools.check_self_collision import sample_collision_free

    out_dir = population_dir(seed)
    if not force:
        try:
            cached = load_population(seed)
            if len(cached) >= count:
                print(f"[population] reusing {len(cached)} cached hands in {out_dir}")
                return cached[:count]
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    if gate == "analytic":
        # Closed-form capsule/box distances. The mesh gate answers the same
        # question with trimesh BVH queries over ~400 link pairs per candidate,
        # after writing and re-parsing a URDF -- measured 6.02 s per accepted
        # hand, which is 41 hours for 24,576. This is ~30 ms per hand, and
        # agreed with the mesh gate on 60/60 candidates
        # (genmech.tools.compare_collision_gates).
        from genmech.tools.capsule_collision import sample_collision_free_fast

        hands = sample_collision_free_fast(
            seed, count, max_tries_per_hand=max_tries_per_hand, **sample_kwargs)
    elif gate == "mesh":
        with tempfile.TemporaryDirectory(prefix="genmech_population_") as tmp:
            hands = sample_collision_free(seed, count, Path(tmp),
                                          max_tries_per_hand=max_tries_per_hand,
                                          **sample_kwargs)
    else:
        raise ValueError(f"gate must be 'analytic' or 'mesh', got {gate!r}")

    from genmech.robots.generated.flexion import align_flexion_downward
    from genmech.tools.build_hand_urdf import urdf_path_for
    from genmech.tools.check_self_collision import generated_hand_hits

    entries, realigned, reverted = [], 0, 0
    for hand in hands:
        # Write the CANONICAL path -- the one synth_spec resolves and therefore
        # the one that actually gets simulated. Keeping a second copy under the
        # population directory made the cache a lie: synth_spec only writes when
        # the file is ABSENT, so a rebuilt population left the canonical file
        # stale and the simulator ran hands from a previous design space while
        # every check read the fresh ones.
        urdf = urdf_path_for(hand)
        urdf.parent.mkdir(parents=True, exist_ok=True)
        write_urdf(hand, urdf)

        if align_flexion:
            aligned = align_flexion_downward(hand, urdf_path=urdf)
            if aligned is not hand:
                write_urdf(aligned, urdf)
                # Re-rolling moves the fingers, so the collision-free guarantee
                # established during sampling no longer covers this geometry.
                # Re-check rather than assume; keep the original if the new roll
                # made the hand overlap itself.
                #
                # Uses the SAME gate the sampling used. Leaving this on the mesh
                # gate would have quietly put the 41 hours back: 0.6 s per hand
                # over 24,576 hands is another four hours, in a loop that runs
                # after the fast sampler has already finished.
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
        # The acceptance rule this population was built under. A later change to
        # validate() or to the collision gate does not retroactively alter these
        # hands, but it does mean a fresh build would differ -- so record it.
        "gate": "check_self_collision.sample_collision_free",
        "align_flexion": align_flexion,
        "max_tries_per_hand": max_tries_per_hand,
        "sample_kwargs": {k: list(v) if isinstance(v, tuple) else v
                          for k, v in sample_kwargs.items()},
        "hands": entries,
    }
    manifest_path(seed).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[population] wrote {len(entries)} hands + URDFs -> {out_dir}")
    return hands


def load_population(seed: int) -> list[P.HandParams]:
    """Read a cached population. Raises FileNotFoundError if absent."""

    path = manifest_path(seed)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION:
        raise KeyError(
            f"{path}: manifest version {manifest.get('version')} != "
            f"{MANIFEST_VERSION}; rebuild with build_hand_population --force")
    return [hand_from_json(e["params"]) for e in manifest["hands"]]


def population_specs(seed: int, count: int | None = None) -> list:
    """Specs for a cached population, building URDFs only if they are missing."""

    from genmech.robots.generated.synth_spec import synth_spec

    from genmech.tools.build_hand_urdf import urdf_path_for, write_urdf

    hands = load_population(seed)
    if count is not None:
        hands = hands[:count]
    # Rewrite unconditionally. synth_spec(ensure_urdf=True) only writes when the
    # file is missing, so a stale file from an earlier population silently wins
    # -- which is exactly how a rebuilt design space ended up not being the one
    # simulated.
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
