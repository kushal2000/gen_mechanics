"""Put a population on disk, so a design can be looked at instead of re-derived.

A population is reproducible from ``(seed, count)`` and was therefore never
written down -- which makes an individual design awkward to reach: you rebuild
the whole sampler to see design 8113, and a run records its git commit but not
the hands it actually trained. This is the file that closes that gap.

The sampler stays the source of truth. Nothing here feeds ``population_from_name``;
a file is an artifact OF a population, not an input to one. ``verify`` is what
catches a file that has drifted from the sampler that wrote it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from hand_sampler import design_space

FORMAT_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "assets" / "populations"


# --- the tree, as plain data -------------------------------------------------
# Written field by field rather than through dataclasses.asdict: the optional
# fields are None for every generated design, and writing them out would triple
# the file for nothing. Anything absent reads back as its dataclass default, so
# a hand-edited file can leave them out too.

def _joint_to_dict(joint: design_space.Joint) -> dict:
    out: dict = {"theta": joint.theta, "phi": joint.phi}
    if joint.offset:
        out["offset"] = joint.offset
    for name in ("axis_override", "limits", "drive"):
        value = getattr(joint, name)
        if value is not None:
            out[name] = list(value)
    return out


def _joint_from_dict(data: dict) -> design_space.Joint:
    return design_space.Joint(
        theta=float(data["theta"]),
        phi=float(data.get("phi", math.pi / 2)),
        offset=float(data.get("offset", 0.0)),
        axis_override=_tuple(data.get("axis_override")),
        limits=_tuple(data.get("limits")),
        drive=_tuple(data.get("drive")),
    )


def _tuple(value):
    return None if value is None else tuple(value)


def _segment_to_dict(segment: design_space.Segment) -> dict:
    out: dict = {"joint": _joint_to_dict(segment.joint), "length": segment.length}
    if segment.cross_section is not None:
        out["cross_section"] = list(segment.cross_section)
    if segment.meshes is not None:
        out["meshes"] = list(segment.meshes)
    if segment.token_box is not None:
        # (4, 3) nested; keep the nesting rather than flattening it.
        out["token_box"] = [list(row) for row in segment.token_box]
    return out


def _segment_from_dict(data: dict) -> design_space.Segment:
    token_box = data.get("token_box")
    return design_space.Segment(
        joint=_joint_from_dict(data["joint"]),
        length=float(data["length"]),
        cross_section=_tuple(data.get("cross_section")),
        meshes=_tuple(data.get("meshes")),
        token_box=None if token_box is None else tuple(tuple(row) for row in token_box),
    )


def hand_to_dict(hand: design_space.Hand) -> dict:
    return {
        "palm": {"thickness": hand.palm.thickness,
                 "width": hand.palm.width,
                 "length": hand.palm.length},
        "fingers": [
            {"mount": {"face": finger.mount.face, "u": finger.mount.u, "v": finger.mount.v},
             "segments": [_segment_to_dict(s) for s in finger.segments]}
            for finger in hand.fingers
        ],
    }


def hand_from_dict(data: dict) -> design_space.Hand:
    return design_space.Hand(
        palm=design_space.Palm(**{k: float(v) for k, v in data["palm"].items()}),
        fingers=tuple(
            design_space.Finger(
                mount=design_space.Mount(face=finger["mount"]["face"],
                                         u=float(finger["mount"]["u"]),
                                         v=float(finger["mount"]["v"])),
                segments=tuple(_segment_from_dict(s) for s in finger["segments"]),
            )
            for finger in data["fingers"]
        ),
    )


# --- files -------------------------------------------------------------------

def default_path(name: str) -> Path:
    return DEFAULT_DIR / f"{name}.json"


def save_population(hands, path, *, name: str) -> Path:
    """Write ``hands`` as one JSON file. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT_VERSION,
        "name": name,
        "count": len(hands),
        "designs": [hand_to_dict(h) for h in hands],
    }
    # indent=1 rather than 0 or 2: a design stays greppable line by line without
    # the file doubling in whitespace at 24k of them.
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_population(path) -> list[design_space.Hand]:
    """Read a population file back. Every design is re-validated on construction."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    got = int(data.get("format", 0))
    if got != FORMAT_VERSION:
        raise ValueError(
            f"{path}: format {got}, this build reads {FORMAT_VERSION}")
    hands = [hand_from_dict(d) for d in data["designs"]]
    if len(hands) != int(data["count"]):
        raise ValueError(
            f"{path}: header says {data['count']} designs, file holds {len(hands)}")
    return hands


def verify(path, hands) -> None:
    """Raise unless the file on disk is the population the sampler produces now.

    A stored population is a record, and a record that has silently drifted from
    the code is worse than no record: it would send you to look at a design the
    run never trained.
    """
    stored = load_population(path)
    if len(stored) != len(hands):
        raise ValueError(
            f"{path}: holds {len(stored)} designs, the sampler now makes {len(hands)}")
    for i, (a, b) in enumerate(zip(stored, hands)):
        if a != b:
            raise ValueError(f"{path}: design {i} differs from the sampler's")


# --- CLI ---------------------------------------------------------------------

def main() -> None:
    import argparse

    from hand_sampler.robot_spec import is_population_name, population_from_name

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="population name, e.g. gen_s0_n24576")
    ap.add_argument("--out", default=None,
                    help=f"output path (default: {DEFAULT_DIR}/<name>.json)")
    ap.add_argument("--verify", action="store_true",
                    help="check an existing file against the sampler instead of writing")
    args = ap.parse_args()

    if not is_population_name(args.name):
        raise SystemExit(f"not a population name: {args.name!r} (want gen_s<seed>_n<count>)")
    out = Path(args.out) if args.out else default_path(args.name)

    print(f"[population_io] sampling {args.name} ...", flush=True)
    hands = list(population_from_name(args.name).hands)

    if args.verify:
        verify(out, hands)
        print(f"[population_io] {out} matches the sampler ({len(hands)} designs)")
        return

    save_population(hands, out, name=args.name)
    joints = sum(h.n_joints for h in hands)
    print(f"[population_io] wrote {len(hands)} designs ({joints} joints, "
          f"{out.stat().st_size / 1e6:.1f} MB) to {out}")


if __name__ == "__main__":
    main()
