"""Population files: grammar-derivation provenance + sha256 tamper-checked
round trip, plus the two entry sources (`sampled_entries`, `projected_entry`)
`make_grammar_population.py` uses to build one.

Numpy + `hand_sampler` only (no `isaaclab`/`pxr`), like `grammar_envelope.py`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from hand_sampler.grammar.adapters.projection import ProjectionFailure, project_to_derivation
from hand_sampler.grammar.adapters.urdf import load_urdf
from hand_sampler.grammar.derive import (
    Derivation,
    derivation_from_dict,
    derivation_to_dict,
    derive,
    validate_derivation,
)
from hand_sampler.grammar.distributions import DEFAULT_DISTRIBUTION, Distribution
from hand_sampler.grammar.rules import GRAMMAR_VERSION
from hand_sampler.grammar.variants import G_SERIAL

from . import grammar_envelope as ge

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = REPO_ROOT / "hand_sampler" / "grammar_bench"
MANIFEST_PATH = BENCH_DIR / "manifest.json"

POPULATION_SCHEMA = "grammar_population/0.1"
ENVELOPE_ID = "grammar_envelope/1"

# Revolute-only: only the "R" module kind ever gets sampled (see
# distributions.Distribution.module_probabilities' docstring -- weights need
# not sum to 1, renormalized at sample time).
REVOLUTE_ONLY_PROBABILITIES = (("R", 1.0), ("C", 0.0), ("P", 0.0), ("Coupled", 0.0))


# --------------------------------------------------------------------------
# Canonical JSON + sha256
# --------------------------------------------------------------------------


def _canonical_json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry_sha256(derivation_dict: dict) -> str:
    return _sha256_hex(_canonical_json_bytes(derivation_dict))


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------
# Entries
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PopulationEntry:
    source: str
    derivation_dict: dict
    sha256: str


def make_entry(source: str, derivation: Derivation) -> PopulationEntry:
    d = derivation_to_dict(derivation)
    return PopulationEntry(source=source, derivation_dict=d, sha256=_entry_sha256(d))


def _reason_key(reason: str) -> str:
    """Coarse bucket for a rejection-reason string, for reporting counts
    without one bucket per random detail (body names, counts)."""
    return " ".join(reason.split(" ")[:4])


def _variant_distribution(variant: str) -> Distribution:
    """`"G_SERIAL"` / `"DEFAULT"`, revolute-only, `digit_count_range=(1,5)`
    -- the design note's E-R1 recommendation. `branch_probability=0` and
    `palm_body_count_range=(0,2)` are applied to DEFAULT (whose own
    defaults are 0.12 and (0,3)); G_SERIAL already samples with
    `branch_probability=0.0` and `palm_body_count_range=(0,0)` (its whole
    point -- no additional palm bodies at all), so widening the latter to
    (0,2) for G_SERIAL would silently change what "G_SERIAL" means rather
    than merely constrain it -- left as G_SERIAL's own value."""
    from dataclasses import replace

    if variant == "G_SERIAL":
        return replace(G_SERIAL, module_probabilities=REVOLUTE_ONLY_PROBABILITIES, digit_count_range=(1, 5))
    if variant == "DEFAULT":
        return replace(
            DEFAULT_DISTRIBUTION,
            module_probabilities=REVOLUTE_ONLY_PROBABILITIES,
            digit_count_range=(1, 5),
            branch_probability=0.0,
            palm_body_count_range=(0, 2),
        )
    raise ValueError(f"unknown variant {variant!r}; expected 'G_SERIAL' or 'DEFAULT'")


def sampled_entries(variant: str, seeds: Sequence[int]) -> Tuple[List[PopulationEntry], Dict[str, int]]:
    """`(entries, rejection_counts)`: every `seed` that samples an
    `admit`-ted design becomes an entry named `f"sampled:{variant}:{seed}"`;
    every rejection is bucketed (`_reason_key`) into `rejection_counts`."""
    from hand_sampler.grammar.derive import sample_derivation

    dist = _variant_distribution(variant)
    entries: List[PopulationEntry] = []
    rejections: Dict[str, int] = {}
    for seed in seeds:
        derivation = sample_derivation(seed, dist)
        model = derive(derivation)
        result = ge.admit(model)
        if not result.ok:
            for reason in result.reasons:
                key = _reason_key(reason)
                rejections[key] = rejections.get(key, 0) + 1
            continue
        entries.append(make_entry(f"sampled:{variant}:{seed}", derivation))
    return entries, rejections


# --------------------------------------------------------------------------
# Projected commercial hands (E13 pipeline)
# --------------------------------------------------------------------------


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def _resolve_hand(hand: dict, manifest: dict) -> Tuple[Optional[Path], str, Optional[str]]:
    """Mirrors `hand_sampler/grammar/experiments/e13_representation.py`'s
    own `_resolve_hand` exactly (same reasoning: this package never imports
    `grammar_bench`, only reads its `manifest.json` and mirrors its
    resolution logic)."""
    if hand.get("split") == "excluded":
        return None, "excluded", hand.get("notes") or "excluded split: never imported"

    fixture_path = hand.get("fixture_path")
    if fixture_path:
        p = BENCH_DIR / fixture_path
        if p.is_file():
            return p, "available", None
        return None, "unavailable", f"fixture_path listed in manifest but missing on disk: {fixture_path}"

    source_path = hand.get("source_path")
    if isinstance(source_path, str) and source_path.startswith("REPO:"):
        p = REPO_ROOT / source_path[len("REPO:"):]
        if p.is_file():
            return p, "available", None
        return None, "unavailable", f"REPO path listed in manifest but missing on disk: {source_path}"

    if isinstance(source_path, str) and source_path:
        source_root = manifest.get("source_root")
        if source_root:
            p = Path(source_root) / source_path
            if p.is_file():
                return p, "available", None
        return None, "unavailable", f"local-only:{hand['id']} source not present at manifest source_root on this machine"

    return None, "unavailable", "no fixture_path/source_path in manifest"


def projected_entry(hand_id: str) -> Tuple[Optional[PopulationEntry], str, Optional[str]]:
    """`(entry_or_None, status, reason)`. `status` is one of `"admitted"`,
    `"rejected"` (parsed/projected fine but not envelope-admissible, or the
    manifest sha256 didn't match) or `"unavailable"` (not on disk / not in
    the manifest / excluded split). Follows the E13 pipeline
    (`load_urdf` -> `project_to_derivation` -> `derive` -> `validate_derivation`
    -> `grammar_envelope.admit`) exactly."""
    manifest = _load_manifest()
    hand = next((h for h in manifest["hands"] if h["id"] == hand_id), None)
    if hand is None:
        return None, "unavailable", f"hand id {hand_id!r} not in manifest"

    path, availability, reason = _resolve_hand(hand, manifest)
    if availability != "available":
        return None, "unavailable", reason

    expected_sha256 = hand.get("sha256")
    if expected_sha256 is not None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            return None, "unavailable", f"sha256 mismatch: manifest declares {expected_sha256}, disk has {actual}"

    try:
        imported = load_urdf(path, hand_root=hand.get("hand_root"))
    except Exception as exc:  # noqa: BLE001
        return None, "unavailable", f"import failed: {type(exc).__name__}: {exc}"

    try:
        pr = project_to_derivation(
            imported.model, palm_joints=hand.get("palm_joints") or (), tip_frames=hand.get("tip_frames") or {}
        )
    except ProjectionFailure as exc:
        return None, "rejected", f"projection failed: {exc}"

    issues = validate_derivation(pr.derivation)
    if issues:
        return None, "rejected", f"validate_derivation issues: {issues}"

    model = derive(pr.derivation)
    result = ge.admit(model)
    if not result.ok:
        return None, "rejected", "; ".join(result.reasons)

    entry = make_entry(f"projected:{hand_id}", pr.derivation)
    return entry, "admitted", None


# --------------------------------------------------------------------------
# write_population / load_population
# --------------------------------------------------------------------------


def write_population(
    path,
    entries: Sequence[PopulationEntry],
    generator: str = "make_grammar_population.py",
    envelope: str = ENVELOPE_ID,
) -> dict:
    designs = [{"source": e.source, "derivation": e.derivation_dict, "sha256": e.sha256} for e in entries]
    population_sha256 = _sha256_hex(_canonical_json_bytes([e.sha256 for e in entries]))
    doc = {
        "schema": POPULATION_SCHEMA,
        "grammar_version": GRAMMAR_VERSION,
        "envelope": envelope,
        "git_sha": _git_sha(),
        "generator": generator,
        "population_sha256": population_sha256,
        "designs": designs,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, sort_keys=True, indent=2))
    return doc


def load_population(path) -> List["ge.EnvelopeDesign"]:
    """Load, verify every sha256 (entry + population), re-derive every
    design from its stored derivation, and re-`admit` it -- any failure is
    fatal (raises `ValueError`), per the design note's own contract."""
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != POPULATION_SCHEMA:
        raise ValueError(f"unsupported population schema {doc.get('schema')!r}, expected {POPULATION_SCHEMA!r}")

    designs_raw = doc.get("designs", [])
    expected_population_sha = _sha256_hex(_canonical_json_bytes([d["sha256"] for d in designs_raw]))
    if expected_population_sha != doc.get("population_sha256"):
        raise ValueError(
            f"population_sha256 mismatch (expected {expected_population_sha}, "
            f"file declares {doc.get('population_sha256')}): population file is tampered or corrupt"
        )

    out: List[ge.EnvelopeDesign] = []
    for d in designs_raw:
        actual_sha = _entry_sha256(d["derivation"])
        if actual_sha != d["sha256"]:
            raise ValueError(
                f"design {d.get('source')!r} sha256 mismatch (expected {d['sha256']}, "
                f"derivation hashes to {actual_sha}): entry is tampered or corrupt"
            )
        derivation = derivation_from_dict(d["derivation"])
        model = derive(derivation)
        result = ge.admit(model)
        if not result.ok:
            raise ValueError(f"design {d.get('source')!r} fails re-admission: {'; '.join(result.reasons)}")
        out.append(ge.canonicalize(model, source=d["source"]))
    return out
