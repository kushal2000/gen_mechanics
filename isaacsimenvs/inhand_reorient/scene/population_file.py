"""Population files: grammar hands with provenance and a sha256-checked round
trip, plus the two entry sources (`sampled_entries`, `commercial_entry`)
`make_grammar_population.py` uses.

Numpy + `hand_sampler` only (no `isaaclab`/`pxr`), like `grammar_envelope.py`.

Each entry stores the hand (`hand_sampler.grammar.hand.hand_to_dict`), its
sha256, a digest of the simulated tables (`_derived_digest`) and, for a
commercial hand, the overlap exemption and its collision-filtered pairs.
Files built for another grammar or layout fail loudly with a rebuild
message.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hand_sampler.grammar import build_conformed
from hand_sampler.grammar import operators as gops
from hand_sampler.grammar import viability as gvb
from hand_sampler.grammar.hand import EVOLUTION_RULES, GRAMMAR_VERSION, Hand, Rules, hand_from_dict, hand_to_dict

from . import grammar_envelope as ge

REPO_ROOT = Path(__file__).resolve().parents[3]

POPULATION_SCHEMA = "grammar_population/1.0"
ENVELOPE_ID = "grammar_envelope/3"
"""The articulation layout a file was built for: 36 slots (6 finger slots of
a palm-joint carrier and 5 finger joints), rounded-box links along +x, hull
palm plates, coupled joints as mimic joints. `load_population` refuses any
other."""

SAMPLED_PREFIX = "sampled:"
COMMERCIAL_PREFIX = "commercial:"


def _canonical_json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry_sha256(hand_dict: dict) -> str:
    return _sha256_hex(_canonical_json_bytes(hand_dict))


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True,
                             check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass(frozen=True)
class PopulationEntry:
    source: str
    hand_dict: dict
    sha256: str
    derived_sha256: str = ""
    exempt_overlap: bool = False
    filtered_pairs: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)

    @property
    def hand(self) -> Hand:
        return hand_from_dict(self.hand_dict)


def _derived_digest(design: "ge.EnvelopeDesign") -> str:
    """sha256 over the simulated tables (`canonicalize` and `palm_up`), so a
    code change that alters them for the same hand is caught on load."""
    pu = ge.palm_up(design)
    payload = {
        "slot_valid": design.slot_valid.astype(bool).tolist(),
        "slot_real": design.slot_real.astype(bool).tolist(),
        "slot_prismatic": design.slot_prismatic.astype(bool).tolist(),
        "slot_origin": np.round(design.slot_origin.astype(float), 12).tolist(),
        "slot_axis": np.round(design.slot_axis.astype(float), 12).tolist(),
        "slot_limits": np.round(design.slot_limits.astype(float), 12).tolist(),
        "slot_length": np.round(design.slot_length.astype(float), 12).tolist(),
        "slot_tie": design.slot_tie.astype(int).tolist(),
        "slot_gear": np.round(design.slot_gear.astype(float), 12).tolist(),
        "default_q": np.round(pu.default_q.astype(float), 12).tolist(),
        "spawn_offset": np.round(pu.spawn_offset.astype(float), 12).tolist(),
        "base_rot_wxyz": np.round(pu.base_rot_wxyz.astype(float), 12).tolist(),
        "filtered_pairs": [list(p) for p in design.filtered_pairs],
    }
    return _sha256_hex(_canonical_json_bytes(payload))


def _design(hand: Hand, source: str, exempt_overlap: bool) -> "ge.EnvelopeDesign":
    design = ge.canonicalize(hand, source=source)
    if exempt_overlap:
        design = ge.mark_filtered_pairs(design, extra_qs=[ge.palm_up(design).default_q])
    return design


def make_entry(source: str, hand: Hand, *, exempt_overlap: bool = False) -> PopulationEntry:
    d = hand_to_dict(hand)
    design = _design(hand, source, exempt_overlap)
    return PopulationEntry(source=source, hand_dict=d, sha256=_entry_sha256(d), derived_sha256=_derived_digest(design),
                           exempt_overlap=exempt_overlap,
                           filtered_pairs=design.filtered_pairs if exempt_overlap else ())


def _reason_key(reason: str) -> str:
    return reason.split(":")[0]


def sampled_entries(seeds: Sequence[int], rules: Rules = EVOLUTION_RULES, stage: str = "coarse",
                    require_viable: bool = True, limit: Optional[int] = None,
                    ) -> Tuple[List[PopulationEntry], Dict[str, int]]:
    """`(entries, rejection_counts)`: every seed whose random hand passes C1
    (and C2 with `require_viable`) becomes `sampled:{seed}`; stops after
    `limit` entries."""
    entries: List[PopulationEntry] = []
    rejections: Dict[str, int] = {}
    for seed in seeds:
        if limit is not None and len(entries) >= limit:
            break
        hand = gops.random_hand(np.random.default_rng(seed), rules, stage)
        reasons = list(ge.admit(hand).reasons)
        if not reasons and require_viable and not gvb.c2_workspace_overlap(hand).ok:
            reasons.append("C2: fingertip workspaces do not meet above the palm")
        if reasons:
            for r in reasons:
                rejections[_reason_key(r)] = rejections.get(_reason_key(r), 0) + 1
            continue
        entries.append(make_entry(f"{SAMPLED_PREFIX}{seed}", hand))
    return entries, rejections


def commercial_entry(hand_id: str) -> Tuple[Optional[PopulationEntry], str, Optional[str]]:
    """`(entry_or_None, status, reason)` for a conformed commercial hand
    (`hand_sampler/grammar_bench/conformed_hands.json`). Commercial hands are
    exempt from the overlap check: their overlaps come from the shared link
    cross-section, so the overlapping pairs are collision-filtered instead."""
    recs = build_conformed.load()
    rec = recs.get(hand_id)
    if rec is None:
        return None, "unavailable", f"{hand_id!r} is not in conformed_hands.json (not on the machine that built it?)"
    result = ge.admit(rec["hand"], check_overlap=False)
    if not result.ok:
        return None, "rejected", "; ".join(result.reasons)
    return make_entry(f"{COMMERCIAL_PREFIX}{hand_id}", rec["hand"], exempt_overlap=True), "admitted", None


def write_population(path, entries: Sequence[PopulationEntry], generator: str = "make_grammar_population.py",
                     envelope: str = ENVELOPE_ID) -> dict:
    designs = [{"source": e.source, "hand": e.hand_dict, "sha256": e.sha256, "derived_sha256": e.derived_sha256,
                "exempt_overlap": e.exempt_overlap, "filtered_pairs": [list(p) for p in e.filtered_pairs]}
               for e in entries]
    doc = {
        "schema": POPULATION_SCHEMA, "grammar_version": GRAMMAR_VERSION, "envelope": envelope, "git_sha": _git_sha(),
        "generator": generator, "population_sha256": _sha256_hex(_canonical_json_bytes([e.sha256 for e in entries])),
        "designs": designs,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, sort_keys=True, indent=2))
    return doc


def load_population(path) -> List["ge.EnvelopeDesign"]:
    """Load and verify (schema, grammar, layout, every sha256, the derived
    digest) and re-admit every design; any failure raises `ValueError`."""
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != POPULATION_SCHEMA:
        raise ValueError(f"{path}: population schema {doc.get('schema')!r}, this checkout reads "
                         f"{POPULATION_SCHEMA!r}; rebuild it with make_grammar_population.py")
    if doc.get("grammar_version") != GRAMMAR_VERSION:
        raise ValueError(f"{path}: grammar {doc.get('grammar_version')!r} != {GRAMMAR_VERSION!r}; rebuild it")
    if doc.get("envelope") != ENVELOPE_ID:
        raise ValueError(f"{path}: built for the layout {doc.get('envelope')!r}, this checkout uses {ENVELOPE_ID!r} "
                         f"({ge.N_SLOTS} joint slots); rebuild it with make_grammar_population.py")
    raw = doc.get("designs", [])
    expected = _sha256_hex(_canonical_json_bytes([d["sha256"] for d in raw]))
    if expected != doc.get("population_sha256"):
        raise ValueError(f"population_sha256 mismatch ({expected} != {doc.get('population_sha256')}): tampered or corrupt")
    out: List[ge.EnvelopeDesign] = []
    for d in raw:
        if _entry_sha256(d["hand"]) != d["sha256"]:
            raise ValueError(f"design {d.get('source')!r}: sha256 mismatch, entry tampered or corrupt")
        hand = hand_from_dict(d["hand"])
        exempt = bool(d.get("exempt_overlap", False))
        result = ge.admit(hand, check_overlap=not exempt)
        if not result.ok:
            raise ValueError(f"design {d.get('source')!r} fails re-admission: {'; '.join(result.reasons)}")
        design = _design(hand, d["source"], exempt)
        if _derived_digest(design) != d.get("derived_sha256"):
            raise ValueError(f"design {d.get('source')!r}: derived-table sha256 mismatch; canonicalize()/palm_up() "
                             f"output for this hand changed, rebuild the file")
        design.sha256 = d["sha256"]
        out.append(design)
    return out
