"""Where designs come from: grammar variants (sampling), derivation and
population files, and the mutation operators the viewer offers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hand_sampler.grammar.derive import (
    EVOLUTION_OPERATORS,
    Derivation,
    VariationImpossible,
    derivation_from_dict,
    derive,
    sample_derivation,
    vary,
)
from hand_sampler.grammar.distributions import Distribution
from hand_sampler.grammar.kinematics import KinematicModel, ModelError
from hand_sampler.grammar.variants import G0_SCREEN_VARIANTS, NAMED_DISTRIBUTIONS

from .envload import load_env_modules

# Variants named in project-notes/grammar/RECENTER-2026-10-05.md first, then
# every other entry of NAMED_DISTRIBUTIONS.
_PREFERRED = ("G_V3S", "G_V2S", "G_V1S", "G_V1", "G_FULL", "G_SERIAL")


def _g0_alias() -> Dict[str, str]:
    alias = {}
    for short, dist in G0_SCREEN_VARIANTS.items():
        for name, d in NAMED_DISTRIBUTIONS.items():
            if d is dist:
                alias[name] = short
    return alias


def variant_names() -> List[str]:
    names = [n for n in _PREFERRED if n in NAMED_DISTRIBUTIONS]
    names += [n for n in NAMED_DISTRIBUTIONS if n not in names]
    return names


def variant_label(name: str) -> str:
    alias = _g0_alias().get(name)
    extra = []
    if name == "G_FULL":
        extra.append("DEFAULT")
    if alias:
        extra.append(alias)
    return f"{name} ({', '.join(extra)})" if extra else name


def variant_from_label(label: str) -> str:
    return label.split(" ")[0]


def distribution(name: str) -> Distribution:
    d = NAMED_DISTRIBUTIONS[name]
    if isinstance(d, tuple):  # (Distribution, operators) pairs
        d = d[0]
    return d


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


@dataclass
class SampleResult:
    derivation: Optional[Derivation]
    model: Optional[KinematicModel]
    seed: int
    tries: int
    viable: bool
    report: Optional[dict]
    seeds_tried: Tuple[int, int] = (0, 0)
    cancelled: bool = False


def sample(variant: str, seed: int) -> Tuple[Derivation, KinematicModel]:
    d = sample_derivation(int(seed), distribution(variant))
    return d, derive(d)


def is_viable_report(report: dict) -> bool:
    return bool(report.get("admitted")) and (report.get("fingertips_reachable") or 0) >= 2


def sample_until_viable(variant: str, start_seed: int, *, step: int = 1, max_tries: int = 500,
                        cancel: Optional[Callable[[], bool]] = None,
                        progress: Optional[Callable[[int, int], None]] = None) -> SampleResult:
    """Draw seeds start_seed, start_seed+step, ... until `viability_report`
    admits a design with >= 2 fingertips reaching the spawn point (the G0
    screen's "viable"). Negative seeds are skipped (search stops at 0 when
    stepping down)."""
    ge = load_env_modules().grammar_envelope
    dist = distribution(variant)
    seed = int(start_seed)
    tries = 0
    first = seed
    while tries < max_tries:
        if seed < 0:
            break
        if cancel is not None and cancel():
            return SampleResult(None, None, seed, tries, False, None, (first, seed), cancelled=True)
        tries += 1
        d = sample_derivation(seed, dist)
        try:
            m = derive(d)
        except ModelError:
            seed += step
            continue
        rep = ge.viability_report(m)
        if progress is not None:
            progress(tries, seed)
        if is_viable_report(rep):
            return SampleResult(d, m, seed, tries, True, rep, (first, seed))
        seed += step
    return SampleResult(None, None, seed - step, tries, False, None, (first, seed - step))


# --------------------------------------------------------------------------
# Mutation
# --------------------------------------------------------------------------

STRUCTURAL_OPERATORS = tuple(op for op in EVOLUTION_OPERATORS if not op.startswith("step_"))
SMALL_STEP_OPS = tuple(op for op in EVOLUTION_OPERATORS if op.startswith("step_"))


@dataclass
class MutationResult:
    child: Optional[Derivation]
    operator: Optional[str]
    tries: int
    message: str
    report: Optional[dict] = None


def mutate(derivation: Derivation, dist: Distribution, rng: np.random.Generator,
           operator: Optional[str] = None) -> MutationResult:
    """One `vary` call: a named operator, or one drawn from EVOLUTION_OPERATORS
    (the evolution driver's pool)."""
    if operator is None:
        op = EVOLUTION_OPERATORS[int(rng.integers(0, len(EVOLUTION_OPERATORS)))]
    else:
        op = operator
    try:
        child = vary(derivation, rng, dist, operator=op)
    except VariationImpossible as exc:
        return MutationResult(None, op, 1, f"`{op}` cannot act on this design (VariationImpossible: {exc})")
    return MutationResult(child, op, 1, f"applied `{op}`")


def mutate_until_viable(derivation: Derivation, dist: Distribution, rng: np.random.Generator, *,
                        max_tries: int = 64, cancel: Optional[Callable[[], bool]] = None) -> MutationResult:
    """Random EVOLUTION_OPERATORS mutations of the SAME parent until a child is
    viable (admitted and >= 2 fingertips reach)."""
    ge = load_env_modules().grammar_envelope
    impossible = 0
    for k in range(1, max_tries + 1):
        if cancel is not None and cancel():
            return MutationResult(None, None, k - 1, "cancelled")
        res = mutate(derivation, dist, rng)
        if res.child is None:
            impossible += 1
            continue
        rep = ge.viability_report(derive(res.child))
        if is_viable_report(rep):
            return MutationResult(res.child, res.operator, k, f"viable child after {k} tries (`{res.operator}`)", rep)
    return MutationResult(None, None, max_tries,
                          f"no viable child in {max_tries} tries ({impossible} operator draws were inapplicable)")


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


@dataclass
class FileEntry:
    label: str
    derivation: Derivation
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedFile:
    path: str
    kind: str
    entries: List[FileEntry]
    variant: Optional[str] = None
    errors: List[str] = field(default_factory=list)


def _entry(label: str, dd: dict, meta: dict, errors: List[str]) -> Optional[FileEntry]:
    try:
        return FileEntry(label=label, derivation=derivation_from_dict(dd), meta=meta)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
        return None


def parse_document(doc: Any, path: str = "<memory>") -> LoadedFile:
    """Recognises: a single derivation (`derivation_to_dict`), a population
    file (`designs[].derivation`), an evolution driver state (`archive.cells
    [].derivation_dict`), a bare archive (`evolution_archive/0.1`), a
    `{"derivation": ...}` wrapper, or a list of any of these entries."""
    errors: List[str] = []
    entries: List[FileEntry] = []
    variant = None
    kind = "unknown"
    if isinstance(doc, dict) and doc.get("schema", "").startswith("hand_grammar_derivation"):
        kind = "derivation"
        e = _entry(Path(path).stem, doc, {}, errors)
        entries = [e] if e else []
    elif isinstance(doc, dict) and isinstance(doc.get("designs"), list):
        kind = f"population ({doc.get('schema', '?')})"
        for i, d in enumerate(doc["designs"]):
            e = _entry(f"{i}: {d.get('source', '?')}", d["derivation"],
                       {k: d.get(k) for k in ("source", "sha256", "exempt_overlap")}, errors)
            if e:
                entries.append(e)
    elif isinstance(doc, dict) and isinstance(doc.get("archive"), dict):
        kind = f"driver state ({doc.get('schema', '?')}), generation {doc.get('generation_completed')}"
        variant = (doc.get("config") or {}).get("variant")
        entries = _archive_entries(doc["archive"], errors)
    elif isinstance(doc, dict) and doc.get("schema", "").startswith("evolution_archive"):
        kind = f"archive ({doc['schema']})"
        entries = _archive_entries(doc, errors)
    elif isinstance(doc, dict) and isinstance(doc.get("derivation"), dict):
        kind = "wrapped derivation"
        e = _entry(Path(path).stem, doc["derivation"], {}, errors)
        entries = [e] if e else []
    elif isinstance(doc, list):
        kind = "list"
        for i, d in enumerate(doc):
            sub = parse_document(d, f"{path}[{i}]")
            for e in sub.entries:
                e.label = f"{i}: {e.label}"
                entries.append(e)
            errors.extend(sub.errors)
    else:
        errors.append("unrecognised JSON layout (expected a derivation, population, driver state or archive)")
    return LoadedFile(path=path, kind=kind, entries=entries, variant=variant, errors=errors)


def _archive_entries(archive: dict, errors: List[str]) -> List[FileEntry]:
    ar = load_env_modules().archive
    out = []
    for row in archive.get("cells", []):
        cell = tuple(row.get("cell", (0, 0)))
        label = f"{ar.cell_label(cell)}: {row.get('design_id', '?')} fit {row.get('fitness', float('nan')):.3g}"
        meta = {k: row.get(k) for k in ("design_id", "founder_id", "parent_id", "generation_born", "fitness",
                                        "episodes", "digit_count", "joint_count", "source")}
        e = _entry(label, row["derivation_dict"], meta, errors)
        if e:
            out.append(e)
    return out


def load_file(path: str) -> LoadedFile:
    p = Path(path).expanduser()
    doc = json.loads(p.read_text())
    return parse_document(doc, str(p))
