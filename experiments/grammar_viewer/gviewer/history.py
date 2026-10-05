"""Mutation history (a browser-style stack with back/forward) and the
parent -> child diff summary."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hand_sampler.grammar.derive import Derivation, derivation_from_dict, derivation_to_dict, derive

from .model import structure

HISTORY_SCHEMA = "grammar_viewer_history/0.1"


@dataclass
class HistoryEntry:
    derivation: Derivation
    label: str          # source description (root) or operator applied
    operator: Optional[str] = None
    note: str = ""
    payload: Any = field(default=None, repr=False, compare=False)
    """Viewer-side object for this entry (not serialised)."""


class History:
    """Entries 0..n-1; `cursor` is the entry on screen. Mutating from a
    non-final entry drops the entries after it (like a browser)."""

    def __init__(self) -> None:
        self.entries: List[HistoryEntry] = []
        self.cursor: int = -1

    def reset(self, derivation: Derivation, label: str) -> None:
        self.entries = [HistoryEntry(derivation=derivation, label=label)]
        self.cursor = 0

    @property
    def current(self) -> Optional[HistoryEntry]:
        return self.entries[self.cursor] if 0 <= self.cursor < len(self.entries) else None

    @property
    def parent(self) -> Optional[HistoryEntry]:
        return self.entries[self.cursor - 1] if self.cursor >= 1 else None

    def push(self, derivation: Derivation, operator: str, note: str = "") -> HistoryEntry:
        if self.cursor < 0:
            raise RuntimeError("history has no root; call reset() first")
        del self.entries[self.cursor + 1:]
        e = HistoryEntry(derivation=derivation, label=operator, operator=operator, note=note)
        self.entries.append(e)
        self.cursor = len(self.entries) - 1
        return e

    def can_back(self) -> bool:
        return self.cursor > 0

    def can_forward(self) -> bool:
        return self.cursor < len(self.entries) - 1

    def back(self) -> Optional[HistoryEntry]:
        if self.can_back():
            self.cursor -= 1
        return self.current

    def forward(self) -> Optional[HistoryEntry]:
        if self.can_forward():
            self.cursor += 1
        return self.current

    def jump(self, index: int) -> Optional[HistoryEntry]:
        if 0 <= index < len(self.entries):
            self.cursor = index
        return self.current

    def lineage_lines(self) -> List[str]:
        out = []
        for i, e in enumerate(self.entries):
            mark = "**>**" if i == self.cursor else "&nbsp;&nbsp;"
            what = e.label if e.operator is None else f"`{e.operator}`"
            note = f" ({e.note})" if e.note else ""
            out.append(f"{mark} {i}. {what}{note}")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": HISTORY_SCHEMA,
            "cursor": self.cursor,
            "entries": [
                {"label": e.label, "operator": e.operator, "note": e.note,
                 "derivation": derivation_to_dict(e.derivation)}
                for e in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "History":
        if d.get("schema") != HISTORY_SCHEMA:
            raise ValueError(f"unsupported history schema {d.get('schema')!r}")
        h = cls()
        h.entries = [HistoryEntry(derivation=derivation_from_dict(e["derivation"]), label=e["label"],
                                  operator=e.get("operator"), note=e.get("note", ""))
                     for e in d["entries"]]
        h.cursor = int(d["cursor"])
        return h


# --------------------------------------------------------------------------
# Diff summary
# --------------------------------------------------------------------------

_MM_KEYS = {"length", "root_length", "capsule_radius_m", "mount_offset", "bend_offset"}
_DEG_KEYS = {"direction_rpy", "mount_rpy", "bend_rpy", "limits", "offset"}


def _flatten(params: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in params.items():
        if k == "uid":
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _fmt(key: str, v: Any) -> str:
    leaf = key.split(".")[-1]
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return str(v)
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if leaf in _MM_KEYS:
            return f"{v * 1000:.1f} mm"
        if leaf in _DEG_KEYS:
            return f"{math.degrees(v):.1f} deg"
        return f"{v:.4g}"
    if isinstance(v, (tuple, list)):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            if leaf in _MM_KEYS:
                return "(" + ", ".join(f"{x * 1000:.1f}" for x in v) + ") mm"
            if leaf in _DEG_KEYS:
                return "(" + ", ".join(f"{math.degrees(x):.0f}" for x in v) + ") deg"
            return "(" + ", ".join(f"{x:.3g}" for x in v) + ")"
        return str(v)
    return str(v)


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-12
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    return a == b


def _step_key(step) -> Tuple[str, Any]:
    uid = step.params.get("uid")
    return ("uid", uid) if isinstance(uid, int) else ("path", step.path)


@dataclass
class DiffSummary:
    digits: Tuple[int, int]
    phalanges: Tuple[List[int], List[int]]
    joints: Tuple[int, int]
    palm_bodies: Tuple[int, int]
    changed: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)

    def lines(self, max_changes: int = 14) -> List[str]:
        def arrow(a, b):
            return f"{a} -> {b}" if a != b else f"{a} (same)"
        out = [
            f"digits {arrow(*self.digits)}; joints {arrow(*self.joints)}; palm bodies {arrow(*self.palm_bodies)}",
            f"phalanges per digit {arrow(self.phalanges[0], self.phalanges[1])}",
        ]
        if self.added:
            out.append("added: " + ", ".join(self.added))
        if self.removed:
            out.append("removed: " + ", ".join(self.removed))
        if self.changed:
            shown = self.changed[:max_changes]
            out.extend(f"- {c}" for c in shown)
            if len(self.changed) > max_changes:
                out.append(f"- ... {len(self.changed) - max_changes} more")
        if not (self.added or self.removed or self.changed):
            out.append("no parameter changed")
        return out


def diff(parent: Derivation, child: Derivation) -> DiffSummary:
    """Steps are matched by their stable `uid` (the grammar's I15 identity:
    renumbering operators keep it), the `hand` step by path."""
    sp, sc = structure(derive(parent)), structure(derive(child))
    pm = {_step_key(s): s for s in parent.steps}
    cm = {_step_key(s): s for s in child.steps}
    changed, added, removed = [], [], []
    for key, s in cm.items():
        if key not in pm:
            added.append(f"{s.production} {s.path}")
            continue
        old = _flatten(pm[key].params)
        new = _flatten(s.params)
        where = s.path if s.path == pm[key].path else f"{pm[key].path} -> {s.path}"
        for k in sorted(set(old) | set(new)):
            if k in ("digit_id", "p", "name", "parent", "mount") and k in old and k in new and old[k] != new[k]:
                changed.append(f"{where}: {k} {old[k]} -> {new[k]} (renumbered)")
                continue
            if k not in old:
                changed.append(f"{where}: {k} = {_fmt(k, new[k])} (new field)")
            elif k not in new:
                changed.append(f"{where}: {k} removed")
            elif not _values_equal(old[k], new[k]):
                changed.append(f"{where}: {k} {_fmt(k, old[k])} -> {_fmt(k, new[k])}")
    for key, s in pm.items():
        if key not in cm:
            removed.append(f"{s.production} {s.path}")
    return DiffSummary(
        digits=(len(sp.top_level_digits), len(sc.top_level_digits)),
        phalanges=(sp.phalanges_per_digit(), sc.phalanges_per_digit()),
        joints=(sp.n_movable, sc.n_movable),
        palm_bodies=(len(sp.palm_bodies), len(sc.palm_bodies)),
        changed=changed, added=added, removed=removed,
    )
