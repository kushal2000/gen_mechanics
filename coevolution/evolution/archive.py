"""Every hand an evolution run ever made, with its lineage and fitness history (``archive.json`` in the run dir).

A hand's status: ``active`` (in the training population), ``retired`` (was active, removed), ``candidate`` (proposed
this generation, awaiting its zero-shot screen) or ``unadmitted`` (screened, not admitted). Nothing is ever deleted:
every hand stays a possible parent (DGM's archive). Fitness entries are ``{"generation", "value", "screen", ...}``;
``screen`` marks a zero-shot score taken before the shared policy trained on the hand.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import descriptors

STATUSES = ("active", "retired", "candidate", "unadmitted")


@dataclass
class HandRecord:
    id: str
    spec: str                          # the hand's spec JSON (what the env loads), repo-relative or absolute
    design: dict                       # the grammar genotype (hand_to_dict)
    born: int                          # generation it was made in (0: a seed hand)
    status: str
    parent: Optional[str] = None
    operator: Optional[str] = None     # the grammar operator that made it from its parent ("seed", "random")
    admitted: Optional[int] = None     # generation it entered the population
    key: str = ""                      # canonical design JSON: duplicates are never re-made
    fitness: list = field(default_factory=list)
    children: list = field(default_factory=list)

    @property
    def niche(self) -> str:
        return descriptors.niche(self.design)

    def latest(self, screen: Optional[bool] = None) -> Optional[dict]:
        """The latest fitness entry (of the given kind, if ``screen`` is set)."""
        es = [e for e in self.fitness if screen is None or bool(e.get("screen")) == screen]
        return es[-1] if es else None

    def current(self) -> Optional[float]:
        """The value selection uses: the latest trained-on evaluation, else the zero-shot screen, else None."""
        e = self.latest(screen=False) or self.latest(screen=True)
        return None if e is None else float(e["value"])


class Archive:
    def __init__(self, records: Optional[dict] = None):
        self.records: dict[str, HandRecord] = dict(records or {})

    # -- persistence --------------------------------------------------------------------------------------
    @classmethod
    def load(cls, path) -> "Archive":
        d = json.loads(Path(path).read_text())
        return cls({r["id"]: HandRecord(**r) for r in d["hands"]})

    def save(self, path) -> None:
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"hands": [asdict(r) for r in self.records.values()]}, indent=1))
        os.replace(tmp, path)                      # atomic: a crash never leaves half an archive

    # -- access -------------------------------------------------------------------------------------------
    def add(self, r: HandRecord) -> None:
        if r.id in self.records:
            raise KeyError(f"hand {r.id} is already in the archive")
        if r.status not in STATUSES:
            raise ValueError(f"status {r.status!r}")
        self.records[r.id] = r
        if r.parent is not None:
            self.records[r.parent].children.append(r.id)

    def __getitem__(self, i: str) -> HandRecord:
        return self.records[i]

    def __len__(self) -> int:
        return len(self.records)

    def with_status(self, *status: str) -> list[HandRecord]:
        return [r for r in self.records.values() if r.status in status]

    def keys(self) -> set[str]:
        return {r.key for r in self.records.values() if r.key}

    def by_spec(self) -> dict[str, HandRecord]:
        return {r.spec: r for r in self.records.values()}

    def record(self, hand_id: str, generation: int, metrics: dict, screen: bool) -> None:
        self.records[hand_id].fitness.append({"generation": generation, "screen": screen, **metrics})

    def elites(self) -> dict[str, HandRecord]:
        """The fittest hand of each niche (hands with a fitness only)."""
        best: dict[str, HandRecord] = {}
        for r in self.records.values():
            f = r.current()
            if f is not None and (r.niche not in best or f > best[r.niche].current()):
                best[r.niche] = r
        return best


__all__ = ["Archive", "HandRecord", "STATUSES"]
