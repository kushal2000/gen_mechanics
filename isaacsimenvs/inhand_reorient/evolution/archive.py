"""MAP-Elites archive over ``(digit_count, joint_count_bin)`` for grammar
grammar hands (plan-rl-grammar-tuning.md's "Revision, 2026-09-27", pilot
E-R2': "MAP-Elites with descriptors (digit count, joint count) and one
shared controller whose weights carry across generations").

Pure python + numpy + ``hand_sampler`` only (mirrors ``grammar_envelope.py``/
``population_file.py``'s own discipline) -- no ``isaacsimenvs``/``isaaclab``/
``torch`` import anywhere in this module, so it is importable and
unit-testable under plain ``python3`` / pytest, with no Kit boot. Callers
(``driver.py``) compute ``digit_count``/``joint_count`` off a
``grammar_envelope.EnvelopeDesign`` themselves and pass them in as plain
ints on a ``Candidate`` -- this module never touches a ``KinematicModel`` or
an ``EnvelopeDesign``.

Descriptor: digit_count in {1..6} (6 bins, clamped) x joint_count in
{1-5, 6-10, 11-15, 16-20, 21-25, 26-36} (6 bins, clamped) -- 36 cells, the
envelope's own bounds (<=6 fingers, <=36 joints: 6 finger slots of a palm
joint and 5 finger joints, see ``grammar_envelope.py``'s module docstring;
``joint_count`` counts finger joints and real palm joints).

Insertion policy, per generation (``update_generation``): every candidate
submitted this generation (each current elite RE-EVALUATED under the new
shared controller, plus every offspring/immigrant) is grouped by its own
cell; the cell's new occupant is whichever candidate in that cell's group
has the highest fitness THIS generation -- this is a plain per-generation
batch argmax over the candidates actually submitted, not a running max
against a stale previous-generation value, so a design's recorded fitness
always reflects its most recent evaluation (an elite that got worse under
the new controller is recorded as worse, unless a better offspring/
immigrant took the cell instead). ``founder_id``/``generation_born``/
``fitness_history`` follow the OCCUPANT: unchanged (history appended) when
the same design_id keeps the cell, reset (fresh history, generation_born =
this call's ``generation``) when a different design_id takes it over.

A corollary the driver must respect: since this is a batch argmax over
*submitted* candidates only, an existing elite that is NOT resubmitted in a
given generation's batch offers no protection at all -- any candidate
landing in its (now candidate-less-for-the-incumbent) cell unconditionally
becomes the new occupant, regardless of fitness. The pilot design keeps
this safe by construction ("all current elites, re-evaluated") -- every
elite is always part of every generation's population and therefore always
resubmitted -- but this module itself does not enforce that; it is a pure
function of whatever candidates it is given.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Descriptor
# --------------------------------------------------------------------------

DIGIT_BINS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
JOINT_BIN_EDGES: Tuple[Tuple[int, int], ...] = (
    (1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 36),
)
N_DIGIT_BINS = len(DIGIT_BINS)
N_JOINT_BINS = len(JOINT_BIN_EDGES)
N_CELLS = N_DIGIT_BINS * N_JOINT_BINS
assert N_CELLS == 36, "6 digit bins x 6 joint-count bins"

Cell = Tuple[int, int]  # (digit_bin, joint_bin), both 0-based


def digit_bin(digit_count: int) -> int:
    """0-based bin index for ``digit_count`` (nominally 1..6). Clamped to
    ``[1, 6]`` first -- guards a pathological caller (a test, or a probe
    hand with 0 digits), never a real admitted grammar design (the
    envelope's own ``admit`` bounds digit_count to that range)."""
    d = max(1, min(int(digit_count), N_DIGIT_BINS))
    return d - 1


def joint_count_bin(joint_count: int) -> int:
    """0-based bin index into ``JOINT_BIN_EDGES`` for ``joint_count``
    (nominally 1..36). Clamped the same way: <= the first edge's high end
    for anything <= 0, the last bin for anything above 36."""
    jc = int(joint_count)
    for i, (_lo, hi) in enumerate(JOINT_BIN_EDGES):
        if jc <= hi:
            return i
    return N_JOINT_BINS - 1


def descriptor(digit_count: int, joint_count: int) -> Cell:
    """``(digit_bin, joint_bin)`` -- the archive's cell key for a design
    with the given ``digit_count``/``joint_count``."""
    return digit_bin(digit_count), joint_count_bin(joint_count)


def cell_label(cell: Cell) -> str:
    """Human-readable label for a cell key, e.g. ``"d3_j11-15"`` -- used in
    reports/CSVs, never as a dict key itself (JSON object keys must be
    strings, so ``to_dict`` uses this for the archive's own serialization
    too)."""
    d, j = cell
    lo, hi = JOINT_BIN_EDGES[j]
    return f"d{DIGIT_BINS[d]}_j{lo}-{hi}"


def all_cells() -> Tuple[Cell, ...]:
    return tuple((d, j) for d in range(N_DIGIT_BINS) for j in range(N_JOINT_BINS))


# --------------------------------------------------------------------------
# Candidate / Elite
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One design's measured fitness this generation, submitted to
    ``Archive.update_generation``. ``design_id`` is a stable identity for
    this exact design (the driver mints one per founder/offspring/immigrant,
    e.g. ``"gen0-0007"``/``"gen5-off-0013"``) -- NOT the population-file
    ``source`` string, which is only unique within one generation's
    population file. ``hand_dict``/``sha256`` are
    ``hand_sampler.grammar.hand.hand_to_dict`` / the population
    file's own entry-level sha256, so a stored elite round-trips through
    ``hand_from_dict`` with no extra state."""

    design_id: str
    hand_dict: dict
    sha256: str
    founder_id: str
    parent_id: Optional[str]
    generation_born: int
    """Advisory only: the archive's OWN ``generation_born`` bookkeeping (see
    ``Archive.update_generation``) is derived from the ``generation``
    argument at the moment a design first WINS a cell, not read off this
    field -- a candidate never has to know in advance whether it is about
    to be a fresh occupant or a re-evaluated incumbent. Kept on ``Candidate``
    for the caller's own logging/debugging."""
    digit_count: int
    joint_count: int
    fitness: float
    episodes: int
    low_confidence: bool = False
    source: str = ""


@dataclass
class Elite:
    """The archive's own record for one occupied cell."""

    design_id: str
    hand_dict: dict
    sha256: str
    founder_id: str
    parent_id: Optional[str]
    generation_born: int
    digit_count: int
    joint_count: int
    fitness: float
    episodes: int
    low_confidence: bool
    fitness_history: List[dict] = field(default_factory=list)
    source: str = ""

    @property
    def cell(self) -> Cell:
        return descriptor(self.digit_count, self.joint_count)


def _elite_from_candidate(c: Candidate, generation: int, generation_born: int, history: List[dict]) -> "Elite":
    history = list(history)
    history.append({
        "generation": int(generation), "fitness": float(c.fitness),
        "episodes": int(c.episodes), "low_confidence": bool(c.low_confidence),
    })
    return Elite(
        design_id=c.design_id, hand_dict=c.hand_dict, sha256=c.sha256,
        founder_id=c.founder_id, parent_id=c.parent_id, generation_born=int(generation_born),
        digit_count=int(c.digit_count), joint_count=int(c.joint_count), fitness=float(c.fitness),
        episodes=int(c.episodes), low_confidence=bool(c.low_confidence), fitness_history=history,
        source=c.source,
    )


# --------------------------------------------------------------------------
# Archive
# --------------------------------------------------------------------------

ARCHIVE_SCHEMA = "evolution_archive/0.1"


class Archive:
    def __init__(self) -> None:
        self.cells: Dict[Cell, Elite] = {}

    # ---- population/query -------------------------------------------------

    def __len__(self) -> int:
        return len(self.cells)

    def elites(self) -> List[Elite]:
        """Every occupied cell's elite, in a fixed (cell-sorted) order."""
        return [self.cells[c] for c in sorted(self.cells)]

    def get(self, cell: Cell) -> Optional[Elite]:
        return self.cells.get(cell)

    def sample_parent(self, rng: np.random.Generator) -> Elite:
        """Uniform choice among the archive's current elites (plan: "Pick a
        parent uniformly from the archive elites"). Raises ``ValueError``
        on an empty archive -- callers must seed generation 0 before any
        offspring are drawn."""
        elites = self.elites()
        if not elites:
            raise ValueError("cannot sample a parent from an empty archive")
        idx = int(rng.integers(0, len(elites)))
        return elites[idx]

    # ---- insertion ----------------------------------------------------

    def update_generation(self, candidates: Sequence[Candidate], generation: int) -> List[Cell]:
        """Apply one generation's worth of candidates (see module
        docstring for the per-cell batch-argmax policy). Returns the list
        of cells whose OCCUPANT (design_id) changed this call -- an empty
        list means every touched cell kept its previous design (fitness
        numbers may still have been refreshed)."""
        by_cell: Dict[Cell, List[Candidate]] = {}
        for c in candidates:
            by_cell.setdefault(descriptor(c.digit_count, c.joint_count), []).append(c)

        changed: List[Cell] = []
        for cell, cands in by_cell.items():
            best = cands[0]
            for c in cands[1:]:
                if c.fitness > best.fitness:
                    best = c
            current = self.cells.get(cell)
            is_new_design = current is None or current.design_id != best.design_id
            # `generation` (this call's own argument), not `best.generation_born`
            # -- see `Candidate.generation_born`'s docstring: a design's
            # recorded birth generation is always the generation in which it
            # first WON its cell, decided here, never read off the candidate.
            generation_born = generation if is_new_design else current.generation_born
            history = [] if is_new_design else current.fitness_history
            self.cells[cell] = _elite_from_candidate(best, generation, generation_born, history)
            if is_new_design:
                changed.append(cell)
        return changed

    # ---- reporting ------------------------------------------------------

    def coverage(self) -> int:
        return len(self.cells)

    def qd_score(self) -> float:
        return float(sum(e.fitness for e in self.cells.values()))

    def fitness_values(self) -> List[float]:
        return [e.fitness for e in self.cells.values()]

    def best_fitness(self) -> Optional[float]:
        vals = self.fitness_values()
        return max(vals) if vals else None

    def mean_fitness(self) -> Optional[float]:
        vals = self.fitness_values()
        return float(np.mean(vals)) if vals else None

    def median_fitness(self) -> Optional[float]:
        vals = self.fitness_values()
        return float(np.median(vals)) if vals else None

    def founder_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self.cells.values():
            counts[e.founder_id] = counts.get(e.founder_id, 0) + 1
        return counts

    def n_distinct_founders(self) -> int:
        return len(self.founder_counts())

    def max_founder_share(self) -> Optional[float]:
        n = len(self.cells)
        if n == 0:
            return None
        counts = self.founder_counts()
        return float(max(counts.values())) / float(n)

    def mean_joint_count(self) -> Optional[float]:
        if not self.cells:
            return None
        return float(np.mean([e.joint_count for e in self.cells.values()]))

    def max_joint_count(self) -> Optional[int]:
        if not self.cells:
            return None
        return int(max(e.joint_count for e in self.cells.values()))

    def summary(self) -> dict:
        """Everything ``driver.py``'s per-generation log needs off the
        archive alone (excludes probes/timings/fps/checkpoint, which the
        driver has no reason to route through this class)."""
        return {
            "coverage": self.coverage(),
            "n_cells_total": N_CELLS,
            "qd_score": self.qd_score(),
            "best_fitness": self.best_fitness(),
            "mean_fitness": self.mean_fitness(),
            "median_fitness": self.median_fitness(),
            "n_distinct_founders": self.n_distinct_founders(),
            "max_founder_share": self.max_founder_share(),
            "mean_joint_count": self.mean_joint_count(),
            "max_joint_count": self.max_joint_count(),
            "cells": {
                cell_label(cell): {
                    "design_id": e.design_id, "founder_id": e.founder_id, "parent_id": e.parent_id,
                    "generation_born": e.generation_born, "digit_count": e.digit_count,
                    "joint_count": e.joint_count, "fitness": e.fitness, "episodes": e.episodes,
                    "low_confidence": e.low_confidence, "sha256": e.sha256, "source": e.source,
                }
                for cell, e in sorted(self.cells.items())
            },
        }

    # ---- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema": ARCHIVE_SCHEMA,
            "cells": [
                {
                    "cell": list(cell), "design_id": e.design_id, "hand_dict": e.hand_dict,
                    "sha256": e.sha256, "founder_id": e.founder_id, "parent_id": e.parent_id,
                    "generation_born": e.generation_born, "digit_count": e.digit_count,
                    "joint_count": e.joint_count, "fitness": e.fitness, "episodes": e.episodes,
                    "low_confidence": e.low_confidence, "fitness_history": e.fitness_history,
                    "source": e.source,
                }
                for cell, e in sorted(self.cells.items())
            ],
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "Archive":
        if doc.get("schema") != ARCHIVE_SCHEMA:
            raise ValueError(f"unsupported archive schema {doc.get('schema')!r}, expected {ARCHIVE_SCHEMA!r}")
        arc = cls()
        for row in doc.get("cells", []):
            cell = (int(row["cell"][0]), int(row["cell"][1]))
            arc.cells[cell] = Elite(
                design_id=row["design_id"], hand_dict=row["hand_dict"], sha256=row["sha256"],
                founder_id=row["founder_id"], parent_id=row.get("parent_id"),
                generation_born=int(row["generation_born"]), digit_count=int(row["digit_count"]),
                joint_count=int(row["joint_count"]), fitness=float(row["fitness"]), episodes=int(row["episodes"]),
                low_confidence=bool(row.get("low_confidence", False)),
                fitness_history=list(row.get("fitness_history", [])), source=row.get("source", ""),
            )
        return arc

    def save(self, path) -> None:
        """Atomic write (temp file + rename), same discipline as
        ``design_scoring.py``/``population_file.py``: a reader never sees a
        half-written archive."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), sort_keys=True, indent=1))
        tmp.replace(out_path)

    @classmethod
    def load(cls, path) -> "Archive":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def copy(self) -> "Archive":
        return Archive.from_dict(copy.deepcopy(self.to_dict()))
