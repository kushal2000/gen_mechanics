"""Envelope analysis for the current design: the simulator's viability oracle
(`grammar_envelope.viability_report`) plus the pieces of it the viewer draws
(spawn point, which fingertips reach it, which capsule pairs overlap), the
MAP-Elites descriptor, structural counts and the nearest commercial hand.

Everything numeric comes from `grammar_envelope` itself; the only thing
re-implemented here is the per-finger split of `palm_up`'s reach sweep
(`palm_up` returns only the COUNT of reaching fingertips). The re-implementation
replays the same seeded sweep, and `finger_reach` asserts that its count equals
`palm_up`'s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hand_sampler.grammar.kinematics import KinematicModel

from .envload import load_env_modules
from .model import Structure, structure

# palm_up's defaults, used by viability_report (do not change: the readout must
# match the oracle).
OBJECT_HALF_SIZE_M = 0.03
REACH_TOL_M = 0.05
N_SWEEP = 4000
SWEEP_SEED = 0
CURL_FRAC = 0.35


def ge():
    return load_env_modules().grammar_envelope


def archive_mod():
    return load_env_modules().archive


@dataclass(frozen=True)
class FingerReach:
    slot: int                 # envelope finger slot 0..5
    digit_id: Optional[str]
    tip_body: str             # last real body of that finger
    hits: int                 # sweep samples within REACH_TOL_M of the spawn point
    min_dist_m: float         # closest approach over the sweep
    tip_at_reset: np.ndarray  # root-frame fingertip at default_q

    @property
    def reaches(self) -> bool:
        return self.hits > 0


@dataclass
class Analysis:
    report: dict
    structural_ok: bool
    structural_reasons: Tuple[str, ...]
    design: object = None          # grammar_envelope.EnvelopeDesign
    pu: object = None              # grammar_envelope.PalmUpResult
    finger_reach: List[FingerReach] = field(default_factory=list)
    pairs_q0: List[Tuple[str, str, float]] = field(default_factory=list)
    pairs_reset: List[Tuple[str, str, float]] = field(default_factory=list)
    fit_without_overlap: Optional[Tuple[bool, Tuple[str, ...]]] = None
    descriptor_cell: Optional[Tuple[int, int]] = None
    descriptor_label: Optional[str] = None
    descriptor_note: str = ""
    structure: Optional[Structure] = None

    @property
    def viable(self) -> bool:
        return is_viable(self.report)

    @property
    def spawn_point(self) -> Optional[np.ndarray]:
        return None if self.pu is None else np.asarray(self.pu.spawn_offset, dtype=float)

    def reset_u(self) -> Dict[str, float]:
        """The env's reset pose (`palm_up(...).default_q`) as model joint values."""
        if self.design is None or self.pu is None:
            return {}
        return slot_q_to_model(self.design, self.pu.default_q)


def is_viable(report: Mapping) -> bool:
    """The G0 screen's definition: admitted AND >= 2 fingertips reach the spawn."""
    return bool(report.get("admitted")) and (report.get("fingertips_reachable") or 0) >= 2


def slot_q_to_model(design, slot_q: Sequence[float]) -> Dict[str, float]:
    out = {}
    for idx, name in enumerate(design.slot_joint_name):
        if name is not None and design.slot_valid[idx]:
            out[name] = float(slot_q[idx])
    return out


def model_q_to_slot(design, q: Mapping[str, float]) -> np.ndarray:
    g = ge()
    out = np.zeros(g.N_SLOTS)
    for idx, name in enumerate(design.slot_joint_name):
        if name is not None and design.slot_valid[idx]:
            out[idx] = float(q.get(name, 0.0))
    return g.tied_q(design, out)


def _slot_body(design, idx: int) -> str:
    g = ge()
    if idx == g.ROOT_NODE:
        return design.model.root
    return design.slot_body_name[idx] or f"slot{idx}"


def named_pairs(design, pairs) -> List[Tuple[str, str, float]]:
    return [(_slot_body(design, i), _slot_body(design, j), float(pen)) for i, j, pen in pairs]


def overlaps_at(design, q: Mapping[str, float]) -> List[Tuple[str, str, float]]:
    """`rest_overlap_pairs` at an arbitrary model pose (body names, metres)."""
    return named_pairs(design, ge().rest_overlap_pairs(design, q=model_q_to_slot(design, q)))


def finger_reach(design, pu) -> List[FingerReach]:
    """Per-finger split of `palm_up`'s reach sweep: the same seeded uniform
    sweep over every valid slot's limits (other slots at default_q), evaluated
    with `authored_fk_batch`. Raises if the count disagrees with `pu`."""
    g = ge()
    valid = [i for i in range(g.N_SLOTS) if design.slot_valid[i]]
    rng = np.random.default_rng(SWEEP_SEED)
    q_batch = np.tile(np.asarray(pu.default_q, dtype=float), (N_SWEEP, 1))
    for idx in valid:
        lo, hi = design.slot_limits[idx]
        q_batch[:, idx] = rng.uniform(lo, hi, size=N_SWEEP)
    tips_all = g.tip_fk(design, g.authored_fk_batch(design, q_batch))          # (N, 6, 3)
    tips_mid = g.tip_fk(design, g.authored_fk(design, np.asarray(pu.default_q, dtype=float)))
    spawn = np.asarray(pu.spawn_offset, dtype=float)
    out: List[FingerReach] = []
    for f in range(g.N_FINGERS):
        used = [g.finger_slot(f, d) for d in range(g.N_JOINTS_PER_FINGER) if design.slot_valid[g.finger_slot(f, d)]]
        if not used:
            continue
        last = max(used)
        dist = np.linalg.norm(tips_all[:, f] - spawn[None, :], axis=1)
        out.append(FingerReach(
            slot=f, digit_id=design.finger_digit_id[f], tip_body=design.slot_body_name[last] or "",
            hits=int(np.count_nonzero(dist <= REACH_TOL_M)), min_dist_m=float(dist.min()),
            tip_at_reset=tips_mid[f],
        ))
    n_reach = sum(1 for r in out if r.reaches)
    if n_reach != int(pu.reachable_fingertips):
        raise AssertionError(f"per-finger reach replay ({n_reach}) disagrees with palm_up ({pu.reachable_fingertips})")
    return out


def descriptor_for(digit_count: int, joint_count: int) -> Tuple[Tuple[int, int], str]:
    ar = archive_mod()
    cell = ar.descriptor(digit_count, joint_count)
    return cell, ar.cell_label(cell)


def analyze(model: KinematicModel, *, with_fit_without_overlap: bool = False) -> Analysis:
    """Full analysis of one model. `viability_report` is called verbatim; the
    design/palm_up objects are rebuilt with the same defaults it uses, so the
    drawn spawn point and reach marks are the ones the oracle judged."""
    g = ge()
    st = structure(model)
    report = g.viability_report(model)
    structural = g._admit_structural(model)
    a = Analysis(report=report, structural_ok=structural.ok, structural_reasons=tuple(structural.reasons),
                 structure=st)
    if with_fit_without_overlap:
        res = g.admit(model, check_overlap=False)
        a.fit_without_overlap = (bool(res.ok), tuple(res.reasons))
    if not structural.ok:
        # No envelope design: the descriptor falls back to the model's own counts.
        n_digits = len(st.top_level_digits)
        cell, label = descriptor_for(n_digits, st.n_movable)
        a.descriptor_cell, a.descriptor_label = cell, label
        a.descriptor_note = "not envelope-admissible: bins from the model's own digit/joint counts"
        return a
    design = g.canonicalize(model)
    pu = g.palm_up(design)
    a.design, a.pu = design, pu
    a.finger_reach = finger_reach(design, pu)
    a.pairs_q0 = named_pairs(design, g.rest_overlap_pairs(design))
    a.pairs_reset = named_pairs(design, g.rest_overlap_pairs(design, q=pu.default_q))
    cell, label = descriptor_for(report["digit_count"], report["joint_count"])
    a.descriptor_cell, a.descriptor_label = cell, label
    return a


def slot_table(design) -> List[str]:
    """One line per envelope finger slot: its finger and its carrier (palm
    joint) role."""
    g = ge()
    lines = []
    for f, role in enumerate(g.carrier_roles(design)):
        used = [g.finger_slot(f, d) for d in range(g.N_JOINTS_PER_FINGER) if design.slot_valid[g.finger_slot(f, d)]]
        c = g.carrier_slot(f)
        on = {g.LOCKED: "on the palm", g.LEADER: f"palm joint of {design.slot_body_name[c]}",
              g.FOLLOWER: f"tied to f{g.slot_finger(int(design.slot_tie[c]))}'s palm joint"}[role]
        if not used:
            lines.append(f"f{f}: " + ("padding" if role == g.LOCKED else f"no finger, {on}"))
            continue
        lines.append(f"f{f}: digit {design.finger_digit_id[f]} ({len(used)} joints), {on}")
    return lines


# --------------------------------------------------------------------------
# Nearest commercial hand
# --------------------------------------------------------------------------


def nearest_commercial(model: KinematicModel, references: Mapping[str, KinematicModel],
                       n_configs: int = 8, seed: int = 0) -> List[Tuple[str, float, dict]]:
    """`phenodist.phenotype_distance(current, reference)` for every reference,
    sorted by tip displacement. Joints are aligned by NAME, which is meaningful
    here only because both the grammar and the projection name digit joints
    `d{k}p{i}_j`; digit numbering is arbitrary, so this is a coarse
    similarity, not a registration."""
    from hand_sampler.grammar.phenodist import phenotype_distance

    out = []
    for hid, ref in references.items():
        try:
            d = phenotype_distance(model, ref, seed=seed, n_configs=n_configs)
        except Exception as exc:  # noqa: BLE001 - a broken reference must not break the panel
            d = {"tip_displacement_m": math.inf, "error": str(exc)}
        out.append((hid, float(d["tip_displacement_m"]), d))
    out.sort(key=lambda r: r[1])
    return out
