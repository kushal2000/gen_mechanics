"""The viewer's side of the generation limits (hand_sampler/grammar/limits.py):
the panel fields, presets, and which mutation operators can act on a hand
under the current limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hand_sampler.grammar.derive import Derivation, VariationImpossible, vary
from hand_sampler.grammar.limits import PRESETS, SIMULATOR, UNLIMITED, GenerationLimits

ANY = "any"
CUSTOM = "Custom"
PRESET_NAMES: Tuple[str, ...] = tuple(PRESETS) + (CUSTOM,)


@dataclass(frozen=True)
class IntField:
    key: str            # GenerationLimits field
    label: str          # one short panel line
    hint: str
    options: Tuple[str, ...]


INT_FIELDS: Tuple[IntField, ...] = (
    IntField("max_digits", "max fingers", "Most fingers on the hand (branch fingers not counted).",
             (ANY,) + tuple(str(i) for i in range(1, 9))),
    IntField("max_joints_per_digit", "max joints per finger", "Most joints in one finger, its branch fingers "
             "included.", (ANY,) + tuple(str(i) for i in range(1, 9))),
    IntField("max_palm_bodies", "max palm parts", "Most palm parts besides the main palm.",
             (ANY,) + tuple(str(i) for i in range(0, 5))),
    IntField("max_jointed_palm_bodies", "max palm joints", "Most palm parts with their own joint.",
             (ANY,) + tuple(str(i) for i in range(0, 5))),
    IntField("max_digits_per_jointed_palm_body", "fingers per palm joint",
             "Most fingers moved by one palm joint (on its palm part, or on rigid palm parts below it).",
             (ANY,) + tuple(str(i) for i in range(0, 4))),
    IntField("max_finger_chains", "max finger slots",
             "Fingers on the rigid palm plus palm joints: the simulator has 5 finger slots, and every palm joint "
             "takes one.", (ANY,) + tuple(str(i) for i in range(1, 9))),
)

BOOL_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("allow_branches", "allow branching fingers", "A finger may grow a branch finger off one of its bones."),
    ("allow_stacked_palm_joints", "allow stacked palm joints", "A jointed palm part may sit on another jointed one."),
    ("require_digit_on_palm_body", "no empty palm parts",
     "Every palm part carries a finger: a new palm part comes with a one-joint finger, and removing a palm "
     "part's last finger removes the palm part."),
)

# Joint types other than Coupled, as one dropdown (plain label -> kinds);
# Coupled is its own checkbox.
JOINT_TYPE_OPTIONS: Dict[str, str] = {
    "hinge": "R", "hinge + continuous": "R C", "hinge + sliding": "R P", "hinge + continuous + sliding": "R C P",
    "continuous": "C", "sliding": "P", "continuous + sliding": "C P",
}
JOINT_TYPES_HINT = "Joint types that may be generated: hinge (limited range), continuous (spins freely), sliding."
COUPLED_HINT = "Coupled joints: a hinge that follows an earlier hinge of the same finger (needs hinges)."


def int_to_option(v: Optional[int]) -> str:
    return ANY if v is None else str(v)


def option_to_int(s: str) -> Optional[int]:
    return None if s == ANY else int(s)


def joint_types_option(limits: GenerationLimits) -> str:
    kinds = " ".join(k for k in ("R", "C", "P") if k in limits.allowed_modules)
    return next(label for label, k in JOINT_TYPE_OPTIONS.items() if k == kinds)


def build_limits(ints: Dict[str, str], bools: Dict[str, bool], joint_types: str, coupled: bool) -> GenerationLimits:
    kinds = tuple(JOINT_TYPE_OPTIONS[joint_types].split()) + (("Coupled",) if coupled else ())
    return GenerationLimits(
        allowed_modules=kinds,
        **{k: option_to_int(v) for k, v in ints.items()},
        **bools,
    )


def preset_of(limits: GenerationLimits) -> str:
    for name, lim in PRESETS.items():
        if lim == limits:
            return name
    return CUSTOM


# --------------------------------------------------------------------------
# Operator applicability
# --------------------------------------------------------------------------

OK, NOT_ALLOWED, NOTHING = "ok", "not allowed by limits", "nothing to act on"


def _applies(derivation: Derivation, dist, op: str, limits: Optional[GenerationLimits]) -> bool:
    rng = np.random.default_rng(12345)
    try:
        vary(derivation, rng, dist, operator=op, limits=limits)
        return True
    except VariationImpossible:
        return False
    except Exception:  # noqa: BLE001 - an off-grid projected value the operator cannot step
        return False


def operator_status(derivation: Derivation, dist, ops: Sequence[str],
                    limits: Optional[GenerationLimits]) -> Dict[str, str]:
    """Per operator: OK, NOT_ALLOWED (it could act on this hand, but not
    within `limits`), or NOTHING (nothing on this hand it can act on, even
    without limits)."""
    out: Dict[str, str] = {}
    for op in ops:
        if _applies(derivation, dist, op, limits):
            out[op] = OK
        elif limits is not None and not limits.is_unlimited and _applies(derivation, dist, op, None):
            out[op] = NOT_ALLOWED
        else:
            out[op] = NOTHING
    return out


__all__ = [
    "ANY", "BOOL_FIELDS", "COUPLED_HINT", "CUSTOM", "INT_FIELDS", "JOINT_TYPES_HINT", "JOINT_TYPE_OPTIONS",
    "NOTHING", "NOT_ALLOWED", "OK", "PRESET_NAMES", "SIMULATOR", "UNLIMITED", "build_limits", "int_to_option",
    "joint_types_option", "operator_status", "option_to_int", "preset_of",
]
