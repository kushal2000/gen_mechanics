"""Generation limits: the second of the hand grammar's three layers.

1. CAPABILITY is the grammar itself (``rules.py``, ``derive.py``,
   ``distributions.py``): every hand the productions can express, including
   several jointed palm bodies, stacked palm joints, coupled joints, branching
   digits and any number of digits per palm body.
2. GENERATION LIMITS (this module) are hard rules that sampling
   (``derive.sample_derivation``) and every mutation operator
   (``derive.vary``/``apply_operator``/``vary_tracked``) obey CONSTRUCTIVELY:
   they only ever choose among options that keep a hand within the limits
   (module kinds from the allowed set, a digit count below the cap, a host
   palm body with room left, a palm joint only where one is allowed, ...), so
   no rejection loop is needed and no generated or mutated hand violates them.
   An operator with no admissible application under the limits is
   inapplicable (``None`` / ``VariationImpossible``, as for an operator with
   nothing to act on).
3. VIABILITY checks are the properties that cannot be guaranteed while
   generating (capsule overlap at the zero and reset poses, spawn height,
   fingertip reach); they live with the simulator's envelope oracle
   (``isaacsimenvs/inhand_reorient/scene/grammar_envelope.py``) and the viewer.

``GenerationLimits()`` (``UNLIMITED``) limits nothing. ``SIMULATOR`` is
exactly the shape of the simulator's padded 32-slot articulation (5 finger
chains of 6 revolute slots plus 2 palm-joint slots): a derivation is within
``SIMULATOR`` iff ``grammar_envelope._admit_structural(derive(d)).ok``
(``grammar_bench/tests/test_generation_limits.py`` checks this equivalence on
sampled and mutated designs, and that every design sampled or mutated under
``SIMULATOR`` passes ``_admit_structural``).

Passing ``limits=None`` anywhere keeps the pre-limits behaviour byte for byte
(same random draws, same derivations). A limit that does not bind on a given
draw does not change that draw either, so ``UNLIMITED`` reproduces ``None``
exactly. When a limit binds, the sampler draws from the restricted options
(for example uniformly among the allowed digit counts, or among the hosts with
room), so the distribution of hands sampled under limits differs from what
rejection sampling (sample freely, discard violators) would give. That is
intended: limits shape the generator, they do not filter it.

Definitions (they match ``_admit_structural`` and ``envelope.fits_envelope``):

- a DIGIT is a top-level digit: a ``Digit`` mounted on the root or on a palm
  body (branch digits mount on a phalanx body and belong to their top-level
  digit);
- the JOINTS OF A DIGIT are the phalanges (one joint each) of a top-level digit
  plus those of every branch digit nested under it;
- a JOINTED palm body is a ``PalmBody`` with ``has_joint``; it is STACKED when
  a palm body between it and the root is jointed too;
- a digit's CARRIER is the nearest jointed palm body on the path from its mount
  to the root (none for digits on the root or on rigid palm bodies attached
  rigidly to the root); a jointed palm body CARRIES the digits whose carrier
  it is;
- FINGER CHAINS are the digits without a carrier plus the jointed palm bodies:
  in the simulator each such digit fills one of 5 finger chains, and each
  jointed palm body reserves one (its carried digit, if any, goes there);
- a BRANCHING link is a phalanx body with two or more child joints.

Stdlib only; works on any sequence of derivation steps (objects with
``production`` and ``params``), so it never imports ``derive.py``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MODULE_KINDS: Tuple[str, ...] = ("R", "C", "P", "Coupled")
_JOINT_KINDS = ("R", "C", "P")


@dataclass(frozen=True)
class GenerationLimits:
    """Hard limits on what sampling and mutation may generate. ``None`` (or
    ``True`` for the ``allow_*`` flags) means "no limit"."""

    # Phalanx module kinds that may be generated ("R" hinge, "C" continuous,
    # "P" prismatic, "Coupled" a hinge driven by an earlier hinge).
    allowed_modules: Tuple[str, ...] = MODULE_KINDS
    # Branch digits (a digit growing off a phalanx, so that link carries two
    # child joints).
    allow_branches: bool = True
    # Top-level digits.
    max_digits: Optional[int] = None
    # Joints (phalanges) in one top-level digit, branches included.
    max_joints_per_digit: Optional[int] = None
    # Palm bodies in addition to the root.
    max_palm_bodies: Optional[int] = None
    # Palm bodies with their own joint.
    max_jointed_palm_bodies: Optional[int] = None
    # A jointed palm body below another jointed palm body.
    allow_stacked_palm_joints: bool = True
    # Digits carried by one jointed palm body.
    max_digits_per_jointed_palm_body: Optional[int] = None
    # Digits without a carrier plus jointed palm bodies.
    max_finger_chains: Optional[int] = None
    # Every palm body carries at least one digit, mounted on it or on a palm
    # body below it (no empty palm parts). Sampling then never makes an empty
    # palm body, ``add_palm_body`` adds the body together with a one-joint
    # digit on it, and removing a palm body's last digit removes the body too.
    require_digit_on_palm_body: bool = False

    def __post_init__(self):
        kinds = tuple(self.allowed_modules)
        unknown = sorted(set(kinds) - set(MODULE_KINDS))
        if unknown:
            raise ValueError(f"unknown module kind(s) {unknown}; expected a subset of {MODULE_KINDS}")
        if not any(k in kinds for k in _JOINT_KINDS):
            raise ValueError("allowed_modules needs at least one of R, C, P (a digit's first phalanx cannot be "
                             "Coupled, and Coupled needs an earlier R)")
        object.__setattr__(self, "allowed_modules", tuple(k for k in MODULE_KINDS if k in kinds))
        for name, lo in (("max_digits", 1), ("max_joints_per_digit", 1), ("max_finger_chains", 1),
                         ("max_palm_bodies", 0), ("max_jointed_palm_bodies", 0),
                         ("max_digits_per_jointed_palm_body", 0)):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < lo):
                raise ValueError(f"{name} must be None or an int >= {lo}, got {v!r}")

    @property
    def is_unlimited(self) -> bool:
        return self == UNLIMITED

    def with_(self, **changes) -> "GenerationLimits":
        return replace(self, **changes)


UNLIMITED = GenerationLimits()

# The whole grammar, except that a palm part must carry a finger.
DEFAULT_LIMITS = GenerationLimits(require_digit_on_palm_body=True)

# The simulator's padded envelope (grammar_envelope.py: N_FINGERS = 5,
# N_JOINTS_PER_FINGER = 6, MAX_JOINTED_PALM_BODIES = 2, revolute-only, no
# couplings, no branching, jointed palm bodies hang off the root palm and carry
# at most one digit each, root digits + jointed palm bodies <= 5). It puts no
# bound on rigid palm bodies (they fold into the root or carrier transform).
SIMULATOR = GenerationLimits(
    allowed_modules=("R",),
    allow_branches=False,
    max_digits=5,
    max_joints_per_digit=6,
    max_palm_bodies=None,
    max_jointed_palm_bodies=2,
    allow_stacked_palm_joints=False,
    max_digits_per_jointed_palm_body=1,
    max_finger_chains=5,
    require_digit_on_palm_body=True,
)

# The simulator envelope's shape alone, without the no-empty-palm rule (the
# envelope itself accepts empty palm bodies): exactly ``_admit_structural``.
SIMULATOR_ENVELOPE = replace(SIMULATOR, require_digit_on_palm_body=False)

PRESETS: Dict[str, GenerationLimits] = {"Simulator": SIMULATOR, "Default": DEFAULT_LIMITS, "Unlimited": UNLIMITED}

LIMIT_KEYS: Tuple[str, ...] = tuple(f.name for f in fields(GenerationLimits))

# One short line per limit, for reports and the viewer, in plain words
# (finger = top-level digit, joint + bone = phalanx, palm part = palm body).
LIMIT_TEXT: Dict[str, str] = {
    "allowed_modules": "joint types",
    "allow_branches": "branching fingers",
    "max_digits": "fingers",
    "max_joints_per_digit": "joints per finger",
    "max_palm_bodies": "palm parts",
    "max_jointed_palm_bodies": "palm joints",
    "allow_stacked_palm_joints": "stacked palm joints",
    "max_digits_per_jointed_palm_body": "fingers per palm joint",
    "max_finger_chains": "finger slots (fingers on the rigid palm + palm joints)",
    "require_digit_on_palm_body": "palm parts without a finger",
}


def _inf(v: Optional[int]) -> float:
    return float("inf") if v is None else v


# --------------------------------------------------------------------------
# Structure: the limit-relevant skeleton of a derivation
# --------------------------------------------------------------------------


@dataclass
class Structure:
    """The parts of a derivation the limits read: palm bodies (parent,
    jointed), digits (mount, phalanx count), each phalanx's module kind. Built
    from derivation steps; small and cheap to copy, so operators can test a
    hypothetical edit (``with_joint``, ``with_palm``, ...) before making it."""

    palm: Dict[str, Tuple[str, bool]] = field(default_factory=dict)            # name -> (parent, jointed)
    digits: Dict[str, Tuple[str, int]] = field(default_factory=dict)           # id -> (mount, phalanx_count)
    kinds: Dict[Tuple[str, int], str] = field(default_factory=dict)            # (digit id, p) -> module kind

    @classmethod
    def from_steps(cls, steps: Iterable) -> "Structure":
        st = cls()
        for s in steps:
            p = s.params
            if s.production == "PalmBody":
                st.palm[p["name"]] = (p["parent"], bool(p["has_joint"]))
            elif s.production == "Digit":
                st.digits[p["digit_id"]] = (p["mount"], int(p["phalanx_count"]))
            elif s.production == "Phalanx":
                st.kinds[(p["digit_id"], int(p["p"]))] = p["module"]["kind"]
        return st

    def copy(self) -> "Structure":
        return Structure(dict(self.palm), dict(self.digits), dict(self.kinds))

    # ---- derived quantities ------------------------------------------------

    def is_palm_host(self, body: str) -> bool:
        return body == "root" or body in self.palm

    def top_digits(self) -> List[str]:
        return [d for d, (mount, _) in self.digits.items() if self.is_palm_host(mount)]

    def carrier(self, host: str) -> Optional[str]:
        """Nearest jointed palm body on the path from palm body ``host`` to the
        root (``host`` itself included); ``None`` for the root's own class."""
        cur, seen = host, set()
        while cur in self.palm and cur not in seen:
            seen.add(cur)
            parent, jointed = self.palm[cur]
            if jointed:
                return cur
            cur = parent
        return None

    def jointed(self) -> List[str]:
        return [n for n, (_, j) in self.palm.items() if j]

    def stacked(self) -> List[str]:
        """Jointed palm bodies with a jointed palm ancestor."""
        out = []
        for name in self.jointed():
            parent = self.palm[name][0]
            if self.carrier(parent) is not None:
                out.append(name)
        return out

    def _body_owner(self) -> Dict[str, str]:
        return {f"d{d}p{k + 1}": d for d, (_, n) in self.digits.items() for k in range(n)}

    def top_of(self) -> Dict[str, str]:
        """digit id -> id of the top-level digit it belongs to."""
        owner = self._body_owner()
        out: Dict[str, str] = {}
        for d in self.digits:
            cur, seen = d, set()
            while cur in self.digits and not self.is_palm_host(self.digits[cur][0]) and cur not in seen:
                seen.add(cur)
                nxt = owner.get(self.digits[cur][0])
                if nxt is None:
                    break
                cur = nxt
            out[d] = cur
        return out

    def joints_per_digit(self) -> Dict[str, int]:
        tops = set(self.top_digits())
        out = {t: 0 for t in tops}
        for d, t in self.top_of().items():
            if t in out:
                out[t] += self.digits[d][1]
        return out

    def branching_links(self) -> List[str]:
        mounted = Counter(m for m, _ in self.digits.values())
        out = []
        for d, (_, n) in self.digits.items():
            for k in range(n):
                body = f"d{d}p{k + 1}"
                children = (1 if k < n - 1 else 0) + mounted.get(body, 0)
                if children >= 2:
                    out.append(body)
        return out

    def carried_counts(self) -> Counter:
        return Counter(c for c in (self.carrier(self.digits[d][0]) for d in self.top_digits()) if c is not None)

    def empty_palm_bodies(self) -> List[str]:
        """Palm bodies with no top-level digit on them or on any palm body
        below them."""
        supported = set()
        for d in self.top_digits():
            cur, seen = self.digits[d][0], set()
            while cur in self.palm and cur not in seen:
                seen.add(cur)
                supported.add(cur)
                cur = self.palm[cur][0]
        return [n for n in self.palm if n not in supported]

    def palm_leaves(self) -> List[str]:
        """Palm bodies with no palm body below them, in name order."""
        parents = {par for par, _ in self.palm.values()}
        return [n for n in self.palm if n not in parents]

    def finger_chains(self) -> int:
        n_root = sum(1 for d in self.top_digits() if self.carrier(self.digits[d][0]) is None)
        return n_root + len(self.jointed())

    # ---- measurement against limits ------------------------------------------

    def excess(self, limits: GenerationLimits) -> Dict[str, int]:
        """Per limit, how far this structure is over it (0 = within)."""
        tops = self.top_digits()
        ex = {k: 0 for k in LIMIT_KEYS}
        ex["allowed_modules"] = sum(1 for k in self.kinds.values() if k not in limits.allowed_modules)
        if not limits.allow_branches:
            ex["allow_branches"] = len(self.branching_links())
        ex["max_digits"] = int(max(0, len(tops) - _inf(limits.max_digits)))
        if limits.max_joints_per_digit is not None:
            ex["max_joints_per_digit"] = sum(max(0, n - limits.max_joints_per_digit)
                                             for n in self.joints_per_digit().values())
        ex["max_palm_bodies"] = int(max(0, len(self.palm) - _inf(limits.max_palm_bodies)))
        ex["max_jointed_palm_bodies"] = int(max(0, len(self.jointed()) - _inf(limits.max_jointed_palm_bodies)))
        if not limits.allow_stacked_palm_joints:
            ex["allow_stacked_palm_joints"] = len(self.stacked())
        if limits.max_digits_per_jointed_palm_body is not None:
            ex["max_digits_per_jointed_palm_body"] = sum(
                max(0, c - limits.max_digits_per_jointed_palm_body) for c in self.carried_counts().values())
        ex["max_finger_chains"] = int(max(0, self.finger_chains() - _inf(limits.max_finger_chains)))
        if limits.require_digit_on_palm_body:
            ex["require_digit_on_palm_body"] = len(self.empty_palm_bodies())
        return ex

    def measured(self) -> Dict[str, str]:
        """Per limit, the measured value as a short string (for reports)."""
        kinds = Counter(self.kinds.values())
        jpd = self.joints_per_digit()
        carried = self.carried_counts()
        return {
            "allowed_modules": " ".join(f"{k}x{kinds[k]}" for k in MODULE_KINDS if kinds[k]) or "none",
            "allow_branches": f"{len(self.branching_links())} branching",
            "max_digits": str(len(self.top_digits())),
            "max_joints_per_digit": str(max(jpd.values(), default=0)),
            "max_palm_bodies": str(len(self.palm)),
            "max_jointed_palm_bodies": str(len(self.jointed())),
            "allow_stacked_palm_joints": f"{len(self.stacked())} stacked",
            "max_digits_per_jointed_palm_body": str(max(carried.values(), default=0)),
            "max_finger_chains": str(self.finger_chains()),
            "require_digit_on_palm_body": f"{len(self.empty_palm_bodies())} empty",
        }

    # ---- hypothetical edits (each returns a modified copy) -------------------

    def with_joint(self, name: str, on: bool) -> "Structure":
        st = self.copy()
        st.palm[name] = (st.palm[name][0], bool(on))
        return st

    def with_palm(self, name: str, parent: str, jointed: bool) -> "Structure":
        st = self.copy()
        st.palm[name] = (parent, bool(jointed))
        return st

    def without_digit(self, digit_id: str) -> "Structure":
        """The structure with a top-level digit (and its branches) removed."""
        st = self.copy()
        top = st.top_of()
        for d in [d for d, t in top.items() if t == digit_id]:
            n = st.digits.pop(d)[1]
            for k in range(n):
                st.kinds.pop((d, k), None)
        return st

    def without_empty_palm_bodies(self) -> "Structure":
        st = self.copy()
        while True:
            empty = set(st.empty_palm_bodies()) & set(st.palm_leaves())
            if not empty:
                return st
            for n in empty:
                del st.palm[n]

    def without_palm_reattach(self, name: str) -> "Structure":
        """``derive._op_remove_palm_body``: children and digits of ``name``
        re-attach to its parent (palm names are not renumbered here; names
        are only labels for the limits)."""
        st = self.copy()
        parent = st.palm.pop(name)[0]
        for n, (par, j) in list(st.palm.items()):
            if par == name:
                st.palm[n] = (parent, j)
        for d, (mount, k) in list(st.digits.items()):
            if mount == name:
                st.digits[d] = (parent, k)
        return st


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LimitReport:
    excess: Dict[str, int]
    measured: Dict[str, str]
    limits: GenerationLimits

    @property
    def ok(self) -> bool:
        return not any(self.excess.values())

    @property
    def failing(self) -> List[str]:
        return [k for k in LIMIT_KEYS if self.excess.get(k)]

    def line(self, key: str) -> str:
        """e.g. ``"digits 6 > 5"`` or ``"joint module kinds: R x3 C x1 not in R"``."""
        lim = getattr(self.limits, key)
        txt = LIMIT_TEXT[key]
        if key == "allowed_modules":
            return f"{txt} {self.measured[key]} (allowed {'/'.join(lim)})"
        if isinstance(lim, bool):
            return f"{txt}: {self.measured[key]}"
        return f"{txt} {self.measured[key]} > {lim}"

    def summary(self) -> str:
        return "yes" if self.ok else "no (" + "; ".join(self.line(k) for k in self.failing) + ")"


def check(derivation_or_steps, limits: GenerationLimits) -> LimitReport:
    """How a derivation (or its steps) measures against ``limits``."""
    steps = getattr(derivation_or_steps, "steps", derivation_or_steps)
    st = Structure.from_steps(steps)
    return LimitReport(excess=st.excess(limits), measured=st.measured(), limits=limits)


def within(derivation_or_steps, limits: Optional[GenerationLimits]) -> bool:
    return limits is None or check(derivation_or_steps, limits).ok


# --------------------------------------------------------------------------
# Constructive enforcement (used by derive.py)
# --------------------------------------------------------------------------


class LimitContext:
    """What ``derive.py`` consults while sampling or mutating under limits.

    ``base`` is the excess of the hand being mutated (all zeros when sampling
    from scratch). An edit is admissible when it raises no limit's excess above
    ``base``; for a hand within the limits that means "stays within", and a
    hand already outside them (a projected commercial hand) may still be
    mutated as long as the mutation does not make any limit worse.

    It also carries the running budgets of one sampling pass: which hosts have
    room for another top-level digit, and how many joints the digit being
    emitted (with its branches) may still use."""

    def __init__(self, limits: GenerationLimits, steps: Sequence = (), base: Optional[Dict[str, int]] = None):
        self.limits = limits
        self.struct = Structure.from_steps(steps)
        self.base = dict(base) if base is not None else self.struct.excess(limits)
        self.joint_left: Optional[int] = None
        self.joint_reserved = 0
        self._tmp = 0

    # ---- generic admissibility ------------------------------------------------

    def allows(self, struct: Structure) -> bool:
        new = struct.excess(self.limits)
        return all(new[k] <= self.base.get(k, 0) for k in new)

    def allows_steps(self, steps: Sequence) -> bool:
        return self.allows(Structure.from_steps(steps))

    # ---- modules and branches ---------------------------------------------------

    @property
    def allowed_modules(self) -> Tuple[str, ...]:
        return self.limits.allowed_modules

    def module_allowed(self, kind: str) -> bool:
        return kind in self.limits.allowed_modules

    # ---- top-level digit hosts ------------------------------------------------------

    def _new_id(self) -> str:
        self._tmp += 1
        return f"__lim{self._tmp}"

    def host_has_room(self, host: str, moving: Optional[str] = None) -> bool:
        """Whether one more top-level digit may mount on palm body ``host``
        (``moving``: a digit already in the structure that is being re-mounted,
        so it is not counted where it is now)."""
        st = self.struct
        if moving is not None and moving in st.digits:
            st = st.copy()
            del st.digits[moving]
        if self.limits.require_digit_on_palm_body:
            # No palm body may be left (or stay) without a digit: a digit
            # leaving its palm body, or a regrown digit whose old palm body is
            # empty for now, must go where it keeps every palm body covered.
            trial = st.copy()
            trial.digits["__room"] = (host, 1)
            if len(trial.empty_palm_bodies()) > self.base["require_digit_on_palm_body"]:
                return False
        lim = self.limits
        tops = st.top_digits()
        if len(tops) + 1 > _inf(lim.max_digits) and len(tops) + 1 - _inf(lim.max_digits) > self.base["max_digits"]:
            return False
        c = st.carrier(host)
        if c is None:
            chains = st.finger_chains() + 1
            if chains > _inf(lim.max_finger_chains) and chains - _inf(lim.max_finger_chains) > self.base["max_finger_chains"]:
                return False
        elif lim.max_digits_per_jointed_palm_body is not None:
            counts = st.carried_counts()
            new = counts[c] + 1
            over = sum(max(0, v - lim.max_digits_per_jointed_palm_body) for v in counts.values())
            if new > lim.max_digits_per_jointed_palm_body and over + 1 > self.base["max_digits_per_jointed_palm_body"]:
                return False
        return True

    def eligible_hosts(self, hosts: Sequence[str], moving: Optional[str] = None) -> List[str]:
        return [h for h in hosts if self.host_has_room(h, moving=moving)]

    def take_host(self, host: str) -> None:
        """Record a new top-level digit on ``host`` (sampling bookkeeping)."""
        self.struct.digits[self._new_id()] = (host, 0)

    def add_palm(self, name: str, parent: str, jointed: bool) -> None:
        self.struct.palm[name] = (parent, bool(jointed))

    def can_add_jointed_palm(self, parent: str, digits_to_place: int) -> bool:
        """While sampling a fresh hand (no digits placed yet): may the next
        palm body, on ``parent``, have a joint, so that ``digits_to_place``
        digits can still all be mounted within the limits?"""
        lim = self.limits
        n_j = len(self.struct.jointed()) + 1
        if n_j > _inf(lim.max_jointed_palm_bodies) or n_j > _inf(lim.max_finger_chains):
            return False
        if not lim.allow_stacked_palm_joints and self.struct.carrier(parent) is not None:
            return False
        capacity = (_inf(lim.max_finger_chains) - n_j) + n_j * _inf(lim.max_digits_per_jointed_palm_body)
        return capacity >= digits_to_place

    def leaves_feasible(self, palm: Dict[str, Tuple[str, bool]], digits_to_place: int) -> bool:
        """While sampling a fresh hand under ``require_digit_on_palm_body``:
        can ``digits_to_place`` digits cover every palm leaf (one digit
        each, on the leaf) of this palm structure within the limits?"""
        st = Structure(palm=dict(palm))
        lim = self.limits
        leaves = st.palm_leaves()
        if len(leaves) > digits_to_place:
            return False
        demand = Counter(st.carrier(leaf) for leaf in leaves)
        n_j = len(st.jointed())
        if demand[None] + n_j > _inf(lim.max_finger_chains):
            return False
        if lim.max_digits_per_jointed_palm_body is not None:
            if any(c is not None and k > lim.max_digits_per_jointed_palm_body for c, k in demand.items()):
                return False
        return True

    def digit_cap(self) -> float:
        """Most top-level digits a fresh hand may get. Every digit fills a
        finger chain when no palm body is jointed, so this is
        ``min(max_digits, max_finger_chains)``."""
        return min(_inf(self.limits.max_digits), _inf(self.limits.max_finger_chains))

    # ---- joints per digit ----------------------------------------------------------------

    def joint_room(self, digit_id: str, struct: Optional[Structure] = None) -> Optional[int]:
        """Joints the top-level digit owning ``digit_id`` may still gain
        (``None``: no limit)."""
        if self.limits.max_joints_per_digit is None:
            return None
        st = struct or self.struct
        top = st.top_of().get(digit_id, digit_id)
        used = st.joints_per_digit().get(top, 0)
        return self.limits.max_joints_per_digit - used

    def begin_digit(self, room: Optional[int]) -> None:
        """Start the joint budget for one digit about to be emitted (with
        whatever branches it grows): ``room`` joints, ``None`` for no limit."""
        self.joint_left = room
        self.joint_reserved = 0

    def phalanx_cap(self) -> Optional[int]:
        if self.joint_left is None:
            return None
        return self.joint_left - self.joint_reserved

    def spend(self, n: int) -> None:
        if self.joint_left is not None:
            self.joint_left -= n

    def branch_cap(self, max_branch_digits: int, unit: int) -> int:
        """Most branch digits one phalanx may spawn now (each needs at least
        ``unit`` joints)."""
        if not self.limits.allow_branches:
            return 0
        cap = self.phalanx_cap()
        if cap is None:
            return max_branch_digits
        return min(max_branch_digits, max(0, cap) // max(1, unit))

    def reserve(self, n: int) -> None:
        if self.joint_left is not None:
            self.joint_reserved += n

    def release(self, n: int) -> None:
        if self.joint_left is not None:
            self.joint_reserved -= n


def context(limits: Optional[GenerationLimits], steps: Sequence = (), fresh: bool = False) -> Optional[LimitContext]:
    """A ``LimitContext`` for ``limits`` (``None`` stays ``None``). ``fresh``:
    sampling a new hand from scratch, so the baseline excess is zero."""
    if limits is None:
        return None
    if fresh:
        return LimitContext(limits, steps, base={k: 0 for k in LIMIT_KEYS})
    return LimitContext(limits, steps)


__all__ = [
    "DEFAULT_LIMITS",
    "GenerationLimits",
    "LIMIT_KEYS",
    "LIMIT_TEXT",
    "LimitContext",
    "LimitReport",
    "MODULE_KINDS",
    "PRESETS",
    "SIMULATOR",
    "SIMULATOR_ENVELOPE",
    "Structure",
    "UNLIMITED",
    "check",
    "context",
    "within",
]
