"""One child: one grammar mutation of its parent (or a fresh random hand), built into the files the env loads, and
checked -- the grammar's rules and clearance (``operators.mutate`` keeps them by construction), not a duplicate of any
hand ever made, and fits the run's slot grid (so the shared network takes it)."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from hand_sampler.grammar import hand as gh
from hand_sampler.grammar import operators as gops
from hand_sampler.grammar.urdf_export import build

from .config import EvolutionConfig


def design_key(hand, rules) -> str:
    """Canonical JSON: the same hand turned or mirrored on the palm is the same key."""
    return gh.hand_to_json(gops.canonical(hand, rules))


def fits_grid(spec_path: str, grid: dict) -> bool:
    from isaacsimenvs.common.robots import get_robot_spec
    from isaacsimenvs.common.robots.finger_layout import SlotGrid, finger_layout
    try:
        finger_layout(get_robot_spec(spec_path)).with_grid(SlotGrid(**grid))
        return True
    except ValueError:
        return False


def make_child(parent_design: Optional[dict], name: str, out_dir: Path, cfg: EvolutionConfig,
               rng: np.random.Generator, taken: set) -> Optional[dict]:
    """Build ``name`` under ``out_dir/name``; returns its record fields, or None (no admissible child this time)."""
    rules = cfg.rule_set()
    if parent_design is None:
        hand, op = gops.random_hand(rng, rules), "random"
    else:
        res = gops.mutate(gh.hand_from_dict(parent_design), rng, rules, tries=cfg.mutation_tries)
        if res is None:
            return None
        hand, op, _ = res
    if gh.check(hand, rules):
        return None
    key = design_key(hand, rules)
    if key in taken:
        return None
    row = build(hand, name, out_dir / name)
    if not fits_grid(row["spec"], cfg.grid):
        shutil.rmtree(out_dir / name, ignore_errors=True)
        return None
    taken.add(key)
    return {"spec": row["spec"], "design": gh.hand_to_dict(hand), "operator": op, "key": key}


__all__ = ["design_key", "fits_grid", "make_child"]
