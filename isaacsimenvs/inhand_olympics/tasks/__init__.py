"""Tasks. Each is a sequence of 6D goal poses for the pen: a task places the pen and sets the first goal at reset
(``reset``), and sets the next goal when one is hit (``next_goal``). A task may also set ``success_steps`` (consecutive
steps within tolerance a goal needs; default 1) and ``perturbed`` (the env pushes and twists the pen while it is
lifted; default False) and ``carried`` (once the first goal is hit, the pen touching the table ends the episode;
default False). Success, rewards and terminations are the env's.

TASK_NAMES fixes the one-hot order the policy sees; append new tasks, never reorder. The one-hot is NUM_TASK_SLOTS
wide whatever the number of tasks, so adding one does not change the observation (or orphan a checkpoint).
"""
from __future__ import annotations

from .pen_flip import PenFlipAir
from .pen_lift import PenLift
from .pen_repose import PenRepose
from .pen_spin import PenSpin, PenSpinAir

TASK_NAMES = ("pen_spin", "pen_lift", "pen_spin_air", "pen_flip_air", "pen_repose")
TASK_CLASSES = {"pen_spin": PenSpin, "pen_lift": PenLift, "pen_spin_air": PenSpinAir, "pen_flip_air": PenFlipAir,
                "pen_repose": PenRepose}
NUM_TASK_SLOTS = 8
assert len(TASK_NAMES) <= NUM_TASK_SLOTS


def make_tasks(env) -> list:
    """The configured tasks, in TASK_NAMES order, each built once for the env."""
    names = env.cfg.task.names
    unknown = set(names) - set(TASK_NAMES)
    if unknown:
        raise ValueError(f"unknown tasks {sorted(unknown)}; known: {TASK_NAMES}")
    return [TASK_CLASSES[n](env) for n in TASK_NAMES if n in names]
