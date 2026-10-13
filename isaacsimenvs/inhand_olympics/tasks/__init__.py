"""Tasks. Each is a sequence of 6D goal poses for the pen: a task places the pen and sets the first goal at reset
(``reset``), and sets the next goal when one is hit (``next_goal``). Success, rewards and terminations are the env's.

TASK_NAMES fixes the one-hot order the policy sees; append new tasks, never reorder.
"""
from __future__ import annotations

from .pen_spin import PenSpin

TASK_NAMES = ("pen_spin",)
TASK_CLASSES = {"pen_spin": PenSpin}


def make_tasks(env) -> list:
    """The configured tasks, in TASK_NAMES order, each built once for the env."""
    names = env.cfg.task.names
    unknown = set(names) - set(TASK_NAMES)
    if unknown:
        raise ValueError(f"unknown tasks {sorted(unknown)}; known: {TASK_NAMES}")
    return [TASK_CLASSES[n](env) for n in TASK_NAMES if n in names]
