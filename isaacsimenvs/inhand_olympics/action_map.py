"""Finger actions in [-1, 1] <-> absolute joint targets (ActionCfg.finger_target). Kit-free; the env delegates here.

"range": lower + (a + 1) / 2 * (upper - lower): 0 is the middle of each joint's range.
"home":  0 is the hand's default pose, +1 the upper limit, -1 the lower, each side scaled separately:
         home + a * (upper - home) for a >= 0, home + a * (home - lower) for a < 0.
"""
from __future__ import annotations

import torch

MODES = ("range", "home")


def action_to_target(a: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, home: torch.Tensor, mode: str) -> torch.Tensor:
    a = a.clamp(-1.0, 1.0)
    if mode == "home":
        return home + torch.where(a >= 0.0, a * (hi - home), a * (home - lo))
    if mode == "range":
        return lo + 0.5 * (a + 1.0) * (hi - lo)
    raise ValueError(f"finger_target must be one of {MODES}, got {mode!r}")


def target_to_action(t: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, home: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "home":
        up, down = (hi - home).clamp(min=1e-6), (home - lo).clamp(min=1e-6)
        return torch.where(t >= home, (t - home) / up, (t - home) / down).clamp(-1.0, 1.0)
    if mode == "range":
        return 2.0 * (t - lo) / (hi - lo).clamp(min=1e-6) - 1.0
    raise ValueError(f"finger_target must be one of {MODES}, got {mode!r}")
