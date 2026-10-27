"""Scene-step timing, shared across tasks."""

from __future__ import annotations

import time


def _log_scene_step(start_time: float, message: str) -> None:
    print(f"[scene_utils][+{time.perf_counter() - start_time:.2f}s] {message}", flush=True)
