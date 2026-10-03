"""CPU part of ``tools/zero_action_check.py``: summarising per-design
episodes under zero actions (held PD targets) from the design-scoring
snapshot."""

from __future__ import annotations

from isaacsimenvs.inhand_reorient.tools import zero_action_check as z


def test_summary_per_design():
    snap = {"designs": {"0": {"source": "projected:allegro_right", "episodes": 10, "time_held_sum_s": 50.0,
                              "rotation_progress_sum_rad": 6.28, "envs_per_design": 4},
                        "1": {"source": "arch:x", "episodes": 0, "time_held_sum_s": 0.0,
                              "rotation_progress_sum_rad": 0.0, "envs_per_design": 4}}}
    rows = z.summarise(snap)
    assert rows["projected:allegro_right"]["ttt_mean_s"] == 5.0
    assert rows["projected:allegro_right"]["episodes"] == 10
    assert rows["arch:x"]["ttt_mean_s"] is None
