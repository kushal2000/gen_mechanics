"""Does the tolerance curriculum survive a checkpoint round trip?

rl_games saves whatever ``vec_env.get_env_state()`` returns and hands it back to
``set_env_state()`` on resume. Nothing implemented either, so env_state was None
in every checkpoint this repo wrote, and a resumed run silently restarted its
curriculum. That is worse than it sounds: ``_current_success_tolerance`` also
scales the keypoint reward, so the continuation trained an EASIER task on a
DIFFERENT reward scale than the checkpoint was written under -- and the reward
curve jumped in a way that read as the fine-tune working unusually well.
Nothing errored. It was caught by eye, on a curve.

So the round trip is tested rather than assumed. No Isaac Sim: the state is
plain data on the env, and the rl_games adapter only ferries it, so both halves
can be exercised against a stub.

    .venv_isaacsim/bin/python tests/test_curriculum_checkpoint_state.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeEnv:
    """The curriculum-state half of PoseReachEnv, without a simulator.

    Bound from the real methods, so this cannot drift from the env: if the
    implementation changes, the test exercises the change.
    """

    def __init__(self, tol=0.075, resume_override=0.0):
        from genmech.tasks.pose_reach.utils import curriculum

        self._current_success_tolerance = tol
        self._frame_counter = 0
        self._last_curriculum_update = 0
        self.cfg = SimpleNamespace(
            termination=SimpleNamespace(resume_success_tolerance=resume_override))
        self.get_curriculum_state = lambda: curriculum.get_curriculum_state(self)
        self.set_curriculum_state = lambda st: curriculum.set_curriculum_state(self, st)


def check_round_trip() -> None:
    src = _FakeEnv(tol=0.075)
    # Walk the curriculum the way the real one does: x0.9 per step.
    for _ in range(11):
        src._current_success_tolerance *= 0.9
    src._frame_counter, src._last_curriculum_update = 265_600, 264_000
    state = src.get_curriculum_state()

    dst = _FakeEnv(tol=0.075)          # a fresh env starts at the curriculum START
    assert dst._current_success_tolerance == 0.075
    dst.set_curriculum_state(state)

    assert abs(dst._current_success_tolerance - src._current_success_tolerance) < 1e-12, \
        f"tolerance not restored: {dst._current_success_tolerance}"
    assert dst._frame_counter == 265_600
    assert dst._last_curriculum_update == 264_000
    print(f"  OK  round trip restored {dst._current_success_tolerance:.6f} "
          f"(a fresh env would have used 0.075)")


def check_missing_state_is_survivable() -> None:
    """Checkpoints written before this existed carry None, and must still load."""
    for empty in (None, {}):
        dst = _FakeEnv(tol=0.075)
        dst.set_curriculum_state(empty)
        assert dst._current_success_tolerance == 0.075
    print("  OK  env_state=None / {} leaves the configured tolerance alone")


def check_explicit_override_wins() -> None:
    """An explicit resume_success_tolerance beats the checkpoint.

    It is a direct instruction, and it is the ONLY way to place the curriculum
    when continuing a checkpoint from before this state existed.
    """
    dst = _FakeEnv(tol=0.031, resume_override=0.031)
    dst.set_curriculum_state({"version": 1, "current_success_tolerance": 0.075,
                              "frame_counter": 10, "last_curriculum_update": 5})
    assert dst._current_success_tolerance == 0.031, dst._current_success_tolerance
    # Counters still restore -- they do not conflict with the override.
    assert dst._frame_counter == 10
    print("  OK  explicit resume_success_tolerance overrides the checkpoint")


def check_partial_state() -> None:
    """A future/older schema restores what matches instead of raising."""
    dst = _FakeEnv(tol=0.075)
    dst.set_curriculum_state({"version": 99, "frame_counter": 7})
    assert dst._frame_counter == 7
    assert dst._current_success_tolerance == 0.075
    print("  OK  unknown version restores what it can, without raising")


def check_vecenv_ferries_it() -> None:
    """The rl_games adapter must delegate, and tolerate a non-genmech env."""
    src = _FakeEnv(tol=0.05)

    class _Adapter:
        """Mirrors _GenMechRlGamesGpuEnv's delegation."""
        def __init__(self, inner):
            self.env = SimpleNamespace(unwrapped=inner)

        def _genmech_env(self):
            inner = getattr(self.env, "unwrapped", self.env)
            return inner if hasattr(inner, "get_curriculum_state") else None

        def get_env_state(self):
            i = self._genmech_env()
            return None if i is None else i.get_curriculum_state()

        def set_env_state(self, st):
            i = self._genmech_env()
            if i is not None:
                i.set_curriculum_state(st)

    assert _Adapter(src).get_env_state()["current_success_tolerance"] == 0.05
    # A non-genmech env must not blow up the save path.
    assert _Adapter(SimpleNamespace()).get_env_state() is None
    _Adapter(SimpleNamespace()).set_env_state({"current_success_tolerance": 1.0})
    print("  OK  adapter delegates, and ignores a non-genmech env")


def main() -> int:
    print("Curriculum state across a checkpoint:")
    check_round_trip()
    check_missing_state_is_survivable()
    check_explicit_override_wins()
    check_partial_state()
    check_vecenv_ferries_it()
    print("\ncurriculum checkpoint state test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
