"""Per-episode logging and an in-run viewer for the IsaacLab in-hand reference.

WHY THIS EXISTS. rl_games logs `rewards/step` and nothing else about the task, so a
reference run showed a rising scalar with no way to ask WHICH term was rising, how many
goals an episode actually scored, or what the hand was doing. Our own env answers all
three (``episode_cumulative/*``, ``episode_final/*``) from the env side. A manager-based
env cannot do that -- but it does not need to, because Isaac Lab already computes the
same quantities and puts them in ``extras["log"]`` on every reset. Nothing was reading
them: the fork's DefaultAlgoObserver only knows about 'lives', 'battle_won' and
'scores'. This observer reads them.

THREE THINGS WORTH KNOWING about the numbers, all verified in the installed source
rather than assumed:

* Reward terms are accumulated as ``func * weight * dt`` (reward_manager.py:150), and
  ``RewardManager.reset`` divides the episode sum by ``max_episode_length_s``
  (reward_manager.py:120) -- so the value in the log is a per-SECOND rate, not a return.
  Multiplying by ``max_episode_length_s`` recovers the mean episode return in the units
  the agent is actually optimising, which is what makes ``episode_return/*`` sum to
  ``episode_return/total`` and stay comparable to rl_games' own ``rewards/step``.
* ``Metrics/object_pose/consecutive_success`` increments once per step spent inside the
  threshold, but the command term resamples the goal the moment it is reached, so the
  error jumps immediately and the counter advances once per GOAL. It is per-episode:
  ``CommandTerm.reset`` zeroes it for the envs that reset. That is successes/episode.
* ``extras["log"]`` is rebuilt from scratch in ``_reset_idx`` and left untouched on
  steps where nothing reset, so the dict goes STALE rather than empty. Accumulating it
  every step would count the same batch many times over. Hence the weighting by the
  number of envs that finished on that step, which also makes the epoch average a
  correctly pooled mean over episodes instead of a mean of means.

The viewer piggybacks on the rollout: ``process_infos`` runs once per env step, so a
frame is read from env 0 there and nothing extra is ever stepped. The on-policy data is
untouched, which is the whole reason this is not a separate capture loop.
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Any, Callable

from rl_games.common.algo_observer import AlgoObserver

from .viewer import build_html, capture_frame


class InHandObserver(AlgoObserver):
    """Logs per-episode reward breakdowns and successes, and renders a viewer page.

    Args:
        env: the ``ManagerBasedRLEnv`` itself (already unwrapped). Passed in rather than
            dug out of ``algo.vec_env``, whose nesting is an rl_games implementation
            detail that has changed between versions.
        viewer_every: epochs between viewer pages; 0 disables the viewer entirely.
        viewer_frames: length of the rolling frame buffer a page is built from.
        log_html: ``(key, html) -> None``, e.g. a wandb.Html log. Optional; the page is
            written to ``out_dir`` either way so it survives without wandb.
    """

    def __init__(self, env, *, viewer_every: int = 200, viewer_frames: int = 400,
                 viewer_stride: int = 1, out_dir: str = "",
                 log_html: Callable[[str, str], None] | None = None) -> None:
        super().__init__()
        self.env = env
        self.viewer_every = int(viewer_every)
        self.viewer_stride = max(1, int(viewer_stride))
        self.out_dir = out_dir
        self.log_html = log_html
        self.writer = None
        self._frames: deque = deque(maxlen=int(viewer_frames))
        self._steps = 0
        self._last_page_epoch = -1
        # name -> [weighted sum, total weight]; weight is the number of episodes behind
        # each reported mean, so the epoch value is a pooled mean over episodes.
        self._acc: dict[str, list[float]] = {}
        self._episodes = 0.0
        # Reward terms carry seconds; read from the env so the ladder's 10 s rungs are
        # not silently scaled by the reference's 20 s.
        self.episode_s = float(getattr(env, "max_episode_length_s", 1.0))
        # Frames land one per policy step, thinned by the stride. Read from the env so the
        # decimation-2 rung plays back at its own 60 Hz rather than the reference's 30.
        step_dt = float(getattr(env, "step_dt", 1.0 / 30.0)) or (1.0 / 30.0)
        self.fps = 1.0 / (step_dt * self.viewer_stride)

    # --- rl_games hooks ---------------------------------------------------------

    def after_init(self, algo) -> None:
        self.writer = algo.writer

    def process_infos(self, infos, done_indices, **kwargs) -> None:
        if self.viewer_every and self._steps % self.viewer_stride == 0:
            try:
                self._frames.append(capture_frame(self.env, 0))
            except Exception:
                self.viewer_every = 0          # never let drawing break training
        self._steps += 1

        if not isinstance(infos, dict):
            return
        log = infos.get("episode")
        if not log:
            return
        # No dones means the dict is last reset's, already counted. Skipping here is
        # what keeps this an average over episodes rather than over steps.
        n = len(done_indices) if done_indices is not None else 0
        if n == 0:
            return
        self._episodes += n
        for key, value in log.items():
            try:
                v = float(value.item() if hasattr(value, "item") else value)
            except Exception:
                continue
            if not math.isfinite(v):
                continue
            slot = self._acc.setdefault(key, [0.0, 0.0])
            slot[0] += v * n
            slot[1] += n

    def after_print_stats(self, frame, epoch_num, total_time) -> None:
        if self.writer is not None and self._acc:
            self._write(frame)
        self._acc.clear()
        self._episodes = 0.0
        if (self.viewer_every and epoch_num % self.viewer_every == 0
                and epoch_num != self._last_page_epoch and len(self._frames) > 1):
            self._last_page_epoch = epoch_num
            self._emit_page(epoch_num)

    # --- logging ----------------------------------------------------------------

    def _write(self, frame: int) -> None:
        mean = {k: s / w for k, (s, w) in self._acc.items() if w > 0}
        out: dict[str, float] = {}

        # Per-term episode returns, in the agent's own units, plus each term's share of
        # the total. The share is what answers "which term is the policy chasing".
        returns = {k.split("/", 1)[1]: v * self.episode_s
                   for k, v in mean.items() if k.startswith("Episode_Reward/")}
        total = sum(returns.values())
        for name, v in returns.items():
            out[f"episode_return/{name}"] = v
            if abs(total) > 1e-9:
                out[f"episode_return_share/{name}"] = v / total
        if returns:
            out["episode_return/total"] = total

        # Goals scored per episode, and how close it got. Named to match our own env's
        # keys so the two runs can be read on one wandb chart.
        for src, dst, scale in (
                ("Metrics/object_pose/consecutive_success", "episode_final/successes", 1.0),
                ("Metrics/object_pose/orientation_error",
                 "episode_final/orientation_error_deg", 180.0 / math.pi),
                ("Metrics/object_pose/position_error", "episode_final/position_error_m", 1.0)):
            if src in mean:
                out[dst] = mean[src] * scale

        # CAUTION, and the one asymmetry in this file: Isaac Lab's Episode_Termination/* is
        # last_episode_dones.float().mean(dim=0) over ALL envs (termination_manager.py:144,
        # :180), i.e. the fraction of every env whose MOST RECENT episode ended this way --
        # a whole-population statistic that persists between an env's terminations. Every
        # other key here is an average over the episodes that ended in this epoch. So these
        # sum to ~1 only once every env has terminated at least once; early in a run the
        # shortfall is envs still inside their first episode, not a missing cause.
        for k, v in mean.items():
            if k.startswith("Episode_Termination/"):
                out[f"episode_final/done_{k.split('/', 1)[1]}"] = v

        # Anything Isaac Lab logs that this does not recognise still gets through, so a
        # new term or a curriculum appearing in the config is not silently dropped.
        known = ("Episode_Reward/", "Episode_Termination/", "Metrics/object_pose/")
        for k, v in mean.items():
            if not k.startswith(known):
                out[f"isaaclab_log/{k}"] = v

        out["episode_final/episodes_counted"] = self._episodes
        for k, v in out.items():
            self.writer.add_scalar(k, v, frame)

    # --- viewer -----------------------------------------------------------------

    def _emit_page(self, epoch_num: int) -> None:
        try:
            html = build_html(list(self._frames), fps=self.fps)
        except Exception as exc:                # a broken page must not kill a run
            print(f"[observer] viewer page failed: {exc}", flush=True)
            self.viewer_every = 0
            return
        if self.out_dir:
            d = os.path.join(self.out_dir, "viewers")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"epoch_{epoch_num:06d}.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"[observer] viewer {path} ({os.path.getsize(path) / 1024:.0f} KiB)",
                  flush=True)
        if self.log_html is not None:
            try:
                self.log_html("viewer/rollout", html)
            except Exception as exc:
                print(f"[observer] wandb viewer log failed: {exc}", flush=True)


__all__ = ["InHandObserver"]
