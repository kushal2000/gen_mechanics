"""Per-design episode returns, accumulated from the env step and written to disk.

Selection needs one number per DESIGN, and nothing in the loop produces it:
rl_games sees per-env rewards and dones, the env knows which design each env
holds, and the two are never joined. This wrapper joins them -- running return
per env, banked into its design's total when the episode ends -- and writes the
table periodically, because rl_games does not call close() on MAX_EPOCHS and a
file written only at exit would never be written.

Every rank writes its own file. With a population smaller than the env count
both ranks hold every design; with a larger one they hold disjoint halves. The
selection step sums across ranks either way.
"""

from __future__ import annotations

import atexit
import json
import time
from pathlib import Path

import gymnasium as gym
import torch


_LIVE: list["DesignRewardWrapper"] = []
"""Every wrapper built in this process, so ``flush_all`` can reach them.
train.py leaves through ``os._exit`` -- Kit's shutdown hangs otherwise -- and
``os._exit`` runs no atexit handlers. The smoke that motivated the atexit hook
(48 steps, no file) still wrote nothing: the hook was registered and never ran."""


def flush_all() -> None:
    """Write every live table. Call before ``os._exit``."""
    for w in _LIVE:
        w.flush()


class DesignRewardWrapper(gym.Wrapper):
    """Bank each env's episode return against the design it holds."""

    def __init__(self, env: gym.Env, *, output_path: str | Path, rank: int,
                 flush_every: int = 50) -> None:
        super().__init__(env)
        inner = self.env.unwrapped
        record = getattr(inner, "scene_record", None)
        idx = getattr(record, "robot_design_index", None)
        if idx is None:
            raise RuntimeError(
                "design rewards need a population: scene_record.robot_design_index is None")
        population = record.population

        self.output_path = Path(output_path)
        self.rank = int(rank)
        self.flush_every = int(flush_every)
        self.design_of_env = idx.to(torch.long)                    # (N,)
        self.n_designs = int(population.n_designs)
        dev = self.design_of_env.device

        self.running = torch.zeros(inner.num_envs, device=dev)     # return so far, per env
        self.sum = torch.zeros(self.n_designs, device=dev)         # banked, per design
        self.count = torch.zeros(self.n_designs, device=dev)
        self.goals_sum = torch.zeros(self.n_designs, device=dev)      # goals hit, summed
        self.succeeded = torch.zeros(self.n_designs, device=dev)      # episodes with >= 1 goal
        self._steps = 0
        self._t0 = time.time()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # rl_games does not call close() on MAX_EPOCHS, so the final window --
        # or, in a run shorter than flush_every, the whole thing -- would never
        # reach disk. atexit covers a normal interpreter exit; train.py's
        # os._exit does not run it, so train.py also calls flush_all().
        atexit.register(self.flush)
        _LIVE.append(self)
        print(f"[design_rewards] rank {self.rank}: {inner.num_envs} envs over "
              f"{int(self.design_of_env.unique().numel())} of {self.n_designs} designs "
              f"-> {self.output_path}", flush=True)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        self._steps += 1

        r = rew.reshape(-1).to(self.running.dtype)
        self.running += r
        done = (terminated | truncated).reshape(-1)
        if done.any():
            d = self.design_of_env[done]
            self.sum.index_add_(0, d, self.running[done])
            self.count.index_add_(0, d, torch.ones_like(self.running[done]))
            # The env's `successes` is the number of goals reached THIS episode,
            # not a 0/1 -- summing it and dividing by episodes gave a "success
            # rate" of 2.2. Keep the goal count and, separately, whether the
            # episode reached any goal at all; the second is the rate.
            succ = self._successes(info)
            if succ is not None:
                g = succ.reshape(-1)[done].to(self.running.dtype)
                self.goals_sum.index_add_(0, d, g)
                self.succeeded.index_add_(0, d, (g > 0).to(self.running.dtype))
            self.running[done] = 0.0

        if self._steps % self.flush_every == 0:
            self.flush()
        return obs, rew, terminated, truncated, info

    def _successes(self, info):
        inner = self.env.unwrapped
        for src in (info if isinstance(info, dict) else {}, getattr(inner, "extras", {})):
            for key in ("successes", "success"):
                v = src.get(key) if isinstance(src, dict) else None
                if torch.is_tensor(v) and v.numel() == self.running.numel():
                    return v
        return None

    def flush(self) -> None:
        if self.count.sum().item() == 0 and self._steps == 0:
            return                                  # nothing to say yet
        count = self.count.cpu()
        total = self.sum.cpu()
        goals = self.goals_sum.cpu(); succ = self.succeeded.cpu()
        mean = torch.where(count > 0, total / count.clamp(min=1), torch.zeros_like(total))
        inner = self.env.unwrapped
        payload = {
            "rank": self.rank,
            "steps": self._steps,
            # The curriculum lives on the env and does not survive a checkpoint
            # (see reward_utils/curriculum.py). The next generation reads it
            # from here and passes resume_success_tolerance, or it silently
            # restarts at 0.075 on a different reward scale.
            "success_tolerance": float(getattr(inner, "_current_success_tolerance", float("nan"))),
            "elapsed_s": round(time.time() - self._t0, 1),
            "n_designs": self.n_designs,
            "episodes": int(count.sum().item()),
            # One row per design this rank holds; the merge sums across ranks.
            "designs": {
                str(i): {"episodes": int(count[i]), "return_sum": float(total[i]),
                         "return_mean": float(mean[i]),
                         "goals_sum": float(goals[i]), "succeeded": int(succ[i])}
                for i in range(self.n_designs) if count[i] > 0
            },
        }
        tmp = self.output_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(self.output_path)          # atomic: a reader never sees half a file

    def close(self) -> None:
        self.flush()
        return self.env.close()


def merge_rank_files(paths) -> dict:
    """Sum the per-rank tables into one ``design -> stats`` dict."""
    acc: dict[int, dict] = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        for k, row in data["designs"].items():
            a = acc.setdefault(int(k), {"episodes": 0, "return_sum": 0.0, "goals_sum": 0.0, "succeeded": 0})
            a["episodes"] += row["episodes"]
            a["return_sum"] += row["return_sum"]
            a["goals_sum"] += row.get("goals_sum", row.get("success_sum", 0.0))
            a["succeeded"] += row.get("succeeded", 0)
    for a in acc.values():
        n = max(a["episodes"], 1)
        a["return_mean"] = a["return_sum"] / n
        a["goals_per_episode"] = a["goals_sum"] / n
        a["success_rate"] = a["succeeded"] / n
    return acc
