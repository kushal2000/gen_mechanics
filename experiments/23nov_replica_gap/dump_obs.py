"""The actor's observation, field by field, for one hand (23 Nov 2026): the env built exactly as zero_shot_commercial.py
builds it (the policy's training config, multi:finger:<hand>, seed 7400), env 0's obs at reset and over the first steps
under the policy (deterministic actions). Run once per hand, then compare (--compare a.npz b.npz).

    .venv_isaacsim/bin/python experiments/23nov_replica_gap/dump_obs.py --hand <hand> --policy v9/k5_margin:2 --out x.npz
    .venv_isaacsim/bin/python experiments/23nov_replica_gap/dump_obs.py --compare a.npz b.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/14nov_analyze_v4"))


def dump(a) -> None:
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    la, _ = lp.parse_known_args([])
    la.headless = True
    app = AppLauncher(la).app  # noqa: F841
    import torch
    import yaml
    from omegaconf import OmegaConf
    import isaacsimenvs  # noqa: F401
    import zero_shot_commercial as zs
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg
    from isaacsimenvs.inhand_olympics.observations import field_sizes
    from isaacsimenvs.inhand_olympics.tasks import NUM_TASK_SLOTS
    zs.CAMPAIGN = REPO / "debug_outputs/coevolve/v9"
    p = zs.resolve(a.policy)
    full = OmegaConf.to_container(OmegaConf.load(Path(p["train_run"]) / ".hydra/config.yaml"), resolve=True)
    env_d = full["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    ref = f"multi:finger:{zs.hand_ref(a.hand)}"
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    cfg.scene.num_envs = 4
    cfg.reset.stagger_initial_episodes = False
    cfg.reset.table_safe = True
    cfg.log.group_stats = False
    cfg.log.design_stats = False
    cfg.seed = 7400
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()
    spec = env.hand_set.specs[0] if env.hand_set is not None else None
    n_hand = int(env.joint_enabled.shape[-1])
    sizes = field_sizes(env.cfg.action_space, n_hand, env.fingertips().shape[1], NUM_TASK_SLOTS)
    acfg = full["agent"]
    acfg["params"]["config"]["multi_gpu"] = False
    acfg["params"]["network"]["robot_spec"] = ref
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump({"train": acfg}, f)
    f.close()
    player = RlPlayer(obs["policy"].shape[-1], env.action_space.shape[-1], f.name, p["policy"], device=env.device,
                      sapg_expl_coef=0.0, num_envs=env.num_envs)
    obs, _ = env.reset()
    rows, acts = [obs["policy"][0].cpu().numpy()], []
    for _ in range(a.steps):
        with torch.no_grad():
            act = player.get_normalized_action(obs["policy"], deterministic_actions=True)
        acts.append(act[0].cpu().numpy())
        obs, *_ = env.step(act)
        rows.append(obs["policy"][0].cpu().numpy())
    np.savez(a.out, obs=np.stack(rows), act=np.stack(acts), fields=np.array(list(env.cfg.obs.obs_list)),
             sizes=np.array([sizes[k] for k in env.cfg.obs.obs_list]), enabled=env.joint_enabled[0].cpu().numpy())
    print(f"[dump_obs] {a.hand}: {len(rows)} obs of width {rows[0].shape[0]} -> {a.out}", flush=True)
    os._exit(0)


def compare(pa: str, pb: str) -> None:
    A, B = np.load(pa), np.load(pb)
    fields, sizes = list(A["fields"]), list(A["sizes"])
    assert fields == list(B["fields"]) and list(sizes) == list(B["sizes"]), "different layouts"
    print(f"enabled slots A {np.flatnonzero(A['enabled']).tolist()}\n              B {np.flatnonzero(B['enabled']).tolist()}")
    off = 0
    for f, s in zip(fields, sizes):
        a, b = A["obs"][:, off:off + s], B["obs"][:, off:off + s]
        d0, dmax = np.abs(a[0] - b[0]), np.abs(a - b).max(axis=0)
        print(f"{f:22s} w {s:4d}  reset |A-B| max {d0.max():7.3f} mean {d0.mean():6.3f}   over steps max {dmax.max():7.3f}"
              f"   |A| mean {np.abs(a[0]).mean():6.3f}")
        if f in ("joint_pos", "joint_enabled", "joint_origin") or d0.max() > 0.2:
            print(f"   A[0] {np.round(a[0][:24], 2).tolist()}\n   B[0] {np.round(b[0][:24], 2).tolist()}")
        off += s
    print("actions step 0  A", np.round(A["act"][0][:24], 2).tolist(), "\n                B", np.round(B["act"][0][:24], 2).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="")
    ap.add_argument("--policy", default="v9/k5_margin:2")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()
    compare(*a.compare) if a.compare else dump(a)
