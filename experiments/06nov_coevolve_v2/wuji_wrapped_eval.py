"""Zero-shot on the UNCHANGED vendor Wuji v2 URDF (wuji2_left_uniform_handonly, multi:finger: on the 6 x 5 grid),
with adapters only on what goes into the policy and what comes out of it. A ladder of cumulative adapters:

  A  none
  B  virtual limits: each joint shows the training range of its kind (flexion -30..90 deg, abduction -30..30) in
     joint_lower / joint_upper and in the normalised joint_pos / prev_joint_pos; the policy's action is read as a
     target in that virtual range (the env's "home" map, home 0) and clipped to the real limits
  C  + sign: joints whose positive angle opens (the thumb's CMC abduction) are shown and driven with the sign flipped
     (joint pos / vel / previous target / axis token, and the action)
  D  + thumb un-rolled: the thumb's axis and link tokens rotated about the thumb's long axis so its CMC flexion axis
     lies in the palm plane, as a training finger's does
  E  + x2 lengths: joint_origin, joint_link_vec and palm_extents shown twice as large (the evolved hands' size)

Every env's first episode (100 envs per rung, sampled actions, the SAPG leader), per controller.

    .venv_isaacsim/bin/python experiments/06nov_coevolve_v2/wuji_wrapped_eval.py [--envs 100] [--rungs A B C D E]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTROLLERS = [("k5_none", 7), ("k1_none", 8)]
KIND_RANGE = {"flex": (-30.0, 90.0), "abd": (-30.0, 30.0)}
FLIP = {"l_thumb_cmc_abd"}
OUT = REPO / "debug_outputs/coevolve/studies/wuji_wrapped"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=100)
    ap.add_argument("--rungs", nargs="+", default=list("ABCDE"))
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--hand", default="wuji2_left_uniform_handonly", help="registered name or vendor spec JSON")
    ap.add_argument("--tag", default="", help="results file suffix")
    ap.add_argument("--table-safe", type=int, default=1)
    ap.add_argument("--knuckle", type=int, default=0,
                    help="1: present each 2-DOF knuckle (flexion then abduction ~14 mm on) as training hands do -- "
                         "abduction first at the base with a zero-length link, then flexion with its link to the next "
                         "joint; tokens and actions of the two slots swapped (observation/action adapter only)")
    ap.add_argument("--dump-obs0", default=None, help="write the zero-pose observation after the adapters (rung A) "
                    "per field and exit (debugging the adapters)")
    ap.add_argument("--controllers", nargs="+", default=["v2/k5_none:latest"],
                    help="campaign/run:generation (or :latest = the newest generation with a trained policy); "
                         "v2 checkpoints by default since 7 Nov")
    ap.add_argument("--first-link-scales", type=float, nargs="+", default=[1.0],
                    help="observation-only intervention: each finger's FIRST joint's joint_link_vec token times this "
                         "(looped like --bias-mm); physics unchanged")
    ap.add_argument("--palm-extents-mm", type=float, nargs=3, default=None,
                    help="override the palm_extents token (thickness, width, length mm) at every rung")
    ap.add_argument("--bias-mm", type=float, nargs="+", default=[0.0],
                    help="wrist lowered by this much at each episode's start (an output adapter: the wrist's height "
                         "command and current height; the policy's own wrist actions apply from there)")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO))
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app  # noqa: F841

    import numpy as np
    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg
    from isaacsimenvs.inhand_olympics.observations import field_sizes

    def _resolve(ctl):
        cr, gg = ctl.split(":")
        if gg == "latest":
            gg = str(max(int(p_.parent.name[4:]) for p_ in (REPO / "debug_outputs/coevolve" / cr).glob("gen_*/policy.pth")))
        return f"{cr}:{gg}"
    args.controllers = [_resolve(c_) for c_ in args.controllers]
    print(f"[wrap] controllers {args.controllers}", flush=True)
    c0, g0 = args.controllers[0].split(":")
    first = REPO / "debug_outputs/coevolve" / c0
    tr0 = json.loads((first / "state.json").read_text())["train_runs"][str(int(g0))]
    full = OmegaConf.to_container(OmegaConf.load(next(Path(tr0).rglob(".hydra/config.yaml"))), resolve=True)
    env_d = full["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    ref = f"multi:finger:{args.hand}"
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    cfg.scene.num_envs = args.envs
    cfg.reset.stagger_initial_episodes = False
    cfg.reset.table_safe = bool(args.table_safe)      # finger-scene table-safe fixed 7 Nov
    cfg.log.group_stats = False
    cfg.log.design_stats = False
    cfg.seed = 7400
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()
    dev, n = env.device, env.num_envs

    # observation field offsets (observations.field_sizes, obs_list order)
    n_all = len(env.canon_joint_names)
    n_hand = env.joint_enabled.shape[1]
    n_tips = env.fingertips().shape[1]
    n_tasks = env.task_onehot().shape[1]
    sizes = field_sizes(n_all, n_hand, n_tips, n_tasks)
    off, o = {}, 0
    for f in env.cfg.obs.obs_list:
        off[f] = (o, o + sizes[f])
        o += sizes[f]
    assert o == obs["policy"].shape[-1], (o, obs["policy"].shape)

    # slots -> real joints, kinds, signs, real and virtual limits
    hs = env.hand_set
    spec, lay = hs.specs[0], hs.layouts[0]
    slot_name = {s: nm for nm, s in zip(spec.hand_joint_names, lay.slot_of_joint)}
    en = env.joint_enabled[0] > 0.5
    lo_r, hi_r = env.joint_lower_hand[0].clone(), env.joint_upper_hand[0].clone()
    vlo, vhi, sign = lo_r.clone(), hi_r.clone(), torch.ones(n_hand, device=dev)
    for k in range(n_hand):
        nm = slot_name.get(k)
        if nm is None:
            continue
        kind = "abd" if "abd" in nm else "flex"
        vlo[k], vhi[k] = (math.radians(x) for x in KIND_RANGE[kind])
        if nm in FLIP:
            sign[k] = -1.0
    thumb = [k for k in range(n_hand) if slot_name.get(k, "").startswith("l_thumb")]
    print(f"[wrap] slots: {len(slot_name)} joints, thumb slots {thumb}, flipped "
          f"{[slot_name[k] for k in range(n_hand) if sign[k] < 0]}", flush=True)

    def seg(x, f):
        return x[:, off[f][0]:off[f][1]]

    # thumb un-roll rotation (rung D): about the thumb's base->tip direction, putting CMC flex's axis in the palm plane
    def thumb_rotation(o_):
        ax = seg(o_, "joint_axis").view(n, n_hand, 3)[0]
        org = seg(o_, "joint_origin").view(n, n_hand, 3)[0]
        lv = seg(o_, "joint_link_vec").view(n, n_hand, 3)[0]
        d = (org[thumb[-1]] + lv[thumb[-1]] - org[thumb[0]])
        d = d / d.norm()
        a0 = ax[thumb[0]]
        best = None
        for phi in np.linspace(-math.pi, math.pi, 721):
            c, s = math.cos(phi), math.sin(phi)
            K = torch.tensor([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]], device=dev, dtype=torch.float32)
            R = torch.eye(3, device=dev) + s * K + (1 - c) * (K @ K)
            ar = R @ a0
            score = abs(float(ar[0]))
            if best is None or score < best[0] - 1e-9:
                best = (score, phi, R)
        print(f"[wrap] thumb un-roll: {math.degrees(best[1]):.0f} deg about the thumb axis (|axis . normal| -> {best[0]:.3f})")
        return best[2]

    # joint kinds: vendor names (*_abd) or, for a grammar spec, the design's axes (f<i>_j<j>)
    kind_of = {}
    sp_ = (REPO / args.hand) if args.hand.endswith(".spec.json") else None
    if sp_ is not None and "design" in json.loads(sp_.read_text()):
        des = json.loads(sp_.read_text())["design"]
        KD = {(0, 0): "flex", (90, 0): "abd", (0, 90): "roll"}
        for i, f_ in enumerate(des["fingers"]):
            for j_, jt in enumerate(f_["joints"]):
                kind_of[f"f{i}_j{j_}"] = KD.get(tuple(jt["axis"]), "flex")
    def kind(nm):
        return kind_of.get(nm, "abd" if nm.endswith("_abd") else "flex")
    knuckle_pairs = [(k, k + 1) for k in range(n_hand - 1)
                     if k in slot_name and (k + 1) in slot_name and kind(slot_name[k]) == "flex"
                     and kind(slot_name[k + 1]) == "abd" and k // 5 == (k + 1) // 5
                     and (k % 5 == 0 or (k - 1) not in slot_name)]          # a finger's base knuckle only
    hide_perm, shown = None, set()
    if args.knuckle == 2:
        # the non-thumb knuckles' abductions hidden; each finger's remaining joints packed to the front of its slots
        # (training hands have empty slots only at a finger's end), the freed slots shown as an empty slot
        knuckle_pairs = [(a, b) for a, b in knuckle_pairs if "thumb" not in slot_name[a]]     # by name (7 Nov fix)
        hidden = {b for _, b in knuckle_pairs}
        free = [k for k in range(n_hand) if k not in slot_name]
        ghost = max(free) if free else None
        perm = list(range(n_hand))
        for f0 in range(0, n_hand, 5):
            width = min(5, n_hand - f0)
            real = [k for k in range(f0, f0 + width) if k in slot_name and k not in hidden]
            empties = [k for k in range(f0, f0 + width) if k not in real]       # this finger's own freed slots
            for i in range(width):
                perm[f0 + i] = real[i] if i < len(real) else (ghost if ghost is not None else empties[i - len(real)])
        hide_perm = torch.tensor(perm, device=dev)
        shown = {k for k in slot_name if k not in hidden}
        print(f"[wrap] hide: policy slot -> real slot {[(i, slot_name.get(p_, 'empty')) for i, p_ in enumerate(perm) if i < 25]}", flush=True)
    if args.knuckle:
        print(f"[wrap] knuckle adapter mode {args.knuckle}: slot pairs {[(slot_name[a], slot_name[b]) for a, b in knuckle_pairs]}", flush=True)
    first_slots = [k for k in range(n_hand) if k in slot_name and (k == 0 or (k - 1) not in slot_name or k // 5 != (k - 1) // 5)]
    first_slots = [k for k in first_slots if k % 5 == 0 or (k - 1) // 5 != k // 5]
    cur_scale = [1.0]
    print(f"[wrap] first slots: {[(k, slot_name[k]) for k in first_slots]}", flush=True)
    R_thumb = thumb_rotation(obs["policy"]) if thumb else None    # a non-vendor hand (the control): no rung D

    def to_policy(o_, rung):
        x = o_.clone()
        if rung >= "B":
            h = slice(len(env.canon_joint_names) - n_hand, None)
            for f in ("joint_pos", "prev_joint_pos"):
                v = seg(x, f)[:, h]
                q = lo_r + 0.5 * (v + 1.0) * (hi_r - lo_r)                 # back to radians (real limits)
                if rung >= "C":
                    q = q * sign
                rng = (vhi - vlo).clamp(min=1e-6)
                seg(x, f)[:, h] = torch.where(en, 2.0 * (q - vlo) / rng - 1.0, v)
            if rung >= "C":
                for f in ("joint_vel", "prev_joint_vel", "prev_action_targets"):
                    seg(x, f)[:, h] *= sign
            seg(x, "joint_lower")[:] = torch.where(en, vlo, seg(x, "joint_lower"))
            seg(x, "joint_upper")[:] = torch.where(en, vhi, seg(x, "joint_upper"))
        if rung >= "C":
            axv = seg(x, "joint_axis").view(n, n_hand, 3)
            axv *= sign.view(1, -1, 1)
        if rung >= "D":
            for f in ("joint_axis", "joint_link_vec"):
                v = seg(x, f).view(n, n_hand, 3)
                v[:, thumb] = v[:, thumb] @ R_thumb.T
        if rung >= "E":
            for f in ("joint_origin", "joint_link_vec", "palm_extents"):
                seg(x, f)[:] *= 2.0
        if args.knuckle in (2, 3) and knuckle_pairs:
            wa = n_all - n_hand
            org = seg(x, "joint_origin").view(n, n_hand, 3)
            lv = seg(x, "joint_link_vec").view(n, n_hand, 3)
            ghost = max(k for k in range(n_hand) if k not in slot_name)
            for s_, t_ in knuckle_pairs:
                l_total = lv[:, s_] + lv[:, t_]
                if args.knuckle == 2:                     # hide (compaction applied below)
                    lv[:, s_] = l_total
                else:                                     # stub at the PIP: flexion's full link, abduction zero-length there
                    o_end = org[:, s_] + l_total
                    lv[:, s_] = l_total
                    org[:, t_] = o_end
                    lv[:, t_] = 0.0
        if args.knuckle == 2 and hide_perm is not None:
            wa = n_all - n_hand
            src = x.clone()
            for f in ("joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets"):
                if f in off:
                    seg(x, f)[:, wa:] = seg(src, f)[:, wa:][:, hide_perm]
            for f in ("joint_lower", "joint_upper", "joint_enabled"):
                if f in off:
                    seg(x, f)[:] = seg(src, f)[:, hide_perm]
            for f in ("joint_origin", "joint_axis", "joint_link_vec"):
                seg(x, f).view(n, n_hand, 3)[:] = seg(src, f).view(n, n_hand, 3)[:, hide_perm]
        if args.knuckle == 1 and knuckle_pairs:
            wa = n_all - n_hand
            org = seg(x, "joint_origin").view(n, n_hand, 3)
            lv = seg(x, "joint_link_vec").view(n, n_hand, 3)
            ax_ = seg(x, "joint_axis").view(n, n_hand, 3)
            for s_, t_ in knuckle_pairs:
                for f in ("joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets"):
                    if f in off:
                        v = seg(x, f)
                        a_, b_ = v[:, wa + s_].clone(), v[:, wa + t_].clone()
                        v[:, wa + s_], v[:, wa + t_] = b_, a_
                for f in ("joint_lower", "joint_upper", "joint_enabled"):
                    if f in off:
                        v = seg(x, f)
                        a_, b_ = v[:, s_].clone(), v[:, t_].clone()
                        v[:, s_], v[:, t_] = b_, a_
                a_, b_ = ax_[:, s_].clone(), ax_[:, t_].clone()
                ax_[:, s_], ax_[:, t_] = b_, a_
                o_flex = org[:, s_].clone()
                l_total = lv[:, s_] + lv[:, t_]
                org[:, s_], org[:, t_] = o_flex, o_flex
                lv[:, s_], lv[:, t_] = torch.zeros_like(l_total), l_total
        if cur_scale[0] != 1.0:
            lv2 = seg(x, "joint_link_vec").view(n, n_hand, 3)
            lv2[:, first_slots] *= cur_scale[0]
        if args.palm_extents_mm is not None:
            seg(x, "palm_extents")[:] = torch.tensor(args.palm_extents_mm, device=dev) / 1000.0
        return x

    def to_env(act, rung):
        if args.knuckle == 2 and hide_perm is not None:
            act = act.clone()
            wa_ = act.shape[-1] - n_hand
            pol = act[:, wa_:].clone()
            real = torch.zeros_like(pol)                   # the hidden abductions (and ghosts) held at home (0)
            for k_pol, k_real in enumerate(hide_perm.tolist()):
                if k_real in shown:
                    real[:, k_real] = pol[:, k_pol]
            act[:, wa_:] = real
        if args.knuckle == 1 and knuckle_pairs:
            act = act.clone()
            wa_ = act.shape[-1] - n_hand
            for s_, t_ in knuckle_pairs:
                a_, b_ = act[:, wa_ + s_].clone(), act[:, wa_ + t_].clone()
                act[:, wa_ + s_], act[:, wa_ + t_] = b_, a_          # policy slot s = abduction -> real slot t
        if rung < "B":
            return act
        a_ = act.clone()
        h = slice(len(env.canon_joint_names) - n_hand, None)
        av = a_[:, h].clamp(-1, 1)
        qv = torch.where(av >= 0, av * vhi, -av * vlo)                      # virtual target (home 0)
        q = qv * (sign if rung >= "C" else 1.0)
        q = torch.minimum(torch.maximum(q, lo_r), hi_r)
        ar = torch.where(q >= 0, q / hi_r.clamp(min=1e-6), q / (-lo_r).clamp(min=1e-6))
        a_[:, h] = torch.where(en, ar, av)
        return a_

    if args.dump_obs0:
        env.reset()
        jp0 = torch.zeros_like(env.robot.data.joint_pos)
        env.robot.write_joint_state_to_sim(jp0, torch.zeros_like(jp0))
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        o0 = env._get_observations()["policy"]
        x0 = to_policy(o0, "A")
        np.savez(args.dump_obs0, **{f: seg(x0, f)[0].cpu().numpy() for f in off},
                 **{"raw_" + f: seg(o0, f)[0].cpu().numpy() for f in off})
        print(f"[wrap] wrote {args.dump_obs0}", flush=True)
        import os
        sys.stdout.flush()
        os._exit(0)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for ctl in args.controllers:
        camp_run, g = ctl.split(":")
        g = int(g)
        run = camp_run.replace("/", "_")
        rd = REPO / "debug_outputs/coevolve" / camp_run
        tr = json.loads((rd / "state.json").read_text())["train_runs"][str(g)]
        acfg = OmegaConf.to_container(OmegaConf.load(next(Path(tr).rglob(".hydra/config.yaml"))), resolve=True)["agent"]
        acfg["params"]["config"]["multi_gpu"] = False
        acfg["params"]["network"]["robot_spec"] = ref
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            yaml.safe_dump({"train": acfg}, fh)
        player = RlPlayer(obs["policy"].shape[-1], env.action_space.shape[-1], fh.name,
                          str((rd / f"gen_{g:04d}/policy.pth").resolve()), device=dev, sapg_expl_coef=0.0, num_envs=n)
        zi = env.wrist_ids[2]                              # wrist_z (Lab index); its command is wrist_command[:, 2]
        top = float(cfg.table.top_height)

        def palm_height():
            from isaaclab.utils.math import quat_apply
            palm = env.robot.data.body_state_w[:, env.palm_body]
            c = palm[:, 0:3] + quat_apply(palm[:, 3:7], env.palm_center_offset)
            return float((c[:, 2] - env.scene.env_origins[:, 2]).mean() - top)
        for bias, rung, fls in [(b, r, f_) for b in args.bias_mm for r in args.rungs for f_ in args.first_link_scales]:
            cur_scale[0] = fls
            obs, _ = env.reset()
            if bias > 0:
                b = bias / 1000.0
                lo_w, hi_w = env.joint_lower[0, zi], env.joint_upper[0, zi]
                env.wrist_command[:, 2] = (env.wrist_command[:, 2] - b).clamp(lo_w, hi_w)
                env.targets[:, zi] = env.wrist_command[:, 2]
                jp = env.robot.data.joint_pos.clone()
                jp[:, zi] = (jp[:, zi] - b).clamp(lo_w, hi_w)
                env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
                obs = {"policy": env._get_observations()["policy"]}
            heights = {0: palm_height()}
            open_ = torch.ones(n, dtype=torch.bool, device=dev)
            goals, length = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
            t0, steps = time.time(), 0
            # behaviour (7 Nov): closest fingertip pad to the pen, the pen's highest lift, steps with a tip within
            # 20 mm of the pen, and how hard the hand joints are driven / how flexed they get (open episodes only)
            pen0 = env.pen_pose()[0][:, 2].clone()
            best_d = torch.full((n,), 9.0, device=dev)
            lift = torch.zeros(n, device=dev)
            near = torch.zeros(n, device=dev)
            hand_cols = slice(env.action_space.shape[-1] - n_hand, None)
            act_abs, qn_sum, nstep = torch.zeros(n_hand, device=dev), torch.zeros(n_hand, device=dev), 0
            base_sum = torch.zeros((), device=dev)
            with torch.no_grad():
                while steps < args.max_steps and bool(open_.any()):
                    act = player.get_action(to_policy(obs["policy"], rung), deterministic_actions=False) \
                        if hasattr(player, "get_action") else player.get_action_and_value(
                            to_policy(obs["policy"], rung), deterministic_actions=False)[0]
                    obs, rew, term, trunc, extras = env.step(to_env(act, rung))
                    steps += 1
                    tips = env.fingertips()
                    pen = env.pen_pose()[0]
                    d = (tips - pen.unsqueeze(1)).norm(dim=-1)
                    d = torch.where(env.tip_valid > 0, d, torch.full_like(d, 9.0)).amin(1)
                    best_d = torch.where(open_, torch.minimum(best_d, d), best_d)
                    lift = torch.where(open_, torch.maximum(lift, pen[:, 2] - pen0), lift)
                    near += (open_ & (d < 0.02)).float()
                    a_h = act[:, hand_cols].abs().mean(0) if act.shape[-1] >= n_hand else torch.zeros(n_hand, device=dev)
                    act_abs += a_h
                    jpn = (env.robot.data.joint_pos[:, env.perm_lab_to_canon] - env.joint_lower_canon) / \
                          (env.joint_upper_canon - env.joint_lower_canon).clamp_min(1e-6)
                    qn_sum += jpn[:, -n_hand:].mean(0)
                    base_sum += jpn[:, -n_hand:][:, first_slots].mean()
                    nstep += 1
                    if steps in (50, 300, 599):
                        heights[steps] = palm_height()
                    length += open_.float()
                    done = (term | trunc) & open_
                    if done.any():
                        goals = torch.where(done, extras["episode_final"]["successes"].float(), goals)
                        open_ &= ~done
            goals = torch.where(open_, env.successes.float(), goals)
            r = {"goals_per_episode": float(goals.mean()), "frac_with_goal": float((goals > 0).float().mean()),
                 "max_goals": float(goals.max()), "episode_steps": float(length.mean())}
            r["palm_height_mm"] = {k: round(1000 * v) for k, v in heights.items()}
            en_ = env.joint_enabled[0, -n_hand:] > 0.5 if env.joint_enabled.shape[1] >= n_hand else torch.ones(n_hand, dtype=torch.bool, device=dev)
            r.update(closest_tip_pen_mm=float(1000 * best_d.mean()), pen_max_lift_mm=float(1000 * lift.mean()),
                     steps_tip_within_20mm=float(near.mean()),
                     hand_action_abs=float((act_abs / max(nstep, 1))[en_].mean()),
                     hand_joint_pos_frac=float((qn_sum / max(nstep, 1))[en_].mean()),
                     base_joint_pos_frac=float(base_sum / max(nstep, 1)), first_link_scale=fls)
            print(f"[wrap-behav] {run} g{g} rung {rung} first-link x{fls:g}: base joints {r['base_joint_pos_frac']:.2f}, closest tip-pen {r['closest_tip_pen_mm']:.0f} mm, pen lift "
                  f"{r['pen_max_lift_mm']:.0f} mm, steps with a tip <20 mm {r['steps_tip_within_20mm']:.0f}, "
                  f"|hand action| {r['hand_action_abs']:.2f}, joint pos (0 lower, 1 upper) {r['hand_joint_pos_frac']:.2f}",
                  flush=True)
            results[f"{run}_g{g}_{rung}_bias{bias:g}_fl{fls:g}"] = r
            print(f"[wrap] {run} g{g} bias {bias:g} mm rung {rung} first-link x{fls:g}: palm {r['palm_height_mm']} mm | goals/ep {r['goals_per_episode']:.2f}  envs with a goal "
                  f"{100 * r['frac_with_goal']:.0f}%  max {r['max_goals']:.0f}  ep {r['episode_steps']:.0f} steps "
                  f"({time.time() - t0:.0f} s)", flush=True)
            (OUT / f"results{('_' + args.tag) if args.tag else ''}.json").write_text(json.dumps(results, indent=1))
    import os
    sys.stdout.flush()
    os._exit(0)                                         # app.close() can hang for the whole timeout (7 Nov)


if __name__ == "__main__":
    main()
