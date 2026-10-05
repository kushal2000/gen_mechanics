r"""GET-Zero-style distillation on a population env: per-hand MLP experts into
the joint-token transformer by DAgger (``--mode train``), and deterministic
evaluation of the student, the experts or zero actions (``--mode eval``).
Kit entry point: AppLauncher boots before any ``isaaclab`` import, as in
``coevolution/train.py``. The method and its adaptations are in
``dagger.py``; the hyperparameters in ``InHandHoraTokenDistill.yaml``
(``params.distill``, overridable as ``agent.params.distill.<key>=...``).

Train (resumes from ``OUT/student_last.pth`` when it exists):

    OMNI_KIT_ACCEPT_EULA=YES timeout -k 30 6000 .venv_isaacsim/bin/python3 -m \
        isaacsimenvs.inhand_reorient.distill.run --mode train --out OUT --experts EXPERTS.json \
        --minutes 90 env.task_profile=hora env.assets.hand_population=TRAIN.json \
        env.scene.num_envs=4096 env.anyrotate.grasp_cache=CACHE.npz \
        env.anyrotate.grasp_cache_generate=false env.anyrotate.grasp_per_design=4000 hydra.run.dir=OUT

Evaluate (each env's first episode after a full reset, ``--eval-rounds``
resets; deterministic actions):

    ... -m isaacsimenvs.inhand_reorient.distill.run --mode eval --out EVAL \
        --policies zero,expert,student --student OUT/student_last.pth --experts EXPERTS.json ...

``EXPERTS.json`` maps a design's ``source`` (e.g. ``projected:allegro_right``)
to its expert's rl_games checkpoint. The env runs with ``hora.token_obs``
and, whenever experts are used, ``hora.teacher_obs`` (set here).

Outputs in ``OUT``: ``distill_log.jsonl`` (rows ``train``: loss overall and
per design, beta, samples, the rollout's per-design holding time and
rotations; rows ``eval``: the student's deterministic per-design results),
``student_last.pth`` (rl_games ``{0: weights}`` plus the DAgger state),
``student_it<N>.pth`` at every evaluation, ``eval.json`` (eval mode).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


EXPERT_AGENT = "rl_games_anyrotate_ppo_cfg_entry_point"  # the per-hand MLP experts' train YAML (HORA's PPO)


def main() -> None:
    import argparse

    # design_scoring's windows close only when this script says so.
    os.environ.setdefault("GENMECH_DESIGN_SCORE_WRITE_EVERY", "1000000000")
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="GenMech-InHandReorient-Direct-v0")
    ap.add_argument("--agent", default="rl_games_hora_token_distill_cfg_entry_point")
    ap.add_argument("--mode", choices=("train", "eval"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", default=None, help="JSON: design source -> expert rl_games checkpoint")
    ap.add_argument("--student", default=None,
                    help="eval: student checkpoint(s), comma-separated; train: initial weights (fresh run)")
    ap.add_argument("--policies", default="student", help="eval: comma list of zero, expert, student")
    ap.add_argument("--minutes", type=float, default=60.0, help="train: DAgger wall-clock budget, resumes included")
    ap.add_argument("--max-iters", type=int, default=0, help="train: stop after this many iterations (0: no cap)")
    ap.add_argument("--eval-rounds", type=int, default=2, help="final and eval-mode evaluation: full resets")
    AppLauncher.add_app_launcher_args(ap)
    args, hydra_args = ap.parse_known_args()
    args.headless = True
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    import yaml

    import isaacsimenvs  # noqa: F401  registers the task
    import coevolution.networks  # noqa: F401
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml
    from isaacsimenvs.inhand_reorient import design_scoring
    from isaacsimenvs.inhand_reorient import token_layout as tl
    from isaacsimenvs.inhand_reorient import token_policy  # noqa: F401  registers inhand_joint_transformer
    from isaacsimenvs.inhand_reorient.distill import dagger as dg
    from isaacsimenvs.inhand_reorient.tools.zero_action_check import summarise as summarise_window

    torch.backends.cuda.matmul.allow_tf32 = True
    policies = [p for p in args.policies.split(",") if p] if args.mode == "eval" else ["expert", "student"]
    for p in policies:
        if p not in ("zero", "expert", "student"):
            raise SystemExit(f"--policies: unknown policy {p!r}")
    need_experts = "expert" in policies
    if need_experts and not args.experts:
        raise SystemExit("--experts is required to train or to evaluate experts")
    if args.mode == "eval" and "student" in policies and not args.student:
        raise SystemExit("--student is required to evaluate the student")

    @hydra_task_config_with_yaml(args.task, args.agent)
    def run(env_cfg, agent_cfg: dict) -> None:
        if env_cfg.task_profile != "hora":
            raise SystemExit("distill.run needs env.task_profile=hora")
        params = agent_cfg["params"]
        # eval mode also reads other token configs (e.g. an RL-trained transformer's own YAML via --agent)
        dcfg = params["distill"] if args.mode == "train" else (params.get("distill") or {})
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        env_cfg.hora.token_obs = True
        env_cfg.hora.teacher_obs = bool(need_experts)
        seed = int(params.get("seed", 42))  # agent.params.seed: the student's init, the env's randomness, the DAgger draws
        env_cfg.seed = seed
        dg.seed_everything(seed)
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        dev = inner.device
        obs, _ = env.reset()
        clip = float(params["env"]["clip_observations"])
        n_act = int(inner.cfg.action_space)
        n_designs = design_scoring._n_designs(inner)
        sources = [design_scoring._design_source(inner, d) for d in range(n_designs)]
        design = inner.scene_record["design_idx"].to(dev).long()
        ep_max = float(inner.cfg.episode_length_s)
        s_dim = int(obs["policy"].shape[1])
        print(f"[distill] {n_designs} designs {sources}, {inner.num_envs} envs, student obs {s_dim}, "
              f"actions {n_act}", flush=True)

        student = dg.build_rlg_model(params, s_dim, n_act).to(dev)
        stu_pol = dg.MeanPolicy(student, obs_clip=clip)
        bank = None
        if need_experts:
            mapping = json.loads(Path(args.experts).read_text())
            paths = dg.resolve_experts(sources, mapping)
            eyaml = gym.spec(args.task.split(":")[-1]).kwargs[dcfg.get("expert_agent", EXPERT_AGENT)]
            eparams = yaml.safe_load(Path(eyaml).read_text())["params"]
            t_dim = int(obs["teacher_obs"].shape[1])
            uniq = sorted({p for p in paths if p})
            experts = [dg.load_expert(eparams, p, t_dim, n_act, dev) for p in uniq]
            bank = dg.ExpertBank(experts, design, [uniq.index(p) if p else None for p in paths])
            print(f"[distill] experts (teacher obs {t_dim}): "
                  + ", ".join(f"{s} <- {p}" for s, p in zip(sources, paths)), flush=True)
            if args.mode == "train" and bank.missing:
                raise SystemExit(f"no expert for {[sources[d] for d in bank.missing]}")

        def act_fn(name):
            if name == "zero":
                return lambda o: torch.zeros(inner.num_envs, n_act, device=dev)
            if name == "expert":
                return lambda o: bank.act(o["teacher_obs"], n_act)
            return lambda o: stu_pol(o["policy"])

        def evaluate(name: str, rounds: int):
            was_training = student.training
            student.eval()
            fn, tot, o = act_fn(name), None, None
            for _ in range(int(rounds)):
                o, _ = env.reset()
                tr = dg.FirstEpisodeTracker(design, n_designs)
                for _ in range(int(inner.max_episode_length) + 2):
                    with torch.no_grad():
                        a = fn(o)
                    o, _, term, trunc, extras = env.step(a)
                    ep = extras["episode_final"]
                    tr.update(term | trunc, ep["ttt_s"], ep["rotation_rad"])
                    if tr.all_done:
                        break
                tot = dg.add_sums(tot, tr.sums(ep_max))
            student.train(was_training)
            return dg.summarise(tot, sources), o

        if args.mode == "eval":
            students = [s for s in (args.student or "").split(",") if s]
            res = {"populations": sources, "num_envs": int(inner.num_envs), "rounds": int(args.eval_rounds),
                   "student": students[0] if students else None, "experts": args.experts}
            for name in policies:
                for k, ck in enumerate(students if name == "student" else [None]):
                    if ck is not None:
                        dg.load_checkpoint(ck, student)
                    t = time.time()
                    r, _ = evaluate(name, args.eval_rounds)
                    if name == "student" and len(students) > 1:  # every checkpoint under its path, the first also as "student"
                        res.setdefault("students", {})[ck] = r
                    if k == 0:
                        res[name] = r
                    print(f"[distill] eval {name} {ck or ''} ({time.time() - t:.0f} s): {json.dumps(r)}", flush=True)
                    (out / "eval.json").write_text(json.dumps(res, indent=1))
            return

        # ---------------------------------------------------------------- train
        lr, wd = float(dcfg["learning_rate"]), float(dcfg.get("weight_decay", 0.0))
        opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=wd)
        last = out / "student_last.pth"
        state = {"iteration": 0, "samples": 0, "grad_steps": 0, "train_s": 0.0, "last_eval_s": 0.0}
        if last.exists():
            state.update(dg.load_checkpoint(last, student, opt))
            print(f"[distill] resumed {last} at iteration {state['iteration']} ({state['train_s']:.0f} s)", flush=True)
        elif args.student:
            dg.load_checkpoint(args.student, student)
            print(f"[distill] initial weights from {args.student}", flush=True)
        data = dg.AggregatedDataset(int(dcfg["buffer_capacity"]), s_dim, n_act, dev)
        T, G, B = int(dcfg["rollout_steps"]), int(dcfg["grad_steps"]), int(dcfg["minibatch"])
        warm, decay, bmin = int(dcfg["warmup_iters"]), int(dcfg["decay_iters"]), float(dcfg.get("beta_min", 0.0))
        log_every, ckpt_every = int(dcfg["log_every_iters"]), int(dcfg["ckpt_every_iters"])
        eval_every, eval_rounds_train = float(dcfg["eval_every_s"]), int(dcfg["eval_rounds"])
        gnorm = float(dcfg["grad_norm"])
        log = out / "distill_log.jsonl"
        budget = float(args.minutes) * 60.0
        t0, base = time.time(), float(state["train_s"])
        elapsed = lambda: base + time.time() - t0  # noqa: E731
        design_scoring.maybe_write(inner)  # open the first rollout window
        obs, _ = env.reset()
        loss_acc, n_acc, used_acc = 0.0, 0, 0.0
        per_design_loss = torch.zeros(n_designs, device=dev)

        def save(path):
            state["train_s"] = elapsed()
            dg.save_checkpoint(path, student, opt, state)

        while elapsed() < budget and not (args.max_iters and state["iteration"] >= args.max_iters):
            it = int(state["iteration"])
            beta = dg.beta_at(it, warm, decay, bmin)
            student.eval()
            for _ in range(T):
                s_obs = obs["policy"].clamp(-clip, clip)
                a_exp = bank.act(obs["teacher_obs"], n_act)
                data.add(s_obs, a_exp, design)
                if beta < 1.0:
                    with torch.no_grad():
                        a_stu = stu_pol(s_obs)
                    act, used = dg.mix_actions(a_exp, a_stu, beta)
                    used_acc += float(used.float().mean())
                else:
                    act = a_exp
                    used_acc += 1.0
                obs, _, _, _, _ = env.step(act)
            student.train()
            for _ in range(G):
                o, y, d = data.sample(B)
                mu = student.a2c_network({"obs": student.norm_obs(o)})[0]
                loss, per = dg.masked_action_mse(mu, y, dg.valid_from_tokens(o, n_act, tl.TOKEN_DIM, tl.ENABLED_COL))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), gnorm)
                opt.step()
                loss_acc += float(loss.detach())
                n_acc += 1
            per_design_loss, _ = dg.per_design_mean(per.detach(), d, n_designs)
            state["iteration"] = it + 1
            state["samples"] = int(state["samples"]) + T * inner.num_envs
            state["grad_steps"] = int(state["grad_steps"]) + G
            if (it + 1) % log_every == 0:
                window = summarise_window(design_scoring.read_snapshot(inner))
                design_scoring.maybe_write(inner)
                row = {"type": "train", "iteration": it + 1, "t_s": round(elapsed(), 1), "beta": beta,
                       "expert_share": used_acc / (log_every * T), "loss": loss_acc / max(n_acc, 1),
                       "loss_by_design": {s: float(v) for s, v in zip(sources, per_design_loss)},
                       "samples": int(state["samples"]), "dataset": len(data), "grad_steps": int(state["grad_steps"]),
                       "rollout": window}
                dg.append_jsonl(log, row)
                print(f"[distill] it {it + 1} t {row['t_s']:.0f}s beta {beta:.2f} loss {row['loss']:.5f} "
                      f"samples {row['samples']} rollout ttt "
                      + " ".join(f"{v.get('ttt_mean_s') or 0:.1f}" for v in window.values()), flush=True)
                loss_acc, n_acc, used_acc = 0.0, 0, 0.0
            if (it + 1) % ckpt_every == 0:
                save(last)
            if elapsed() - float(state["last_eval_s"]) >= eval_every:
                res, obs = evaluate("student", eval_rounds_train)
                design_scoring.maybe_write(inner)  # the evaluation's episodes leave the rollout window
                state["last_eval_s"] = elapsed()
                dg.append_jsonl(log, {"type": "eval", "iteration": it + 1, "t_s": round(elapsed(), 1),
                                      "beta": beta, "samples": int(state["samples"]), "student": res})
                save(out / f"student_it{it + 1}.pth")
                save(last)
                print(f"[distill] eval it {it + 1}: " + json.dumps(res), flush=True)

        save(last)
        res, _ = evaluate("student", args.eval_rounds)
        dg.append_jsonl(log, {"type": "eval", "final": True, "iteration": int(state["iteration"]),
                              "t_s": round(elapsed(), 1), "samples": int(state["samples"]), "student": res})
        (out / "eval_final.json").write_text(json.dumps({"student": res, "iteration": int(state["iteration"])},
                                                        indent=1))
        print(f"[distill] done: {state['iteration']} iterations, {state['samples']} samples; final eval "
              + json.dumps(res), flush=True)

    run()
    sys.stdout.flush()
    sys.stderr.flush()
    del app
    os._exit(0)


if __name__ == "__main__":
    main()
