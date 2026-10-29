"""Where the joint transformer's time goes, against the MLP, on today's exact setup (README.md). Pure torch: no sim.

    .venv_isaacsim/bin/python experiments/29oct_transformer_speed/bench.py [--quick] [--out results.json]

Builds the real networks on the real layout -- hand coarse_s0_000 (20 hand joints + the 6 wrist joints: 21 tokens),
the olympics observation list without joint_origin (471 values), 26 actions -- and measures, per configuration:

  params        total, and split trunk (per-token layers) / heads (per-sample MLPs: wrist, value, the finger head)
  MACs/sample   multiply-accumulates of every Linear for one observation (attention scores excluded: ~5% at 21 tokens)
  inference     ms for one rollout step's policy forward at 24,576 envs (fp16 autocast, no grad)
  train step    ms for forward + backward + Adam at the learner's minibatch (fp16 autocast)
  epoch         the modelled epoch: sim + 16 rollout inferences + mini_epochs x (458,752 / minibatch) train steps,
                with the sim at 1.85 s (hand 000 measured: 212k steps/s x 16 x 24,576) -- and samples/s from it
  memory        peak allocated during the train step

The MLP runs its own learner (minibatch 114,688, 5 mini-epochs); the transformers the 14 Oct learner (16,384, 2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

HAND = "assets/urdf/27oct_grammar_random_hands/coarse_s0_000/coarse_s0_000.spec.json"
OBS_LIST = ["joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets",
            "joint_link_bbox", "joint_lower", "joint_upper", "joint_enabled",
            "keypoints_rel_ee", "keypoints_rel_goal", "object_vel", "palm_extents", "task_onehot"]
N_ENVS, HORIZON, AUG_ROWS = 24576, 16, 458752          # SAPG: 393,216 rows + the leader block's 65,536 copies
SIM_S = 1.85                                           # hand 000: 212k sim steps/s (job 697372's log)
SPACE = {"continuous": {"mu_activation": "None", "sigma_activation": "None", "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": 0}, "fixed_sigma": "fixed"}}


def layout():
    from coevolution.networks.joint_transformer import _WithArmJoints
    from isaacsimenvs.common.obs_layout import build_token_layout
    from isaacsimenvs.common.robots import get_robot_spec
    spec = _WithArmJoints(get_robot_spec(HAND), 6)
    lay = build_token_layout(spec, OBS_LIST, {"task_onehot": 8})
    return lay["obs_dim"], lay["n_joints"], lay


def make_mlp(obs_dim, n_actions, units=(1024, 1024, 512, 512)) -> nn.Module:
    from rl_games.algos_torch.network_builder import A2CBuilder
    b = A2CBuilder()
    b.load({"separate": False, "space": SPACE, "mlp": {"units": list(units), "activation": "elu", "d2rl": False,
                                                       "initializer": {"name": "default"}, "regularizer": {"name": "None"}}})
    return b.build("mlp", actions_num=n_actions, input_shape=(obs_dim,), value_size=1, num_seqs=1, type="simple")


def make_jt(obs_dim, n_actions, **over) -> nn.Module:
    from coevolution.networks.joint_transformer import JointTransformerNet
    params = {"robot_spec": HAND, "obs_list": OBS_LIST, "extra_arm_joints": 6, "extra_field_sizes": {"task_onehot": 8},
              "d_model": 128, "n_layers": 4, "n_heads": 1, "ff_mult": 4, "dropout": 0.0,
              "arm_head_units": [1024, 512], "value_head_units": [512, 256], "mu_head_units": [],
              "mask_ghost_actions": True, "masked_arm_dims": [3, 4, 5], "space": SPACE, **over}
    return JointTransformerNet(params, actions_num=n_actions, input_shape=(obs_dim,), value_size=1, num_seqs=1,
                               type="simple")


def make_jt_mock(obs_dim, n_actions, n_tokens=7, token_dim=100, **over) -> nn.Module:
    """A stand-in for the hierarchical (one token per finger) transformer's TRUNK cost: ``n_tokens`` hand tokens of
    ``token_dim`` features (6 fingers x 5 joint slots x 19 + 5 valid bits = 100, plus a palm-cut token), the same
    global vector, built by stubbing the network's layout. Its head is still one action per token, so the per-slot
    head (5 actions per finger token) is not modelled; that is a Linear(d, 5) in place of Linear(d, 1), negligible."""
    from coevolution.networks.joint_transformer import JointTransformerNet
    global_dim = 71
    mock_obs = n_tokens * token_dim + n_tokens + global_dim
    lay = {"obs_dim": mock_obs, "n_arm": 6, "n_hand": n_tokens, "n_joints": 6 + n_tokens,
           "hand_joint_names": [f"finger{i}" for i in range(n_tokens)],
           "token_columns": [list(range(i * token_dim, (i + 1) * token_dim)) for i in range(n_tokens)],
           "token_dim": token_dim, "token_field_slices": {"all": [0, token_dim]},
           "enabled_columns": list(range(n_tokens * token_dim, n_tokens * token_dim + n_tokens)),
           "global_slices": [[n_tokens * token_dim + n_tokens, mock_obs]], "global_dim": global_dim}
    orig = JointTransformerNet._build_layout
    JointTransformerNet._build_layout = lambda self, d: lay
    try:
        net = make_jt(mock_obs, 6 + n_tokens, **over)
    finally:
        JointTransformerNet._build_layout = orig
    net._mock_layout = lay
    return net


def n_params(net) -> dict:
    tot = sum(p.numel() for p in net.parameters())
    heads = sum(p.numel() for n, p in net.named_parameters()
                if any(k in n for k in ("mu_head", "arm_head", "value_head", "mu.", "value.", "sigma")))
    return {"total": tot, "heads": heads, "trunk": tot - heads}


def macs_per_sample(net, obs) -> int:
    total, hs = 0, []

    def hook(m, inp, _):
        nonlocal total
        total += (inp[0].numel() // inp[0].shape[-1]) * m.in_features * m.out_features
    for m in net.modules():
        if isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(hook))
    with torch.no_grad():
        net({"obs": obs})
    for h in hs:
        h.remove()
    return total // obs.shape[0]


def ms(fn, iters, warmup) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t) / iters


def obs_batch(n, obs_dim, lay, device) -> torch.Tensor:
    x = torch.randn(n, obs_dim, device=device)
    a, b = lay["enabled_columns"][0], lay["enabled_columns"][-1] + 1
    x[:, a:b] = 1.0                                   # every joint real: the attention mask reads this raw
    return x


def measure(name, net, obs_dim, n_actions, lay, mb, mini_epochs, device, quick, dtype=torch.float16) -> dict:
    net = net.to(device)
    iters, warm = (5, 2) if quick else (20, 5)
    x_inf = obs_batch(N_ENVS, obs_dim, lay, device)
    x_tr = obs_batch(mb, obs_dim, lay, device)
    p = n_params(net)
    net.eval()
    macs = macs_per_sample(net, x_inf[:256])

    def inf():
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            net({"obs": x_inf})
    t_inf = ms(inf, iters, warm)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            out = net({"obs": x_tr})
            loss = sum(t.float().square().mean() for t in out if isinstance(t, torch.Tensor))
        loss.backward()
        opt.step()
    torch.cuda.reset_peak_memory_stats()
    t_tr = ms(step, iters, warm)
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    steps = mini_epochs * -(-AUG_ROWS // mb)
    rollout_s, update_s = HORIZON * t_inf / 1e3, steps * t_tr / 1e3
    epoch_s = SIM_S + rollout_s + update_s
    r = {"name": name, "params": p["total"], "params_trunk": p["trunk"], "params_heads": p["heads"], "macs": macs,
         "inference_ms": t_inf, "train_ms": t_tr, "minibatch": mb, "mini_epochs": mini_epochs, "updates": steps,
         "rollout_s": rollout_s, "update_s": update_s, "epoch_s": epoch_s,
         "samples_per_s": N_ENVS * HORIZON / epoch_s, "peak_gb": mem}
    print(f"{name:34s} params {p['total']/1e6:5.2f}M (trunk {p['trunk']/1e6:4.2f}, heads {p['heads']/1e6:4.2f})  "
          f"MACs {macs/1e6:5.2f}M  inf {t_inf:6.1f} ms  train {t_tr:6.1f} ms @{mb}  epoch {epoch_s:5.1f} s "
          f"(roll {rollout_s:4.1f} upd {update_s:4.1f})  {r['samples_per_s']/1e3:5.0f}k/s  mem {mem:4.1f} GB", flush=True)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--only", default="", help="config names separated by |")
    args = ap.parse_args()
    device = torch.device("cuda")
    obs_dim, n_actions, lay = layout()
    print(f"hand 000 layout: obs {obs_dim}, actions {n_actions}, tokens {lay['n_hand']} + 1 global, token dim "
          f"{lay['token_dim']}, global dim {lay['global_dim']}; {torch.cuda.get_device_name()}", flush=True)
    JT = dict(mb=16384, mini_epochs=2)
    configs = [
        ("mlp [1024,1024,512,512]", lambda: make_mlp(obs_dim, n_actions), dict(mb=114688, mini_epochs=5)),
        ("jt d128 L4 (today)", lambda: make_jt(obs_dim, n_actions), JT),
        ("jt d128 L4 compiled", lambda: make_jt(obs_dim, n_actions, compile_net=True), JT),
        ("jt d128 L0 (plumbing floor)", lambda: make_jt(obs_dim, n_actions, n_layers=0), JT),
        ("jt d128 L1", lambda: make_jt(obs_dim, n_actions, n_layers=1), JT),
        ("jt d128 L2", lambda: make_jt(obs_dim, n_actions, n_layers=2), JT),
        ("jt d128 L8", lambda: make_jt(obs_dim, n_actions, n_layers=8), JT),
        ("jt d32 L4 (fast config width)", lambda: make_jt(obs_dim, n_actions, d_model=32), JT),
        ("jt d64 L4", lambda: make_jt(obs_dim, n_actions, d_model=64), JT),
        ("jt d256 L4", lambda: make_jt(obs_dim, n_actions, d_model=256), JT),
        ("jt d512 L2", lambda: make_jt(obs_dim, n_actions, d_model=512, n_layers=2), JT),
        ("jt d128 L4 h4", lambda: make_jt(obs_dim, n_actions, n_heads=4), JT),
        ("jt d128 L4 h2", lambda: make_jt(obs_dim, n_actions, n_heads=2), JT),
        ("jt d128 L4 h4 compiled", lambda: make_jt(obs_dim, n_actions, n_heads=4, compile_net=True), JT),
        ("jt d128 L4 sdpa", lambda: make_jt(obs_dim, n_actions, attention="sdpa"), JT),
        ("jt d128 L4 sdpa compiled noaffine", lambda: make_jt(obs_dim, n_actions, attention="sdpa", compile_net=True,
                                                             affine_norm=False), JT),
        ("jt d256 L2 sdpa compiled noaffine", lambda: make_jt(obs_dim, n_actions, d_model=256, n_layers=2,
                                                             attention="sdpa", compile_net=True, affine_norm=False), JT),
        ("mock 7 finger tokens: d128 L4", lambda: make_jt_mock(obs_dim, n_actions), JT),
        ("mock 7 finger tokens: d256 L4", lambda: make_jt_mock(obs_dim, n_actions, d_model=256), JT),
        ("mock 7 finger tokens: d256 L4 sdpa compiled noaffine",
         lambda: make_jt_mock(obs_dim, n_actions, d_model=256, attention="sdpa", compile_net=True, affine_norm=False), JT),
        ("mock 7 finger tokens: d512 L4 sdpa compiled noaffine",
         lambda: make_jt_mock(obs_dim, n_actions, d_model=512, attention="sdpa", compile_net=True, affine_norm=False), JT),
        # MLP-parity candidates on the per-finger decomposition: ~2.3M params, >= 2 attention layers, as fast as it goes
        *[(f"parity: mock 7 tokens d{d} L{L} ff{ff} heads{'-big' if big else ''} sdpa compiled noaffine",
           (lambda d=d, L=L, ff=ff, big=big: make_jt_mock(
               obs_dim, n_actions, d_model=d, n_layers=L, ff_mult=ff, attention="sdpa", compile_net=True,
               affine_norm=False, **({"arm_head_units": [2048, 1024], "value_head_units": [1024, 512]} if big else {}))),
           JT)
          for d, L, ff, big in ((256, 2, 2, False), (256, 2, 4, False), (256, 3, 2, False), (192, 2, 4, False),
                                (384, 2, 2, False), (128, 2, 4, True), (128, 3, 4, False))],
        # parity candidates again WITHOUT sdpa: the fused attention breaks the compiled graph (2x the train step)
        *[(f"parity2: mock 7 tokens d{d} L{L} ff{ff} heads{'-big' if big else ''} compiled noaffine",
           (lambda d=d, L=L, ff=ff, big=big: make_jt_mock(
               obs_dim, n_actions, d_model=d, n_layers=L, ff_mult=ff, compile_net=True, affine_norm=False,
               **({"arm_head_units": [2048, 1024], "value_head_units": [1024, 512]} if big else {}))),
           JT)
          for d, L, ff, big in ((128, 2, 4, True), (128, 3, 4, False), (192, 2, 4, False), (256, 2, 2, False),
                                (256, 2, 4, False), (256, 3, 2, False), (256, 2, 4, True), (384, 2, 2, False))],
        # the floor with 56 updates per epoch: 6 finger tokens (the palm cut folded into the global token) + global
        *[(f"floor: mock 6 tokens d{d} L{L} heads{'-big' if big else ''} compiled noaffine",
           (lambda d=d, L=L, big=big: make_jt_mock(
               obs_dim, n_actions, n_tokens=6, d_model=d, n_layers=L, compile_net=True, affine_norm=False,
               **({"arm_head_units": [2048, 1024], "value_head_units": [1024, 512]} if big else {}))),
           JT)
          for d, L, big in ((128, 0, True), (128, 2, True), (128, 2, False), (64, 2, True), (256, 2, False))],
        # does a bigger minibatch cut the per-row cost on the per-finger trunk (small matmuls at 16,384 x 7 rows)?
        *[(f"minibatch {mb}: mock 7 tokens d{d} L2{' heads-big' if big else ''} compiled noaffine",
           (lambda d=d, big=big: make_jt_mock(
               obs_dim, n_actions, d_model=d, n_layers=2, compile_net=True, affine_norm=False,
               **({"arm_head_units": [2048, 1024], "value_head_units": [1024, 512]} if big else {}))),
           dict(mb=mb, mini_epochs=2))
          for d, big in ((256, False), (128, True)) for mb in (57344, 114688)],
        # the fastest two-layer per-finger trunk with default heads
        *[(f"fast: mock 6 tokens d{d} L2 ff{ff} compiled noaffine",
           (lambda d=d, ff=ff: make_jt_mock(obs_dim, n_actions, n_tokens=6, d_model=d, n_layers=2, ff_mult=ff,
                                             compile_net=True, affine_norm=False)), JT)
          for d, ff in ((64, 4), (64, 2), (128, 2), (96, 4))],
        # THE PICK (29 Oct, the user's call on heads): per-finger, width 128, 2 layers, ff x4, 4 heads, default heads
        ("pick: mock 6 tokens d128 L2 ff4 h4 compiled noaffine",
         lambda: make_jt_mock(obs_dim, n_actions, n_tokens=6, d_model=128, n_layers=2, n_heads=4, compile_net=True,
                              affine_norm=False), JT),
        ("pick: mock 6 tokens d128 L2 ff4 h4 eager", lambda: make_jt_mock(obs_dim, n_actions, n_tokens=6, d_model=128,
                                                                        n_layers=2, n_heads=4), JT),
        # the compile anomaly at width 256 on finger tokens: which of compile / sdpa / no-affine costs?
        ("anomaly: mock d256 L2 ff4 eager", lambda: make_jt_mock(obs_dim, n_actions, d_model=256, n_layers=2), JT),
        ("anomaly: mock d256 L2 ff4 compiled", lambda: make_jt_mock(obs_dim, n_actions, d_model=256, n_layers=2,
                                                                   compile_net=True), JT),
        ("anomaly: mock d256 L2 ff4 compiled sdpa", lambda: make_jt_mock(obs_dim, n_actions, d_model=256, n_layers=2,
                                                                        compile_net=True, attention="sdpa"), JT),
        ("anomaly: mock d256 L2 ff4 compiled noaffine", lambda: make_jt_mock(obs_dim, n_actions, d_model=256,
                                                                            n_layers=2, compile_net=True,
                                                                            affine_norm=False), JT),
        ("anomaly: mock d128 L2 ff4 heads-big eager", lambda: make_jt_mock(
            obs_dim, n_actions, d_model=128, n_layers=2, arm_head_units=[2048, 1024], value_head_units=[1024, 512]), JT),
        ("jt d128 L4 compiled max-autotune", lambda: make_jt(obs_dim, n_actions, compile_net=True,
                                                            compile_mode="max-autotune"), JT),
        ("jt d128 L4 compiled reduce-overhead", lambda: make_jt(obs_dim, n_actions, compile_net=True,
                                                               compile_mode="reduce-overhead"), JT),
        ("jt d256 L4 h4", lambda: make_jt(obs_dim, n_actions, d_model=256, n_heads=4), JT),
        ("jt d256 L2 h4", lambda: make_jt(obs_dim, n_actions, d_model=256, n_layers=2, n_heads=4), JT),
        ("jt d256 L2 h4 compiled noaffine", lambda: make_jt(obs_dim, n_actions, d_model=256, n_layers=2, n_heads=4,
                                                           compile_net=True, affine_norm=False), JT),
        ("jt candidate C: d256 L2 h4 big heads compiled noaffine",
         lambda: make_jt(obs_dim, n_actions, d_model=256, n_layers=2, n_heads=4, arm_head_units=[2048, 1024],
                         value_head_units=[2048, 1024], compile_net=True, affine_norm=False), JT),
        ("jt d128 L4 ff2", lambda: make_jt(obs_dim, n_actions, ff_mult=2), JT),
        ("jt d128 L4 parallel block", lambda: make_jt(obs_dim, n_actions, parallel_block=True), JT),
        ("jt d128 L4 no affine norm", lambda: make_jt(obs_dim, n_actions, affine_norm=False), JT),
        ("jt d128 L4 small heads", lambda: make_jt(obs_dim, n_actions, arm_head_units=[256, 128],
                                                   value_head_units=[256, 128]), JT),
        ("jt d128 L4 big heads", lambda: make_jt(obs_dim, n_actions, arm_head_units=[2048, 1024],
                                                 value_head_units=[2048, 1024]), JT),
        ("jt d128 L4 mu head [256]", lambda: make_jt(obs_dim, n_actions, mu_head_units=[256]), JT),
        ("jt d128 L4 compiled+parallel+noaffine", lambda: make_jt(obs_dim, n_actions, compile_net=True,
                                                                  parallel_block=True, affine_norm=False), JT),
        ("jt d128 L4 mb 57344", lambda: make_jt(obs_dim, n_actions), dict(mb=57344, mini_epochs=2)),
        ("jt d256 L4 compiled", lambda: make_jt(obs_dim, n_actions, d_model=256, compile_net=True), JT),
        # the candidates the first sweep points to: parameters in the per-sample heads and the per-joint head,
        # the plumbing cuts, a wide-shallow trunk
        ("jt d128 L4 mu head [1024,512]", lambda: make_jt(obs_dim, n_actions, mu_head_units=[1024, 512]), JT),
        ("jt d256 L2", lambda: make_jt(obs_dim, n_actions, d_model=256, n_layers=2), JT),
        ("jt d128 L4 ff2 compiled noaffine", lambda: make_jt(obs_dim, n_actions, ff_mult=2, compile_net=True,
                                                            affine_norm=False), JT),
        ("jt candidate: d128 L4 big heads mu[256] compiled noaffine",
         lambda: make_jt(obs_dim, n_actions, arm_head_units=[2048, 1024], value_head_units=[2048, 1024],
                         mu_head_units=[256], compile_net=True, affine_norm=False), JT),
        ("jt candidate: d256 L2 big heads compiled noaffine",
         lambda: make_jt(obs_dim, n_actions, d_model=256, n_layers=2, arm_head_units=[2048, 1024],
                         value_head_units=[2048, 1024], compile_net=True, affine_norm=False), JT),
        # the per-finger network on the REAL layout (token_mode finger, 29 Oct plumbing): the pick and its fallbacks
        ("finger: real d128 L2 h4 compiled noaffine (the pick)",
         lambda: make_jt(obs_dim, n_actions, token_mode="finger", d_model=128, n_layers=2, n_heads=4,
                         decoder_units=[256], compile_net=True, affine_norm=False), JT),
        ("finger: real d128 L2 h4 eager noaffine",
         lambda: make_jt(obs_dim, n_actions, token_mode="finger", d_model=128, n_layers=2, n_heads=4,
                         decoder_units=[256], affine_norm=False), JT),
        ("finger: real d64 L2 h4 compiled noaffine",
         lambda: make_jt(obs_dim, n_actions, token_mode="finger", d_model=64, n_layers=2, n_heads=4,
                         decoder_units=[256], compile_net=True, affine_norm=False), JT),
        ("finger: real d128 L3 h4 compiled noaffine",
         lambda: make_jt(obs_dim, n_actions, token_mode="finger", d_model=128, n_layers=3, n_heads=4,
                         decoder_units=[256], compile_net=True, affine_norm=False), JT),
    ]
    only = {s.strip() for s in args.only.split("|") if s.strip()}
    rows = []
    for name, build, learner in configs:
        if only and name not in only:
            continue
        try:
            torch.manual_seed(0)
            net = build()
            mock = getattr(net, "_mock_layout", None)
            rows.append(measure(name, net, mock["obs_dim"] if mock else obs_dim,
                                mock["n_joints"] if mock else n_actions, mock or lay, learner["mb"],
                                learner["mini_epochs"], device, args.quick))
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            print(f"{name:34s} OOM at minibatch {learner['mb']}", flush=True)
            rows.append({"name": name, "oom": True, "minibatch": learner["mb"]})
            torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps({"obs_dim": obs_dim, "n_actions": n_actions, "tokens": lay["n_hand"] + 1,
                                          "sim_s": SIM_S, "rows": rows}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
