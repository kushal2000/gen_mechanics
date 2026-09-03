"""Audit the flat-observation -> joint-token gather, for correctness and cost.

Two independent questions about the same three lines of ``_trunk``:

    tokens = self.token_proj(env_obs[:, self.token_gather])   # (B, 22, 32)
    glob   = env_obs[:, self.global_index]                    # (B, 74)

CORRECTNESS. ``build_token_layout`` computes the gather indices from the field
list alone, so a drift between the env's field order and the layout table would
silently feed every joint the wrong 32 numbers -- the network would still train,
just on scrambled tokens. This checks each of the 32 slots of each of the 22
tokens against the flat vector it is supposed to have come from, field by
field, rather than spot-checking one field.

COST. Advanced indexing on dim 1 is a scattered gather forward and a
scatter-add backward through a (B, 778) zero tensor. This times it against the
rest of the network so the share is known rather than assumed.

    .venv_isaacsim/bin/python \
      experiments/decentralized_control/profile_token_gather.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from profile_policy_train_step import (  # noqa: E402
    OBS_LIST, make_transformer, stub_kit_free_packages,
)

stub_kit_free_packages()

from isaacsimenvs.pose_reaching_6d.obs_utils.layout import (  # noqa: E402
    HAND_TOKEN_FIELDS, JOINT_WIDTH_FIELDS, build_token_layout, field_offsets,
)
from isaacsimenvs.pose_reaching_6d.scene_utils.robots import (  # noqa: E402
    get_robot_spec,
)


def check_correctness(device) -> None:
    """Rebuild the expected token from the field table and compare, exactly."""
    spec = get_robot_spec("sharpa_iiwa14")
    layout = build_token_layout(spec, OBS_LIST)
    offsets = field_offsets(OBS_LIST, spec)
    n_arm, n_hand = layout["n_arm"], layout["n_hand"]
    obs_dim = layout["obs_dim"]

    # Each column holds its own index, so a mis-gather shows up as a wrong
    # NUMBER, not as a wrong-looking float.
    flat = torch.arange(obs_dim, dtype=torch.float32, device=device).unsqueeze(0)
    gather = torch.tensor(layout["token_columns"], dtype=torch.long, device=device)
    tokens = flat[:, gather][0]  # (22, 32)

    print(f"layout: obs_dim={obs_dim} n_arm={n_arm} n_hand={n_hand} "
          f"token_dim={layout['token_dim']} global_dim={layout['global_dim']}",
          flush=True)

    slot = 0
    problems = []
    for field in JOINT_WIDTH_FIELDS:
        start, _ = offsets[field]
        for joint in range(n_hand):
            want = start + n_arm + joint
            got = int(tokens[joint, slot].item())
            if got != want:
                problems.append(f"{field} joint{joint} slot{slot}: {got} != {want}")
        slot += 1
    for field, stride in HAND_TOKEN_FIELDS.items():
        start, _ = offsets[field]
        for joint in range(n_hand):
            for k in range(stride):
                want = start + joint * stride + k
                got = int(tokens[joint, slot + k].item())
                if got != want:
                    problems.append(
                        f"{field} joint{joint} slot{slot + k}: {got} != {want}")
        slot += stride

    # Every column must be consumed exactly once, by a token or by the global.
    global_index = [i for s, e in layout["global_slices"] for i in range(s, e)]
    used = sorted(int(v) for v in gather.flatten().tolist()) + sorted(global_index)
    missing = sorted(set(range(obs_dim)) - set(used))
    duplicated = sorted({v for v in used if used.count(v) > 1})

    print(f"  token slots checked : {slot * n_hand}", flush=True)
    print(f"  mis-gathered slots  : {len(problems)}", flush=True)
    for line in problems[:10]:
        print(f"    {line}", flush=True)
    print(f"  columns never used  : {len(missing)} {missing[:12]}", flush=True)
    print(f"  columns used twice  : {len(duplicated)} {duplicated[:12]}", flush=True)
    verdict = "PASS" if not problems and not duplicated else "FAIL"
    print(f"  -> token gather {verdict}", flush=True)


def profile_cost(device, batch: int, iters: int, warmup: int) -> None:
    net = make_transformer(1).to(device).train()
    obs = torch.randn(batch, 778, device=device, requires_grad=True)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    gather, gidx = net.token_gather, net.global_index

    def timed(label, fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(device)
        ms = 1e3 * (time.perf_counter() - start) / iters
        print(f"  {label:<38} {ms:8.3f} ms", flush=True)
        return ms

    def fwd_bwd(fn):
        def run():
            out = fn()
            out.sum().backward()
            obs.grad = None
        return run

    print(f"\ncost at batch={batch:,} (fp16 autocast, forward+backward)",
          flush=True)
    with autocast:
        total = timed("full network fwd+bwd", fwd_bwd(
            lambda: net({"obs": obs})[0]))
        gth = timed("  token gather obs[:, (22,32)]", fwd_bwd(
            lambda: obs[:, gather]))
        gl = timed("  global gather obs[:, (74,)]", fwd_bwd(
            lambda: obs[:, gidx]))
        proj = timed("  + token_proj on the gather", fwd_bwd(
            lambda: net.token_proj(obs[:, gather])))
        hidden = torch.randn(batch, 23, net.d_model, device=device,
                             requires_grad=True)

        def layer_step():
            out = net.layers[0](hidden)
            out.sum().backward()
            hidden.grad = None

        layer = timed("  one encoder layer alone", layer_step)
    share = 100.0 * (gth + gl) / max(total, 1e-9)
    print(f"\n  gathers are {share:.1f}% of the full network fwd+bwd", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--skip_cost", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_correctness(device)
    if not args.skip_cost and device.type == "cuda":
        profile_cost(device, args.batch_size, args.iters, args.warmup)


if __name__ == "__main__":
    main()
