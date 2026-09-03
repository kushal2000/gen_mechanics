"""Benchmark policy forward/backward latency without booting Isaac Sim.

This isolates the train-time network cost from PhysX and environment creation.
The batch defaults to the 4k rollout size; use ``--batch_size 16384`` to match
the current SAPG minibatch.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

from rl_games.algos_torch.network_builder import A2CBuilder


OBS_LIST = [
    "joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel",
    "prev_action_targets", "joint_link_bbox", "joint_lower", "joint_upper",
    "joint_enabled", "object_keypoints_rel_joint", "hand_scale", "palm_pos",
    "palm_rot", "object_rot", "keypoints_rel_palm", "keypoints_rel_goal",
    "object_scales",
]


def make_mlp() -> torch.nn.Module:
    params = {
        "separate": False,
        "space": {"continuous": {
            "mu_activation": "None", "sigma_activation": "None",
            "mu_init": {"name": "default"},
            "sigma_init": {"name": "const_initializer", "val": 0},
            "fixed_sigma": "fixed",
        }},
        "mlp": {
            "units": [1024, 1024, 512, 512], "activation": "elu",
            "d2rl": False, "initializer": {"name": "default"},
            "regularizer": {"name": "None"},
        },
    }
    builder = A2CBuilder()
    builder.load(params)
    return builder.build(
        "profile", actions_num=29, input_shape=(778,), value_size=1,
        num_seqs=1, type="simple",
    )


def make_transformer(n_layers: int) -> torch.nn.Module:
    from coevolution.networks.joint_transformer import JointTransformerNet
    params = {
        "robot_spec": "sharpa_iiwa14", "obs_list": OBS_LIST,
        "d_model": 128, "n_layers": n_layers, "n_heads": 4,
        "ff_mult": 4, "dropout": 0.0,
        "arm_head_units": [1024, 512], "value_head_units": [512, 256],
        "space": {"continuous": {
            "mu_activation": "None", "sigma_activation": "None",
            "sigma_init": {"name": "const_initializer", "val": 0},
            "fixed_sigma": "fixed",
        }},
    }
    return JointTransformerNet(
        params, actions_num=29, input_shape=(778,), value_size=1,
        num_seqs=1, type="simple",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    autocast = torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=args.amp
    )
    # The normal obs_utils package eagerly imports Isaac Lab-backed builders.
    # The layout module itself is Kit-free, so expose just that subpackage for
    # this network-only benchmark.
    repo_root = pathlib.Path(__file__).parents[2]
    for pkg_name, relative_path in (
        ("isaacsimenvs.pose_reaching_6d.obs_utils", "isaacsimenvs/pose_reaching_6d/obs_utils"),
        ("isaacsimenvs.pose_reaching_6d.scene_utils", "isaacsimenvs/pose_reaching_6d/scene_utils"),
    ):
        if pkg_name not in sys.modules:
            pkg = type(sys)(pkg_name)
            pkg.__path__ = [str(repo_root / relative_path)]
            sys.modules[pkg_name] = pkg
    obs = torch.randn(args.batch_size, 778, device=device)
    nets = [("MLP", make_mlp())]
    nets += [(f"Transformer-L{layers}", make_transformer(layers))
             for layers in (1, 4)]

    def run(net, backward: bool) -> float:
        net.to(device).train()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

        def iteration():
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                mu, _, value, _ = net({"obs": obs})
                loss = mu.square().mean() + value.square().mean()
            if backward:
                loss.backward()
                optimizer.step()
            return loss

        for _ in range(args.warmup):
            iteration()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            iteration()
        torch.cuda.synchronize(device)
        return 1e3 * (time.perf_counter() - start) / args.steps

    print(
        f"batch={args.batch_size} amp={args.amp} "
        "(latencies include optimizer.step for backward rows)", flush=True,
    )
    for name, net in nets:
        forward_ms = run(net, backward=False)
        backward_ms = run(net, backward=True)
        print(
            f"{name:<16} forward={forward_ms:8.3f} ms  "
            f"forward+backward={backward_ms:8.3f} ms  "
            f"train_SPS={args.batch_size * 1000 / backward_ms:,.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
