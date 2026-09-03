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


def make_transformer(n_layers: int, central_value: bool = False,
                     mu_head_units=None) -> torch.nn.Module:
    """The actor, or the asymmetric critic that SAPG trains alongside it.

    The critic is a second, independently configured joint_transformer reading
    the 800-wide state; it has no action heads but is otherwise the same net,
    and it takes the same number of gradient steps per epoch as the actor.
    """
    from coevolution.networks.joint_transformer import JointTransformerNet
    field_key = "state_list" if central_value else "obs_list"
    params = {
        "robot_spec": "sharpa_iiwa14", field_key: OBS_LIST,
        "central_value": central_value,
        "d_model": 128, "n_layers": n_layers, "n_heads": 4,
        "ff_mult": 4, "dropout": 0.0,
        "arm_head_units": [1024, 512], "value_head_units": [512, 256],
        "mu_head_units": list(mu_head_units or []),
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


def stub_kit_free_packages() -> None:
    """Expose layout.py / robots.py without importing their Kit-backed parents.

    ``joint_transformer`` reaches into ``isaacsimenvs...obs_utils.layout`` and
    ``...scene_utils.robots``, both of which are pure Python -- but importing
    the packages that contain them pulls in Isaac Lab. Registering namespace
    packages that point straight at the directories lets this benchmark build
    the real network with no simulator.
    """
    repo_root = pathlib.Path(__file__).parents[2]
    for pkg_name, relative_path in (
        ("isaacsimenvs.pose_reaching_6d.obs_utils", "isaacsimenvs/pose_reaching_6d/obs_utils"),
        ("isaacsimenvs.pose_reaching_6d.scene_utils", "isaacsimenvs/pose_reaching_6d/scene_utils"),
    ):
        if pkg_name not in sys.modules:
            pkg = type(sys)(pkg_name)
            pkg.__path__ = [str(repo_root / relative_path)]
            sys.modules[pkg_name] = pkg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minibatch_sizes", default="16384,32768,65536,98304,114688",
        help="Minibatch sizes to sweep. The epoch batch is FIXED, so a bigger "
             "minibatch buys proportionally fewer gradient steps -- the same "
             "arithmetic in fewer, larger launches. Sizes that do not divide "
             "the batch are fine: the fork's divisibility assert is commented "
             "out (a2c_common.py:251) and PPODataset gives the remainder to "
             "the last minibatch, which is what the 98304 reference run did.",
    )
    parser.add_argument(
        "--actor_batch", type=int, default=458752,
        help="Samples the ACTOR sees per mini_epoch: 24576 envs x horizon 16 "
             "= 393,216, which SAPG's off_policy_ratio 1.0 augments with one "
             "exploration block to 458,752.",
    )
    parser.add_argument(
        "--critic_batch", type=int, default=393216,
        help="Samples the CENTRAL VALUE net sees per mini_epoch. It is NOT "
             "augmented -- central_value.py:80 sets batch_size = "
             "horizon_length * num_actors -- so the two networks get different "
             "minibatch counts from the same configured size.",
    )
    parser.add_argument("--mini_epochs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--compile", action="store_true",
        help="Also time each network under torch.compile.",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    autocast = torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=args.amp
    )
    stub_kit_free_packages()
    minibatch_sizes = [
        int(v) for v in args.minibatch_sizes.split(",") if v.strip()
    ]
    nets = [("MLP actor", "actor", lambda: make_mlp())]
    for layers in (1, 4):
        nets.append((f"Transformer-L{layers} actor", "actor",
                     lambda n=layers: make_transformer(n)))
        nets.append((f"Transformer-L{layers} critic", "critic",
                     lambda n=layers: make_transformer(n, central_value=True)))

    def measure(net, optimizer, size: int, compile_net: bool = False):
        """(ms per gradient step, peak MiB allocated) at this minibatch size."""
        obs = torch.randn(size, 778, device=device)
        step_net = torch.compile(net) if compile_net else net

        def iteration():
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                out = step_net({"obs": obs})
                # The central-value network returns (value, states); the actor
                # returns (mu, sigma, value, states).
                loss = sum(t.square().mean() for t in out
                           if isinstance(t, torch.Tensor))
            loss.backward()
            optimizer.step()

        # Fewer iterations at the big sizes: the wall-clock per measurement is
        # what matters, not the count.
        iters = max(4, round(args.steps * minibatch_sizes[0] / size))
        for _ in range(args.warmup):
            iteration()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(iters):
            iteration()
        torch.cuda.synchronize(device)
        ms = 1e3 * (time.perf_counter() - start) / iters
        peak = torch.cuda.max_memory_allocated(device) / 2**20
        del obs
        torch.cuda.empty_cache()
        return ms, peak

    print(
        f"amp={args.amp} mini_epochs={args.mini_epochs} "
        f"actor_batch={args.actor_batch:,} critic_batch={args.critic_batch:,}\n"
        "per-epoch time models the ragged split PPODataset actually produces: "
        "(n-1) full minibatches plus one that absorbs the remainder.\n"
        "peak MiB is the training step alone -- add the run's fixed env + "
        "experience-buffer footprint to compare against the 49,140 MiB card.",
        flush=True,
    )

    for name, role, build in nets:
        batch = args.actor_batch if role == "actor" else args.critic_batch
        print(f"\n{name}", flush=True)
        baseline_epoch = None
        for size in minibatch_sizes:
            n_minibatches = batch // size
            if n_minibatches < 1:
                print(f"  minibatch {size:>7,}  skipped (larger than the batch)",
                      flush=True)
                continue
            last_size = size + (batch - n_minibatches * size)
            net = build().to(device).train()
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
            try:
                ms, peak = measure(net, optimizer, size)
                if last_size != size:
                    last_ms, last_peak = measure(net, optimizer, last_size)
                    peak = max(peak, last_peak)
                else:
                    last_ms = ms
                epoch_s = args.mini_epochs * (
                    (n_minibatches - 1) * ms + last_ms
                ) / 1e3
            except torch.cuda.OutOfMemoryError:
                print(f"  minibatch {size:>7,}  OOM", flush=True)
                del net, optimizer
                torch.cuda.empty_cache()
                continue
            if baseline_epoch is None:
                baseline_epoch = epoch_s
            ragged = "" if last_size == size else f" (last {last_size:,})"
            print(
                f"  minibatch {size:>7,}  {n_minibatches:>2} x "
                f"{args.mini_epochs} steps{ragged:<16}  "
                f"{ms:8.2f} ms/step  per epoch={epoch_s:7.3f} s  "
                f"({baseline_epoch / epoch_s:4.2f}x)  peak={peak:7.0f} MiB",
                flush=True,
            )
            del net, optimizer
            torch.cuda.empty_cache()

    print(
        "\nA SAPG epoch trains the actor AND the central-value net, so add the "
        "two rows a run actually uses; the two nets are alive at once, so add "
        "their peaks too.",
        flush=True,
    )


if __name__ == "__main__":
    main()
