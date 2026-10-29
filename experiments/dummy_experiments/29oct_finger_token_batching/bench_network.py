"""The implemented ``packed_tokens`` flag (joint_transformer.py) against the padded network, both token modes.

    .venv_isaacsim/bin/python experiments/dummy_experiments/29oct_finger_token_batching/bench_network.py [--quick]

The network itself, compiled as training runs it, on the 100-hand template layout (bench.py's setup): the same
weights with ``packed_tokens`` off and on; outputs compared first (mu, sigma, value; fp32), then timed: rollout
inference at 24,576 envs and a train step (forward, backward, Adam) at 16,384 rows, fp16 autocast. Joint mode is the
one bench.py did not measure: 36 joint slots + the global token, 13.4 real on average (64% ghosts).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402  (this folder's: the layout, observations and timing)


def make(mode: str, n_layers: int, packed: bool):
    from coevolution.networks.joint_transformer import JointTransformerNet, _WithArmJoints
    from isaacsimenvs.common.obs_layout import field_offsets
    from isaacsimenvs.common.robots import get_robot_spec
    spec = _WithArmJoints(get_robot_spec(bench.HAND), 6)
    obs_dim = max(e for _, e in field_offsets(bench.OBS_LIST, spec, {"task_onehot": 8}).values())
    params = {"robot_spec": bench.HAND, "obs_list": bench.OBS_LIST, "extra_arm_joints": 6,
              "extra_field_sizes": {"task_onehot": 8}, "token_mode": mode, "d_model": 128, "n_layers": n_layers,
              "n_heads": 4, "ff_mult": 4, "dropout": 0.0, "decoder_units": [256], "arm_head_units": [1024, 512],
              "value_head_units": [512, 256], "affine_norm": False, "mask_ghost_actions": True,
              "masked_arm_dims": [3, 4, 5], "compile_net": True, "packed_tokens": packed, "space": bench.SPACE}
    net = JointTransformerNet(params, actions_num=6 + spec.num_hand_joints, input_shape=(obs_dim,), value_size=1,
                              num_seqs=1, type="simple")
    return net, obs_dim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results_network.json"))
    args = ap.parse_args()
    dev = torch.device("cuda")
    iters, warm = (5, 3) if args.quick else (20, 6)
    print(f"{torch.cuda.get_device_name()}; torch {torch.__version__}", flush=True)
    rows = []
    for mode in ("finger", "joint"):
        for n_layers in (2, 4):
            res = {}
            for packed in (False, True):
                torch.manual_seed(0)
                net, obs_dim = make(mode, n_layers, packed)
                net = net.to(dev)
                if packed:
                    net.load_state_dict(ref_state)
                else:
                    ref_state = {k: v.clone() for k, v in net.state_dict().items()}
                x_inf = bench.env_observations(net, obs_dim, bench.N_ENVS, dev, seed=0)
                pick = torch.randint(0, bench.N_ENVS, (bench.MB,), generator=torch.Generator().manual_seed(1)).to(dev)
                x_tr = x_inf[pick] + 0.0
                net.eval()
                with torch.no_grad():
                    res[packed] = {"out": [t.float() for t in net({"obs": x_inf[:8192]})[:3]]}

                def fwd(x, net=net):
                    return net({"obs": x})[:3]
                net.train()
                t_inf, t_tr = bench.time_forward(net, fwd, fwd, x_inf, x_tr, iters, warm)
                res[packed].update(inference_ms=t_inf, train_ms=t_tr)
                if packed:
                    res[packed]["caps"] = dict(net._pack_caps)
                del net
                torch._dynamo.reset()
                torch.cuda.empty_cache()
            err = max(float((a - b).abs().max()) for a, b in zip(res[False]["out"], res[True]["out"]))
            for packed in (False, True):
                r = res[packed]
                net_s = bench.HORIZON * r["inference_ms"] / 1e3 + bench.UPDATES * (r["train_ms"] + bench.LOOP_MS) / 1e3
                rows.append({"mode": mode, "layers": n_layers, "packed": packed, "inference_ms": r["inference_ms"],
                             "train_ms": r["train_ms"], "network_s_per_epoch": net_s, "max_abs_diff": err,
                             "caps": r.get("caps")})
            a, b = rows[-2], rows[-1]
            print(f"{mode:6s} L{n_layers}: packed == padded to {err:.1e} (caps {b['caps']})\n"
                  f"   padded  inference {a['inference_ms']:6.1f} ms  train {a['train_ms']:6.1f} ms  "
                  f"network {a['network_s_per_epoch']:.2f} s/epoch\n"
                  f"   packed  inference {b['inference_ms']:6.1f} ms  train {b['train_ms']:6.1f} ms  "
                  f"network {b['network_s_per_epoch']:.2f} s/epoch  ({a['network_s_per_epoch'] / b['network_s_per_epoch']:.2f}x)",
                  flush=True)
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
