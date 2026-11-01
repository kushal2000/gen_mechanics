"""What log-std does a policy actually output? Runs ../eval_hands_cap.py unchanged (the hand set, sampled actions, the
SAPG leader) with a forward hook on the policy network that records its log-std output every step.

    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/predicted_log_std/probe_log_std.py \
        --probe-out <stats.json> <eval_hands_cap.py's arguments, e.g. --max-steps 600>

Writes, per action dimension that is not masked (ghost slots and the locked wrist output exactly 0 and are dropped):
the mean log-std over envs and steps, its spread across envs and steps (the state-dependent part), min / max, and the
mean over steps 0-49 / 50-199 / 200+ (early vs settled episode). A per-slot (state-independent) policy gives zero spread.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--probe-out", required=True)
    args, rest = ap.parse_known_args()
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "third_party/rl_games"))
    import torch
    from coevolution.eval import rl_player

    rec: list = []

    def dump() -> None:
        if not rec or Path(args.probe_out).exists():
            return
        x = torch.stack(rec)                                       # (steps, envs, actions)
        live = (x != 0).any(dim=0).any(dim=0)
        dims = {}
        for d in torch.nonzero(live).flatten().tolist():
            v = x[:, :, d]
            dims[d] = {"mean": float(v.mean()), "spread": float(v.std()), "min": float(v.min()), "max": float(v.max()),
                       "steps_0_49": float(v[:50].mean()),
                       "steps_50_199": float(v[50:200].mean()) if len(v) > 50 else None,
                       "steps_200_on": float(v[200:].mean()) if len(v) > 200 else None}
        Path(args.probe_out).write_text(json.dumps({"steps": x.shape[0], "envs": x.shape[1], "dims": dims}, indent=1))
        print(f"[probe] {x.shape[0]} steps x {x.shape[1]} envs, {len(dims)} live action dims -> {args.probe_out}", flush=True)

    orig_init = rl_player.RlPlayer.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        net = self.player.model.a2c_network
        net.register_forward_hook(lambda _m, _i, out: rec.append(out[1].detach().float().cpu()))

    rl_player.RlPlayer.__init__ = init
    real_exit = os._exit

    def dump_then_exit(code):            # eval_hands_cap.py ends with os._exit(0), which skips any finally
        dump()
        real_exit(code)

    os._exit = dump_then_exit
    sys.argv = [str(REPO / "experiments/30oct_evolution_evaluation/eval_hands_cap.py")] + rest
    runpy.run_path(sys.argv[0], run_name="__main__")
    dump()


if __name__ == "__main__":
    main()
