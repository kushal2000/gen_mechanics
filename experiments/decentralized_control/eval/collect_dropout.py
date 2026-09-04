"""Merge the per-variant dropout JSONs into one table.

Each eval_dropout.sub job writes debug_outputs/eval_logs/<tag>/<variant>.json;
this reads whatever is there and prints them in finger order, ratioed to the
intact control.

    python experiments/decentralized_control/eval/collect_dropout.py
    python experiments/decentralized_control/eval/collect_dropout.py --tag dropout_seed1
"""

from __future__ import annotations

import argparse
import json
import pathlib

ORDER = ["intact", "thumb", "index", "middle", "ring", "pinky"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="dropout")
    p.add_argument("--dir", default="debug_outputs/eval_logs")
    args = p.parse_args()

    root = pathlib.Path(args.dir) / args.tag
    found = {}
    for variant in ORDER:
        path = root / f"{variant}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        # Each job runs one variant, so its results dict has a single entry.
        for name, row in payload["results"].items():
            found[name] = (row, payload)

    if not found:
        print(f"no results under {root}; jobs may still be running")
        missing = [v for v in ORDER if not (root / f"{v}.json").is_file()]
        print(f"missing: {', '.join(missing)}")
        return

    any_payload = next(iter(found.values()))[1]
    print(f"\nZERO-SHOT FINGER DROPOUT   {any_payload['num_envs']} envs x "
          f"{any_payload['steps']} steps, "
          f"{'deterministic' if any_payload['deterministic'] else 'sampled'}")
    print(f"checkpoint {any_payload['checkpoint']}")
    hdr = (f"\n{'variant':<9} {'hand':>5} {'act':>4} {'tips':>5} {'goals/env':>10} "
           f"{'std':>7} {'vs intact':>10} {'envs scoring':>13} {'episodes':>9}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    base = found.get("intact", (None,))[0]
    base = base["goals_per_env_mean"] if base else None
    for variant in ORDER:
        if variant not in found:
            print(f"{variant:<9} {'(pending)':>50}")
            continue
        r = found[variant][0]
        rel = f"{r['goals_per_env_mean'] / base:9.2f}x" if base else f"{'-':>10}"
        print(f"{variant:<9} {r['hand_joints']:>5} {r['actions']:>4} "
              f"{r['fingertips']:>5} {r['goals_per_env_mean']:>10.3f} "
              f"{r['goals_per_env_std']:>7.3f} {rel:>10} "
              f"{r['frac_env_with_a_goal'] * 100:>12.0f}% "
              f"{r['episodes_per_env']:>9.1f}")

    merged = root / "merged.json"
    merged.write_text(json.dumps(
        {"checkpoint": any_payload["checkpoint"],
         "num_envs": any_payload["num_envs"], "steps": any_payload["steps"],
         "deterministic": any_payload["deterministic"],
         "results": {k: v[0] for k, v in found.items()}}, indent=2))
    print(f"\nwrote {merged}")


if __name__ == "__main__":
    main()
