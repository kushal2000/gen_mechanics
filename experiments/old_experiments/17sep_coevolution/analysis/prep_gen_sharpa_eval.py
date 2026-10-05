"""Build the populations for the zero-shot gen-SHARPA evaluation.

For each generation, gen-SHARPA replaces the design that generation's own
reward table ranked LAST -- a design that was about to be culled anyway, so the
1023 hands it competes against are that generation's strongest 1023 and its
rank among them is if anything pessimistic.

    .venv_isaacsim/bin/python experiments/old_experiments/17sep_coevolution/analysis/prep_gen_sharpa_eval.py <label> <gen> [gen ...]
"""
import glob, json, pathlib, sys

from coevolution.design_rewards import merge_rank_files
from hand_sampler import population_io
from hand_sampler.sharpa_capsule import sharpa_capsule

OUT = pathlib.Path("debug_outputs/17sep_coevo_analysis/gen_sharpa_eval")


def build(label: str, gen: int) -> dict:
    P = pathlib.Path("assets/populations") / label / f"gen_{gen}"
    hands = population_io.load_population(P / "population.json")
    rewards = merge_rank_files(sorted(glob.glob(str(P / "design_rewards_rank*.json"))))
    worst = min(range(len(hands)), key=lambda i: rewards.get(i, {}).get("return_mean", -1e9))
    hands = list(hands); hands[worst] = sharpa_capsule()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{label}_gen{gen:03d}.json"
    population_io.save_population(
        hands, path, name=f"{label}_gen{gen}_with_gen_sharpa",
        provenance=population_io.provenance(
            method="gen_sharpa_eval", source=str(P / "population.json"), generation=gen,
            replaced_index=worst, replaced_return=rewards.get(worst, {}).get("return_mean")))
    meta = dict(label=label, gen=gen, slot=worst, population=str(path),
                checkpoint=(P / "checkpoint.txt").read_text().strip(),
                run_dir=(P / "run_dir.txt").read_text().strip(),
                tolerance=float((P / "success_tolerance.txt").read_text().strip()))
    (OUT / f"{label}_gen{gen:03d}.meta.json").write_text(json.dumps(meta, indent=1))
    print(f"gen {gen:3d}: gen-SHARPA into slot {worst} (its return was "
          f"{rewards.get(worst, {}).get('return_mean', float('nan')):.0f}), tolerance {meta['tolerance']:.4f}")
    return meta


if __name__ == "__main__":
    label, gens = sys.argv[1], [int(g) for g in sys.argv[2:]]
    for g in gens:
        build(label, g)
