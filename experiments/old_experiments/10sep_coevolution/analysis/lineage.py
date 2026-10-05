"""How many generation-0 ancestors are still represented, generation by
generation, traced through each selection.json. Also per-generation design
diversity: distinct designs, finger and joint histograms."""
import collections, json, pathlib, sys
from hand_sampler import population_io

R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "assets/populations/coevo_r500_v1")
gens = sorted(int(p.name.split("_")[1]) for p in R.glob("gen_*") if (p / "population.json").exists())
anc = list(range(len(population_io.load_population(R / "gen_0/population.json"))))
print("gen  ancestors  largest lineage")
for g in gens[1:]:
    s = json.load(open(R / f"gen_{g}/selection.json"))
    src = s["survivors"] + [c["parent"] for c in s["children"]]
    anc = [anc[src[i]] for i in range(len(anc))]
    c = collections.Counter(anc)
    print(f"{g:3d}  {len(c):9d}  {c.most_common(1)[0][1]:6d}  (gen-0 design {c.most_common(1)[0][0]})")
print()
for g in (gens[0], *[x for x in (5, 10, 20) if x in gens], gens[-1]):
    hands = population_io.load_population(R / f"gen_{g}/population.json")
    distinct = len({json.dumps(population_io.hand_to_dict(h), sort_keys=True) for h in hands})
    print(f"gen {g:2d}: {distinct}/{len(hands)} distinct designs; fingers {dict(sorted(collections.Counter(h.n_fingers for h in hands).items()))}; "
          f"joints {dict(sorted(collections.Counter(h.n_joints for h in hands).items()))}")
