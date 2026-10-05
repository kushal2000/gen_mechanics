# Grammar viewer

An interactive viser viewer for the hand-kinematics grammar (`hand_sampler/grammar/`). It samples designs from every named variant, shows commercial hands as the grammar expresses them, applies the evolution operators with a history, and runs the simulator's viability oracle on whatever is on screen. Everything runs on the CPU. Isaac, Kit and the GPU are never touched: the simulator's numpy-only envelope modules are loaded by file path (`gviewer/envload.py`), so `isaacsimenvs/__init__.py` and Isaac Lab are never imported.

## Setup (once)

```bash
cd "/home/singularity/Evolve to Generalize/gen_mechanics"
/home/singularity/.local/bin/uv venv .venv_viewer --python 3.11
/home/singularity/.local/bin/uv pip install --python .venv_viewer/bin/python viser "numpy==1.26.*" scipy trimesh yourdfpy pycollada pytest
/home/singularity/.local/bin/uv pip install --python .venv_viewer/bin/python -e . --no-deps
```

`pycollada` is optional; without it the `.dae` meshes of SVH and Shadow are reported as unloadable. `.venv_viewer/` is git-ignored by the existing `.venv*/` rule.

## Run

```bash
.venv_viewer/bin/python experiments/grammar_viewer/viewer.py --port 8080 --host 127.0.0.1
```

Open http://127.0.0.1:8080. From a laptop: `ssh -L 8080:127.0.0.1:8080 <this machine>`, then open the same URL locally. Optional flags: `--variant G_V1 --seed 12 --no-until-viable --out-dir <dir>`. The viewer starts on the first viable `G_V3S` design at or after seed 0.

## Panels

- **Source**
  - (a) Sample: variant dropdown (all of `NAMED_DISTRIBUTIONS`, G0 aliases in brackets), seed, prev/next/random. With "sample until viable" on, seeds are drawn upward (downward for prev) until `viability_report` admits a design with at least 2 fingertips reaching the spawn; the tries are shown.
  - (b) Commercial hand: every manifest hand outside the `excluded` split. The E13 projection is drawn as capsules over the original URDF meshes (toggle, opacity). The readout gives the fidelity (E13's measurement replayed: zero plus 64 random configurations, joint position and axis errors, fingertip error, pass at 5 mm / 10 deg), the 32-slot envelope fit with its rejection reasons, the capsule radius, `coverage` against the mutation distribution, and every merge, approximation and importer loss.
  - (c) File: a derivation JSON, a `population.json` (`designs[].derivation`), a driver `state.json` (archive elites) or a bare archive, browsable by index. A state file also sets the mutation distribution to its run's variant.
- **Pose**: a slider per independent joint (degrees, within its admissible range), curl-all (fraction of range), zero, env reset (`palm_up(...).default_q`), random, and an animated sweep (all joints together, or one joint at a time).
- **Mutate**: one button per `EVOLUTION_OPERATORS` entry, random mutation (the driver's pool), mutate until viable, a mutation distribution, back/forward and go-to-entry over the history, a lineage list labelled by operator, the parent drawn as a grey ghost, a parent-to-child diff (digits, phalanges, joints, changed parameters matched by step uid), and the mutation spread (K random children on a ring, coloured by viability, click one to adopt it).
- **Analysis**: `viability_report` (admitted, reasons, worst rest overlap, reaching fingertips per finger, spawn height), overlap pairs at q=0 and at the reset pose, live overlaps at the current pose, the MAP-Elites cell (bins from `evolution/archive.py`), the envelope slot map, counts, and the nearest commercial hands by `phenodist`.
- **View**: camera presets (top, side, palm-up), root frame or palm-up orientation (the env's `base_rot`), capsule convention, layer toggles, body and joint labels.
- **Gallery**: a rows by columns grid from one variant, or two variants one row each with the same seeds, coloured by viability. Click a hand to open it.
- **Export**: the derivation (plus a `.meta.json` with the report and lineage), a URDF via `adapters/urdf.to_urdf` with capsule collisions and palm-cell OBJ meshes, or the history. Files go to `outputs/grammar_viewer/` (git-ignored).

## What the colours mean

- Capsules are coloured per top-level digit; branch digits use a lighter shade of their host digit. Palm bodies are translucent grey nearest-spine cells (the cells of `geometry.build_geometry`).
- Joint arrows point along the joint axis (right-hand rule). At the rest pose, each digit joint is compared with its own link direction and the palm normal: blue is flexion, orange is abduction, blended by angle like the old sampler viewer. Purple is a twist joint (axis within 35 deg of its link), teal a palm joint.
- Fingertips are green when the oracle's sweep reaches the spawn point, orange when it does not. The blue sphere is the spawn point at the object's size (3 cm half size); the wire sphere is the 5 cm reach tolerance.
- Overlapping capsules are red beyond the oracle's 3 mm gate and pink when shallower. The highlight follows the current pose by default; it can show the oracle's own poses (q=0 and reset) instead.
- Capsule convention `simulator` is the PhysX capsule `rest_overlap_pairs` checks (total extent [0, L]); `grammar` is `geometry.Capsule` and the URDF export (core [0, L], extent [-r, L+r]).

## Tests

```bash
.venv_viewer/bin/python -m pytest experiments/grammar_viewer/tests -q
```

`pytest.ini` switches off the ROS pytest plugins that leak in through `PYTHONPATH`. The tests cover the render primitives (against `forward_kinematics` and `build_geometry`), the envelope loader (against `viability_report` and the archive bins), the commercial fidelity (equal to E13's `check_hand` without its Pinocchio part) for allegro, sharpa and svh, the mesh-overlay poses, the history round trip and diff, file parsing, and every viewer callback against a local server.

## Known limitations

- The Pinocchio export cross-check of E13 is not run (it needs a separate interpreter); the readout says so.
- `arms_skel` has no meshes at all. The SVH and Shadow URDFs in `karma-hand-metric` point at mesh files that do not exist; the viewer finds same-named `.dae` files in the downloads (`SVH/`, `Shadow/`) and marks them "alignment unverified". About 10 SVH meshes have no same-named file.
- Mesh overlays are reduced by vertex clustering to at most 6000 faces per piece; they are a visual reference.
- Projected hands use one capsule radius (10 mm, the projection's constant), so their rest overlaps are artefacts; the simulator exempts them, the Analysis tab still lists them.
- Nearest commercial: `phenotype_distance` aligns joints by name. Grammar and projection both name digit joints `d{k}p{i}_j`, but digit numbering is arbitrary and the two root frames follow different gauges, so treat it as a coarse similarity.
- Joint classes are computed at the rest pose. Allegro's first finger joints read as twist because at rest they rotate the straight finger about its own axis.
- The envelope's ghost and padding slots are never drawn: the scene is the grammar model, not the authored articulation.
