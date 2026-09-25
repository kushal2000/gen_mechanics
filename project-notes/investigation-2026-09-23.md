# Evolve to Generalize: onboarding for Martin

Investigation date: 23 September 2026. Source checkout: `gen_mechanics`, commit `8d1395320f7027ccd088ba5593bde50a69eecf65`. The existing `codex/martin-hand-sampler` branch, local `master`, and cached `origin/master` point to that commit. No remote fetch, branch change, implementation edit, commit, or push was performed. This document is outside the Git repository.

## 1. Your assignment and its implementation boundary

The team instruction is to branch from `master` and make changes in `hand_sampler`. That is consistent with the most direct assignment in the Notion notes: rethink the grammar and mutations for a search budget of about 40 generations. Understanding the simulator, shared controller, selection score, and experiment setup is necessary to do that work, but changes to those systems should be coordinated rather than included silently in a sampler change.

The source is the [25-page Notion export](../reference/Evolve%20to%20Generalize%20-%20Notion%20export.pdf), exported on 21 September and last edited by Martin Peticco. The top-level file named `Evolve To Generalize.pdf` is actually a ZIP archive. Its embedded PDF is byte-identical to the copy in `reference/`. All 25 pages were inspected, including figures, and three frames from each of the 14 exported videos were sampled. Those frames provide qualitative context, not performance measurements or a complete temporal review. The linked Prior Works page and task-card contents are not included. The exported task CSV has three unnamed-purpose cards with no assignments or dates.

| PDF location | What the discussion says | Interpretation for your work |
|---|---|---|
| pp. 3–4 | Explicitly tags Martin and Vatsal on deliberately diverse objects and suggests Visual Dexterity objects | Define which object geometries, physical properties, and grasp affordances the benchmark should cover. Coordinate simulator changes. |
| p. 9 | Explicitly tags Martin on whether repositioning/reorientation is general enough, mentioning environment interaction, power grasps, and more shapes | Determine what capability the current score measures and which additional tests would distinguish useful hands. Keep simulation cost manageable. |
| pp. 9–10 | Explicitly tags Martin and Vatsal on grammar and mutation design for 40 generations | Audit physical realizability, mutation locality and scale, reachable designs, and discrete versus continuous parameter choices. |
| pp. 5–6 | Questions whether common ancestry proves low diversity; requests tunable mutation size and a physically realizable option | Separate genealogical, geometric, and behavioral diversity. Measure actual changes rather than counting mutation events alone. |
| pp. 8–9 | Questions excessive complexity and asks about mass, energy, and distance between hands | Quantify the search prior and performance–complexity tradeoff before choosing a penalty. |
| pp. 5, 9, 12 | Suggests SHARPA transfer tests, insertion of a reference hand mid-search, controller resets, alternate selection, and random immigrants | Treat these as distinct ablations. They are team discussion, not all explicitly assigned to Martin. |

The export does not reliably identify who wrote each untagged comment. Last-edited-by metadata does not establish authorship of all notes.

## 2. The research question and the evidence it requires

The stated objective is a bilevel design problem: train a controller on one distribution, then optimize hardware for performance under a shift in objects, physics, sensing, disturbances, or tasks. The intended outputs are a hand optimized for generalization and a co-design algorithm.

The present task is grasping a tool-like object from a table and moving it through a sequence of SE(3) goals. The nominal curriculum parameter tightens from 0.075 to 0.01. The implementation compares maximum error over four pose keypoints against that parameter multiplied by `keypoint_scale=1.5`, so the corresponding distance thresholds are 0.1125 m and 0.015 m under the active defaults (`obs_utils/observations.py:233–255`, `coevolution/cfg/task/PoseReach.yaml:99–104,145–150`). The notes' “1 cm” is the nominal parameter, not an independent 1 cm translation bound. These values should not be read as separate translation and orientation thresholds or as an automatic measure of in-hand dexterity.

The default success rule accumulates ten near-goal frames rather than requiring ten consecutive frames. Policy control is 60 Hz, and the nominal ten-second timeout resets after each goal, so successful episodes can last much longer than ten seconds. Both details matter when comparing goals per episode and raw episode return across designs.

Three claims need separate evidence:

1. An evolving population improves training of a shared controller.
2. The resulting hardware is better or easier to control under comparable training.
3. The hardware generalizes better under conditions not used to select it.

The September 18 figures report encouraging training behavior for the first claim. They do not establish the second or third. The recorded loop ranks training reward, and the same controller has a different exposure history to each surviving morphology. Its ranking measures hardware together with controller competence and training history.

A useful evaluation separates policy-training data, design-selection validation data, and an untouched final test distribution. Repeatedly selecting hardware using a nominally held-out set makes that set part of optimization. Report absolute held-out performance as well as retention relative to nominal performance.

Evaluation building blocks exist: condition suites, a policy player, and a specialized finger-dropout evaluator. However, several historical generalization/population-evaluation entry points have been deleted, and the current outer loop does not call held-out evaluation, Thompson sampling, or value-function acquisition. The saved-config reconstruction helper also omits current `object_pool`, `object_assignment`, and `geometry_origin` fields (`coevolution/population/run_config.py:32–77`). A new evaluator must preserve those semantics explicitly. Changing only an object seed will not produce unseen objects while a fixed curated pool remains selected.

Selection currently aggregates completed episodes across a generation while both policy competence and curriculum difficulty change. It is not a frozen-policy score at the end of training. Equal object allocation also does not guarantee equal object weighting when score aggregation is per completed episode: some objects or designs can generate more or shorter episodes. Approach rewards sum over fingertips and velocity costs sum over joints, adding direct morphology-count dependence to return.

The PDF's “imitation” comparison means reconstructing a SHARPA-like morphology from grammar primitives, then training with RL. It does not mean behavior cloning. Comparing a policy trained on one reconstructed hand to a policy trained across 1,024 evolving designs does not isolate hardware quality: the distribution of experience and control problem are different.

The suggested test of training the best evolved hand and reference hands from scratch is therefore important. Match total budgets, evaluation conditions, success tolerance, and seed counts, and disclose physical differences between the reference models.

## 3. How to interpret the PDF's results

The September 18 discussion reports that co-evolution improves training reward and reaches tighter goal tolerances than a fixed population. It also reports ancestry convergence by generation 13 and increasing design complexity. These are reported historical results, not experiments rerun in this investigation.

Several distinctions matter:

- A single surviving founder does not imply a single remaining morphology. Descendants can differ substantially.
- A 40-generation experiment does not give every design 40 mutations. Unchanged survivors remain in the population. The best-design ancestry figure itself shows only 12 mutation events across the illustrated path through generation 39.
- The simple initial designs and the population after 500 neutral mutation rounds are different starting distributions. The latter has already accumulated complexity before reward-driven search.
- Smaller world-frame goal error does not, by itself, establish more finger dexterity. An arm can move an object while the hand maintains a nearly rigid grasp. Measure object motion relative to the palm and use controlled arm/hand ablations.
- The PDF gives both 12 and 90 hours for a 12,000-epoch baseline in different meeting sections. Use saved configurations and logs, not prose elapsed-time estimates, for compute comparisons.
- The latest object illustrations and older training curves should not be assumed to describe the same experimental protocol.

The September 4/11 finger-dropout result is a narrower transfer test: the notes report 25.8 goals/environment with the intact hand, 24.2 without the index finger, and 2.3 without the thumb (pp. 10 and 16). This suggests substantial tolerance to some missing fingers and strong dependence on the thumb in that setup. It does not establish transfer across arbitrary hands or tasks.

The raw population histories, reward tables, and checkpoints are not in this checkout, so the historical curves and lineage identities cannot be independently reconstructed here.

## 4. Literature attached to the discussion

The “paper” hyperlink beside the diversity comment on p. 5 is [Strgar and Kriegman, *Accelerated co-design of robots through morphological pretraining*, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0cfeabacfcdf49ce45826a2f7eb89a3c-Abstract-Conference.html). The relevant evidence is in Sections 2.5–2.7 and 3.2–3.3 and Figure 6.

The paper measures diversity using mean pairwise normalized genotype Hamming distance, separately from performance, across three trials. Its few-shot method resets to the same pretrained controller at the beginning of every generation, reinitializes the optimizer, and performs 60 fine-tuning steps. This is different from carrying the latest policy forward and different from resetting to random weights. The pretrained-reset intervention is a useful additional experimental condition for this project.

Those results concern simulated mass-spring robots with differentiable dynamics, not dexterous articulated hands. The paper explicitly identifies its use of one diversity metric and lack of physical transfer as limitations. Its Hamming distance should not be copied without considering finger permutations, continuous parameters, and kinematic equivalence.

The notes name but do not link a specific Visual Dexterity asset set. The likely reference is [Chen et al., *Visual Dexterity*](https://taochenshh.github.io/projects/visual-dexterity), which studies in-hand reorientation of novel complex shapes using depth sensing. Its reported failures on a duck-shaped object illustrate why shape diversity is useful. It is not evidence for insertion or sustained contact performance. The project page links `Improbable-AI/dexenv`, but selecting assets and checking their licensing and collision representations would be a separate task.

## 5. Recommended initial contribution

The first deliverable should make the existing search interpretable before changing several causes at once:

1. Establish a correctness baseline for mutation geometry, geometric validation, population records, and simulator hand interfaces.
2. Audit mutation outcomes over exactly 40 generations from both simple seeds and the actual round-500 starting population. Track proposals, validity rejection, retries, null copies, accepted operator frequencies, changed joints, and geometric displacement.
3. Report separate ancestry, topology, geometry, and behavioral statistics. Pair mount-face identity with chain structure rather than sorting both independently. Use distances invariant to arbitrary finger ordering where appropriate.
4. Expose mutation step size, number of edited joints, and number of composed operations as independent controls. A whole-hand perturbation and a one-joint edit should not share an unexplained “mutation size.” Compare discrete and continuous variants at matched physical perturbation scales and bounds, rather than attributing a scale change to discretization.
5. Add a physically constrained option with explicit module dimensions, torque, travel, mass, and clearance assumptions. Passing a rest-pose geometry check is not proof of manufacturability.
6. Evaluate one change at a time under matched controller and evaluation budgets. Preserve a performance–complexity archive before choosing a scalar complexity penalty.

Motor count, simulated mass, and energy are distinct quantities. Motor count can be read from topology; the current geometric mass model does not account for a full actuator/transmission bill of materials; energy requires trajectories and a declared mechanical or electrical definition.

For task breadth, propose a small object taxonomy based on grasp width, cross-section, asymmetry, center-of-mass offset, and contact requirements. Measure whether the current policy reorients objects relative to the palm before adding a large collection of meshes. One controlled interaction task can be more informative than many visually different versions of the same handle.

## 6. Experimental controls to agree on before a new GPU run

The latest committed protocol is `experiments/17sep_coevolution`. It uses 1,024 round-500 designs, a shared d64/four-layer joint transformer, and 24,576 environments across two GPU ranks. Each design is assigned the same curated set of 24 objects across the ranks. Every 5,000 epochs, it ranks mean training episode return, keeps 512 hands, and mutates survivors to refill the population. Policy weights and the scalar curriculum tolerance are carried into a new training process.

The actual runner overrides the base YAML: the actor uses privileged `state_list` observations, the separate central critic is disabled, action smoothing is removed, and configured observation/action delays, velocity observation noise, and external disturbance forces are disabled (`experiments/decentralized_control/scaling_laws/run_rank.sh:109–131`). Read the launcher and runner together with the YAML before describing a run as realistic or domain-randomized.

V2 adds palm geometry to the observation, changes geometry coordinates to the EE frame in metres, gives each hand the same object set, and lengthens training between selection events. V1 assigned different object subsets to different designs, so its design ranking was additionally confounded by object assignment (`experiments/17sep_coevolution/README.md:14–31`). V2's curated tool diversity is not itself an untouched generalization benchmark.

The default planned budgets differ: co-evolution specifies 40 generations of 5,000 epochs (200,000 total), while the baseline specifies ten links of 15,000 epochs (150,000 total). Restart cadence also differs. Compare at matched checkpoints or explicitly align these budgets; timeouts can further change the realized budgets. Relevant defaults are in `coevo_gen.sub:34–36` and `baseline.sub:33` under that directory.

The V1 documentation reports a 0.01 m tolerance by generation 31, compared with 0.029 m for the fixed population after approximately 66k epochs, and an increase from 11.2 to 18.1 mean joints. These are reported training outcomes (`experiments/10sep_coevolution/README.md:68–76`), not measurements of held-out generalization.

Useful comparisons to coordinate with the policy and simulator work are:

| Comparison | Question | Essential control |
|---|---|---|
| Simple seeds versus round-500 initialization | Does neutral pre-drift remove the intended simple-to-complex curriculum? | Same search and training budgets, starting seeds, and objects |
| Current policy carry-over versus frozen pretraining versus pretrained reset/fine-tuning | Is selection favoring designs the current controller already understands? | Count pretraining compute and fix evaluation conditions |
| 50% replacement versus gentler replacement, immigrants, or niches | Is useful variation being lost too early? | Define survivor, offspring, and immigrant counts explicitly |
| Local versus whole-hand mutations at controlled scales | Is search too slow, or does mutation disrupt useful structure? | Measure proposal acceptance and actual geometric change |
| Cross-evaluate final policies on both populations | Did hardware improve, or did the controller specialize? | Same tolerance, object set, episode budget, and checkpoint semantics |
| Train selected evolved and reference hands from scratch | Does the hardware improve learnability or final performance? | Equal training budgets and disclosed actuation/geometry differences |
| SHARPA evaluation throughout evolution | Does transfer improve or deteriorate as the population changes? | Correct morphology remapping and observation normalization |

The notes' phrase “reduce selection rate to 25%” needs an explicit interpretation. Keeping 25% increases truncation pressure; replacing 25% decreases it. The code's `KEEP` is the number of survivors.

Also distinguish the final evaluated population from the last population file written. The launcher creates offspring before testing its stopping condition (`experiments/17sep_coevolution/coevo_gen.sub:58–83`). With generations 0–39 trained, it still writes `gen_40/population.json`, but does not train or evaluate that population. New children in that file should not be described as measured final winners without an additional evaluation.

For lexicase selection, aggregate mean return is insufficient. The method needs a per-design, per-case performance table, with the cases and noise treatment defined. Adding a selection method without that data would not implement the intended experiment.

## 7. Existing analysis worth reusing

The cached branch `origin/2026-09-18-analyze_results` adds `experiments/10sep_coevolution/analysis/`, which is not present on local master. Inspect it before writing new lineage or trait scripts:

```bash
# From gen_mechanics/
git show origin/2026-09-18-analyze_results:experiments/10sep_coevolution/analysis/README.md
git diff --stat master origin/2026-09-18-analyze_results
```

There is a concrete inconsistency between its lineage figure labels and analysis text. The branch's README reports founder #521 fixing by generation 13 and founder #933 being removed in generation 1. `phylogeny.py:50–52` sorts the surviving founder first, but lines 62 and 68 label `top[-1]` as the fixed founder. The PDF combines a #933 fixation annotation with a #521 ancestry caption. The annotation-index defect is confirmed from code; reconstructing the founder counts still requires the missing population histories.

The plotting code also masks the first 200 epochs after restarts, interpolates gaps, applies smoothing, and uses a shared 5.5 seconds per epoch conversion despite recording a different measured rate and startup overhead for evolution. Those presentation choices should be disclosed, especially for wall-time claims. Some scripts import an external `_style` module from another repository and depend on cluster-only data paths.

## 8. Repository map for your work

The main flow is:

```text
Hand genotype -> mutation and validity -> saved population
    -> padded robot geometry and metadata -> Isaac environments
    -> shared joint-transformer training -> per-design episode returns
    -> survivor selection and offspring -> next population
```

| Area | Responsibility | Relevance to Martin |
|---|---|---|
| `hand_sampler/design_space.py` | Hand, palm, finger, mount, segment, and joint representation; forward kinematics and geometry | The meaning of a design and of geometric distance |
| `hand_sampler/gen_init_pop.py` | Simple initial hand sampling | The initial search prior |
| `hand_sampler/mutate_design.py` | Structural and parameter mutations | Main implementation surface for mutation locality, scale, and coverage |
| `hand_sampler/validate_design.py` | Grammar bounds and collision checks | Validity must match intended physical constraints |
| `hand_sampler/drift.py` | Neutral mutation walks | Separates mutation bias from fitness-driven selection |
| `hand_sampler/evolve.py` | Rank, retain, mutate, and refill | Selection and ancestry accounting |
| `hand_sampler/population_io.py`, `robot_spec.py` | Population identity, serialization, fixed-size metadata, design assignment | Compatibility boundary with training |
| `hand_sampler/build.py`, `robot_param_constants.py` | Authored geometry, inertial properties, actuator assumptions, collision filtering, viewer URDFs | The physical model behind the genotype |
| `hand_sampler/experiments/` | Resumable sampler experiments, statistics, plots | Reuse this infrastructure for mutation and diversity diagnostics |
| `hand_sampler/sharpa_capsule.py`, `viewer.py` | Grammar-compatible reference approximation and visualization | Inspect representational compromises and individual changes |
| `isaacsimenvs/pose_reaching_6d/` | Task scenes, observations, actions, rewards, resets, goals, and curriculum | Determines what the search score rewards |
| `coevolution/` | RL startup/configuration, custom network, per-design statistics, evaluation, checkpoint integration | Controller history and evaluation protocol affect design ranking |
| `experiments/` | SLURM jobs, active co-evolution protocols, architecture and dropout studies, archived jobs | Effective settings often override YAML defaults |
| `third_party/rl_games/` | Vendored SAPG implementation | Observation normalization and checkpoint semantics are part of the model |
| `assets/`, `debug_outputs/` | Reference robot assets and placeholders for generated data and runs | The checkout is not a complete experimental-data release |

The root README is not a reliable current file map: it lists removed modules and an obsolete `python -m hand_sampler.population` command. `hand_sampler/DESIGN.md` also describes several now-implemented components as future work. The sampler experiments README contains historical results from a larger joint envelope than the current code permits. Neither `PLANNING.md` nor `TASK.md` is present.

### The current design representation

A generated hand is a rectangular palm with 2–5 serial fingers and 1–6 independently actuated joints per finger. It is not an arbitrary graph of branching or coupled mechanisms. The simulator reserves five finger slots with six joint slots each, plus seven arm joints, even when a hand uses only a few of those slots.

The simple seeds have two fingers and 2–4 total joints. They mount on adjacent thin-face pairs, with zero rest offsets and flexion/abduction axes. Sampling and mutation therefore define a specific prior over opposition and initial grasp geometry, not a neutral distribution over all hands.

Key restrictions include 15–80 mm links on a 5 mm grid, 15-degree axis/rest-angle increments, a fixed axis polar angle, and ±90-degree joint travel. Palm width and length are bounded to 40–100 mm; thickness has broader nominal bounds but is not evolved by the current palm mutation. Mounts occupy three thin palm faces. The permitted link count and link-length bounds allow very long chains in principle, so “bounded” does not mean hand-sized or manufacturable.

Parameter mutations and topology mutations have different scopes. Axis, offset, and length perturbations can affect every joint/link, whereas a mount move affects one finger. A finger can only be removed once reduced to a single joint, and splitting/merging is restricted by link-length bounds. These constraints affect how far search can travel within 40 generations.

The neutral walk, accepted-child generator, and resumable sampler experiments use different invalid-proposal/retry/replacement rules. They should not be interpreted as interchangeable samples from one neutral distribution. Track proposals and accepted changes separately.

The interfaces to preserve initially are population identity and provenance, the fixed joint-slot envelope and names, active-joint masks and limits, geometry token definitions and coordinate frames, fingertip semantics, and consistency between the geometry checked by the validator and the geometry authored for simulation. Changing a link dimension is not isolated if the validator, token builder, and simulator use different physical conventions.

The shared controller is a full-attention joint transformer with shared per-joint hand outputs and a separate arm head. “Decentralized” in the notes does not mean independent policies for isolated joints or communication restricted to kinematic graph edges. Padding preserves the action and observation dimensions across designs. The sampler's geometric metadata therefore has a direct effect on policy input, rather than serving only asset generation.

The package dependency arrow in the root README describes the intended structure, not a strict import boundary. Core sampling and mutation run without Isaac, but USD authoring needs simulator libraries, the viewer uses co-evolution viewing helpers, and the evolution CLI calls the co-evolution reward merger. Do not assume every operation under `hand_sampler` is dependency-free.

### What representing a real hand currently means

Generated hands are authored directly to USD. Their generated URDFs are for viewing, so changing that URDF path alone does not change the simulated hand. The physical model is a cuboid palm plus capsule links, with common generated-joint actuator settings rather than explicit motor housings or per-joint hardware selection.

Measured URDF feature extraction is implemented, but `hand_from_urdf` is not a faithful physical importer (`hand_sampler/design_space.py:817–872`). It preserves measured feature boxes while substituting a default palm, artificial mounts, and simplified joints. It does not retain the original mount transforms, axes, limits, actuator settings, masses, or meshes as a physically equivalent generated hand. The actual SHARPA reference instead uses its original URDF and a separate RobotSpec. It has 22 hand joints; the generated capsule approximation has 21 and deliberately changes several anatomical details to satisfy the grammar.

Optional axis, limit, drive, mesh, cross-section, and feature-box fields exist in the dataclasses and JSON, but their presence does not imply consistent simulation or mutation support. Several mutations reconstruct core Joint/Segment fields and silently discard optional metadata. Current generated authoring and the population actuator template also ignore important drive/geometry overrides. Before adding real-hand mutations, decide which properties must survive and test them explicitly.

Rigid, passive, or coupled fingers require a contract change. Setting a real segment's limits to `(0, 0)` is not sufficient: its geometry is marked present while the runtime enabled mask is false, violating the current geometry-validity invariant. Palm cuts similarly require coordinated changes to surface attachment, clearance, mass/inertia, authoring, and token geometry, rather than only a new genotype field.

Some legal designs are unreachable from current seeds. Palm thickness never changes, and 10 mm width/length mutations preserve a parity class despite a 5 mm legal grid. For example, the capsule-SHARPA's legal 85 mm palm length cannot be reached from standard seed lengths with the present palm operator. This matters when assessing whether a 40-generation search can reach a desired reference morphology.

## 9. Fresh CPU diagnostic: neutral complexity growth

A small in-memory experiment used one population of 128 initial hands, generation seed 7, walk seed 7, and 40 rounds with one mutation proposal per hand per round. Settling was disabled by using a 100-round window, so the walk could not stop early. There was no fitness selection.

| Quantity | Initial | After 40 rounds |
|---|---:|---:|
| Mean joints per hand | 2.96094 | 5.52344 |
| Mean fingers per hand | 2.00000 | 3.28906 |
| Hands at five-finger cap | 0 | 19 / 128 |
| Hands containing a six-joint finger | 0 | 1 / 128 |

All 128 final hands passed the current validator. That is a statement about the existing validator, not a certificate of physical validity. This is one small diagnostic, without replicated seeds or uncertainty estimates, and it does not characterize stationarity or task performance. It demonstrates that complexity can increase through the mutation process alone and that the full-depth fingertip case is reachable within 40 rounds.

## 10. Development environment and reproducibility

The project declares Python `>=3.11,<3.12` and does not list its heavy simulator stack as install dependencies (`gen_mechanics/pyproject.toml:1–51`). An editable install alone will not provision training.

The documented training stack is Python 3.11, PyTorch 2.7.0 with CUDA 12.6, NumPy 1.26.0, Isaac Lab 2.3.2.post1, Isaac Sim 5.1.0.0, and the editable vendored `third_party/rl_games`. Follow the pinned recipe in `gen_mechanics/README.md:103–167` on the Linux/NVIDIA training machine. The upstream vendored README's PyPI/older-CUDA instructions are not the project recipe.

The Mac has no project `.venv_isaacsim` or matching Isaac installation. An existing `/tmp/evolve-hand-audit-venv/bin/python` provides Python 3.14 with NumPy, SciPy, PyTorch, pytest, and viewer dependencies. It is useful for source-level CPU checks, but is outside the declared Python range and is not a validated training environment. Other local Python environments are not substitutes for the project's pinned stack either.

What can be done locally:

- Sampling, mutation, neutral walks, population I/O, geometric diagnostics, and CPU unit tests.
- Static images and browser-based hand inspection with the appropriate viewer dependencies.
- Analysis of population snapshots and reward tables once the team provides them.

What needs the training machine:

- Isaac/PhysX rollout validation and GPU regression checks.
- Policy training, checkpoint evaluation, camera captures, and co-evolution.
- NCCL and GPU throughput profiling. Some no-simulator benchmarks still explicitly require CUDA.

The newest SLURM scripts hard-code the existing cluster checkout, virtual environment, resources, and online W&B defaults. Do not run them unchanged on this Mac. Even the runner's dry-run path first enters the hard-coded directory and activates that environment.

For a reproducible baseline, request the exact starting population file, all generation snapshots and selection records, per-design reward tables, saved task/agent configuration, checkpoints including normalizer statistics, and training logs. Preserve population files by content/hash, not only generation seeds: the same seed after a mutation-code change can produce a different population.

The clone includes reference geometry but no training checkpoints, population JSONs, or run data. Dropout-evaluation mesh symlinks also target an absolute cluster path and are broken locally. The external DexToolBench data and some plotting dependencies are absent.

There is no root CI workflow. The nested vendored workflow is not a project-wide test pipeline. The development extra omits `pyflakes`, which one existing static physics test imports. The old archived scripts include targets that no longer exist.

One repository-hygiene issue must be addressed before any future commit: `.env` and `.env.local` are not covered by the current `.gitignore`. No such files were found in the checked scope, and no commit was made. Updating the root ignore file would be a small, explicit exception to the `hand_sampler`-only implementation boundary rather than a silent change in this investigation.

## 11. Correctness findings relevant to the next experiment

These findings were reproduced with CPU probes or traced through the active configuration. They were not repaired during this investigation. The effect on historical RL results remains unmeasured because the corresponding checkpoints, normalizer statistics, and runs are absent.

### A. Observation normalization changes the joint-existence mask

The active transformer configuration enables input normalization. The vendored model normalizes the observation before `JointTransformerNet._trunk` thresholds `joint_enabled > 0.5`. That field encodes categorical existence, so its normalized value is not a valid Boolean mask.

A CPU probe wrapped the actual transformer in the actual vendored `ModelA2CContinuousLogStd`, replacing only the Kit-dependent robot-spec lookup with a synthetic spec of the same dimensions. With 4,096 all-enabled observations, each enabled value became approximately `0.010936406`; all 122,880 real joint tokens were then marked invalid for attention-key masking and value pooling. In a slot present 90% of the time, normalized enabled values were approximately `0.333411`, also below the mask threshold. Action outputs remained finite, so this failure is silent.

This does not switch off the hand actuators or remove each token's own residual features. Tokens can still read the global token and produce hand actions. The failure removes the intended access to those hand tokens as attention keys and omits them from pooled value input. That distinction is important when interpreting why a policy can still learn despite the masking error.

Relevant code: `coevolution/cfg/train/PoseReachJointTransformerSAPG.yaml:97`, `third_party/rl_games/rl_games/algos_torch/models.py:51–56,264–272`, and `coevolution/networks/joint_transformer.py:450–468`.

This needs policy-owner coordination before interpreting new morphology experiments. Preserve categorical existence outside normalization or otherwise construct the mask from unnormalized data, then test the complete actor/critic path. The current block-mask tests do not exercise this integration.

### B. An active object-relative feature mixes local and world coordinates

`build_observations` constructs object keypoints in environment-local coordinates but subtracts a world-space palm/EE position to form `keypoints_rel_ee`. The active September 17 configuration includes that field in `state_list`, and the runner uses `obs_list=state_list`, so it reaches the actor rather than being an unused debug field.

A CPU probe executed the actual observation function extracted from source with synthetic tensor inputs. Translating an otherwise identical scene by environment origin `(1.2, -2.4, 0)` changed its first relative keypoint from `(0.11, 0.11, -0.09)` to `(-1.09, 2.51, -0.09)`. At origin `(12, -24, 0)`, the field clipped to `(-10, 10, -0.09)`. Correctly formed per-joint and goal-relative fields remained invariant.

Relevant code: `isaacsimenvs/pose_reaching_6d/obs_utils/observations.py:339–350,388–391`, `coevolution/cfg/task/PoseReach.yaml:75–80`, and `experiments/decentralized_control/scaling_laws/run_rank.sh:114`.

This introduces a dependency on where an otherwise identical environment is placed in the parallel simulation. Correcting it is outside a sampler-only change and should be coordinated with the simulator owner. A rigid-translation invariance regression should cover the observation assembly.

### C. Six-joint fingers report the wrong fingertip location

Population robot specs identify `f*_link5` as each fingertip body. For shorter fingers, padding bodies are placed at the true endpoint. When all six segments are real, however, `f*_link5` is the last joint's body origin, one final segment proximal to the fingertip. The simulator supplies zero offsets for generated hands and therefore uses that origin for fingertip features and reward-related distances.

A valid six-segment finger with 30 mm links produces a 30 mm tip error. In general, the error is the final link length, which can range from 15 to 80 mm under the grammar. This creates a topology-dependent error, not a constant reference offset.

Relevant code: `hand_sampler/build.py:299–314`, `hand_sampler/robot_spec.py:332`, `isaacsimenvs/pose_reaching_6d/reset_utils/reset.py:160–164`, and `isaacsimenvs/pose_reaching_6d/obs_utils/observations.py:211–215,327–337`.

A sampler-side fix must preserve a consistent definition of fingertip across the full envelope and the downstream zero/nonzero-offset behavior. Add a test for every chain depth, particularly six, rather than relying only on shallow seed hands.

### D. Collision validity depends on finger tuple order

`design_space.segment_distance` mishandles a degenerate second segment, a case used when a short capsule is represented as a sphere. In a legal two-finger example with 80 mm and 15 mm links, the validator reports approximately 40.697 mm separation and accepts the hand. Reversing the finger tuple reports the true 7.5 mm separation and rejects it against the 20 mm combined-radius threshold. The physical geometry is unchanged.

Relevant code: `hand_sampler/design_space.py:374–403` and the capsule checks in `hand_sampler/validate_design.py`.

This belongs in the initial sampler correctness pass. Distance symmetry, degenerate point/segment cases, and permutation-invariant validation are useful regression properties.

### E. Axis wrapping can make a small grid mutation geometrically large

The axis mutation wraps theta modulo pi without adjusting the nonzero rest offset. For a 50 mm link with a 60-degree offset, a normal 150→165-degree step moves the tip by 11.30 mm. The wrapped 165→0-degree step moves it by 85.86 mm, whereas the corresponding unwrapped 165→180-degree step again moves it by 11.30 mm.

Relevant code: `hand_sampler/mutate_design.py:223` and its theta canonicalization.

Axis sign, rest offset, and joint-coordinate semantics need consistent canonicalization. This is directly relevant to the claim that a mutation is small or local. Increasing step size before resolving this discontinuity would mix two different effects.

### F. High survivor fractions produce incorrect offspring records

`next_generation` produces a child for every survivor, truncates the resulting population, but does not truncate its child records. With `N=8` and `keep=6`, the output contains two children but records six. The default 50% survival case is unaffected. The suggested `KEEP=768` experiment for 1,024 hands is affected.

Relevant code: `hand_sampler/evolve.py:52` and the following generation assembly.

Generate exactly the requested number of children and keep their records aligned with output slots. Make parent allocation explicit when survivors outnumber offspring; otherwise truncation favors the leading survivor ranks.

### Validity and physical-model limitations to keep separate from these defects

- The validator does not reject all finger–palm penetration. A three-link, 30 mm-per-link finger with successive 90-degree offsets can fold into the palm while returning no errors. This needs an explicit palm-clearance policy, including intentional mount/adjacent-body overlap exceptions.
- Passing the grammar checks does not establish collision-free motion through the full joint range, motor packaging, transmission feasibility, structural strength, wiring, or thermal limits.
- Grammar/validator/token geometry uses a 10 mm radius, while the authored collision radius is approximately 10.1657 mm. A fresh two-link example passes with a nominal 0.10000 mm surface gap but has a 0.065722 mm overlap under the builder's capsule geometry. References: `hand_sampler/design_space.py:31`, `hand_sampler/robot_param_constants.py:350–356`, and `hand_sampler/build.py:435–446`.
- Population loading checks serialization structure but does not enforce all generated-grammar validity conditions. Imported measured hands may intentionally exceed those conditions, so introduce an explicit validation mode rather than imposing the generated grammar blindly on every input.
- Tiny ghost-body mass does not alone imply negligible ghost-body inertia. Each ghost has diagonal inertia 1e-6 kg m². Five padded ghosts contribute 5e-6 kg m² intrinsic inertia, approximately 13.2% of a single 50 mm real segment's inertia about its base before actuator armature. This is an analytical modeling discrepancy, not a measured control effect. Check padding together with armature and simulator dynamics before describing the envelope as physically cost-free.
- Fixed/reference robot bodies receive explicit gravity/contact settings during conversion; generated bodies are authored later and do not explicitly receive all the same settings. This source-level asymmetry needs an Isaac runtime check before drawing conclusions about effective physics differences.

These are reasons to define the physical scope of the grammar explicitly. They do not establish that a particular proposed new morphology will perform better.

### Interrupted sampler experiments need a separate resume check

Static inspection found that `hand_sampler/experiments/run.py:74–109` flushes statistics each generation but writes the population/RNG checkpoint only at the end of an invocation. An interruption can therefore leave statistics ahead of the saved state; resuming appends duplicate generation rows rather than reconciling them with that state. The clean staged-resume test passes, but interrupted resume was not crash-injection tested during this investigation. Treat this as a high-confidence static finding rather than a reproduced interruption.

## 12. Validation and practical starting commands

Fresh checks completed in the existing local CPU environment:

- `hand_sampler/tests`: **126 passed**, with seven SciPy Euler-conversion gimbal-lock warnings.
- Joint-mask and SHARPA physics-surface suites: **29 passed**, with one gimbal-lock warning.
- Total existing tests across those separate runs: **155 passed**.
- Shell syntax checks: all **74** experiment `.sh` and `.sub` files passed `bash -n`.
- The stdlib-only scaling-study trial query executed successfully.
- Focused CPU probes reproduced the normalization, coordinate-frame, fingertip, distance, mutation-wrap, and selection-record cases described above. These probes exercise selected real code paths without launching Isaac.

The passing tests do not cover the failure cases above. No simulator scene, GPU policy-training run, held-out checkpoint evaluation, or physical experiment was run. No dependency installation was performed.

From the repository root, the existing temporary interpreter can be used for a starting CPU session:

```bash
cd "/Users/mfpet/Documents/github/research/Evolve to Generalize/gen_mechanics"
PY=/tmp/evolve-hand-audit-venv/bin/python

PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -p no:cacheprovider hand_sampler/tests -q
PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -p no:cacheprovider coevolution/networks/tests isaacsimenvs/pose_reaching_6d/tests -q

# Current population-generation entry point; the README's hand_sampler.population is obsolete.
"$PY" -m hand_sampler.population_io gen_s0_n64 --out /tmp/martin-seeds-2026-09-23.json
"$PY" -m hand_sampler.viewer png --population /tmp/martin-seeds-2026-09-23.json --designs 0-8 --out /tmp/martin-seeds-2026-09-23.png
```

The test suites were executed in this investigation. The generation/viewer lines are starting commands using the current entry points, not a claim that those output paths were created in this pass. The temporary Python 3.14 environment may not survive system cleanup; create a durable Python 3.11 CPU environment when beginning development rather than treating this audit environment as the training setup.

## 13. Suggested first work package and ownership split

Start with a bounded sampler change: reproduce the geometric and bookkeeping cases in tests, correct those cases, then add a 40-generation mutation audit with explicit provenance. Keep the 5×6 envelope and observation contract stable initially. This provides a trustworthy baseline for deciding whether to change mutation magnitude, scope, topology moves, or physical constraints.

Before spending a new training budget, coordinate the observation-mask and coordinate-frame findings with the policy/simulator owners. Confirm the baseline budget and evaluation protocol. Those issues can confound a hardware comparison even when the sampler itself is correct.

The first research deliverable should answer three concrete questions: how much physical change one accepted mutation creates, how much distinct geometry remains after 40 generations, and which useful reference designs the representation or validity rules exclude. Report these separately from ancestry and task performance. Then choose the smallest grammar intervention supported by those measurements.

The earlier [21 September investigation](investigation-2026-09-21/README.md) contains additional historical context and a saved reproduction script for several previously identified cases. This report freshly checks the project state, adds the new integration/geometry findings and the small neutral-walk diagnostic, and does not modify that earlier record.
