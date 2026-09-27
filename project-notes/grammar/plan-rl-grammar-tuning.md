# Plan: tune the hand grammar with RL on the CSAIL cluster, using a stationary-hand reorientation task

Status: for approval. Branch `martin/hand-grammar` (clean, representation settled). Supersedes the representation-check plan (complete).

## Context

Martin wants to tune the grammar for evolution (diversity, performance), which needs RL experiments. Those will run on the lab's CSAIL SLURM cluster, and he wants to be a good cluster citizen.

**Team direction (meeting of 2026-09-25 and the investigation summaries of the older Notion notes):**
- Drop the arm and run a stationary hand doing in-hand reorientation, which is faster and has no arm confounder.
- Aim for a controller that works zero-shot on commercial hands: Wuji, Tesollo, DClaw, Allegro, XHand.
- Evolution should optimise diversity, to fill the controller's blind spots.
- Define a MAP-Elites phenotype: fingers, joints, two fitness functions.
- Carry weights across generations; don't re-initialise.
- Watch whether co-evolution degrades control of known hands (SHARPA collapsed in the last runs).

**Findings:**
- **No reorientation task exists on any branch.** `git ls-remote` matches the cached refs; the newest collaborator push (Sept 24) is analysis only.
- **Resume is supported mid-run.** `train.py` resumes weights, optimizer, epoch and curriculum, and `last/model.pth` is written every 3 epochs. The old chain scripts resume from `nn/` checkpoints that can be hours stale.
- **No grammar-to-simulator bridge exists.** Hands reach Isaac only as the old sampler format (a box palm with three faces, a 5×6 slot envelope, the palm fused to the arm link) or as one fixed URDF.
- **The old protocol was expensive:**
  - about 14 days on 2 GPUs for the co-evolution arm, plus up to 480 GPU-hours for the baseline;
  - self-resubmitting chains that are unsafe on preemptible tiers;
  - 1–3-day limits sized to fill the walltime;
  - 128 GB RAM for 2 GPUs against a 35 GB measured peak;
  - W&B hard-coded to Kushal's entity, and outputs inside the repo tree.
- **Environment:** this machine has an RTX 4090, and you installed the `cluster` tool and `slurm` skill. The cluster rules are in `~/research/slurm/CLAUDE.md`.

**Decisions:**
- Martin builds the reorientation task now, as a new package on his branch.
- The first phase is capped at 200 GPU-hours on the main tier, with at most 4 GPUs at once.
- Isaac Lab gets installed locally on the 4090, so debugging never uses cluster GPUs.
- Use the meeting summaries we have.

## Phase 0: access and environments (no cluster GPU time)
1. **Cluster access (mostly done by you, 2026-09-26).**
   - Done: krb5 is installed with a valid ticket, the `Host cluster` ssh block is in place, the system ssh config enables Kerberos (GSSAPI), and `cluster` and the `slurm` skill are installed with local paths.
   - Remaining: you run `ssh -fN cluster` in your own terminal (second factor, once a day).
   - Then I run `cluster check` and `cluster setup` (idempotent).
   - I never attempt logins myself.
2. **Local stack on the 4090.**
   - User-level `uv`, then `.venv_isaacsim` with Python 3.11 managed by uv, following the pinned README recipe: torch 2.7.0+cu126, isaaclab 2.3.2.post1, isaacsim 5.1.0.0, vendored rl_games editable. About 15 GB.
   - Verify with the README's import check, then a 2-minute pose-reach smoke (512 envs) proving the stack runs here.
3. **Cluster stack.**
   - `cluster sync` the repo to `/data/pulkitag/users/mpeticco/code/gen_mechanics`.
   - Build the same venv in a short `--tier cpu` job, not on the login node, since it is a 15 GB install.
   - W&B entity `mpeticco`, set via env, without editing collaborators' scripts.
   - All run outputs go under `/data/pulkitag/users/mpeticco/runs/`.

## Phase 1: stationary-hand in-hand reorientation task (new package; `pose_reaching_6d` untouched)
New `isaacsimenvs/inhand_reorient/`, task id `GenMech-InHandReorient-Direct-v0`. It subclasses and reuses `pose_reaching_6d` utilities so curriculum checkpointing in `rlgames_utils.py` keeps working. The only edit outside the new package is one import line in `isaacsimenvs/__init__.py`.
- **Scene:**
  - a palm-up hand fixed to the world, with no arm and no table;
  - `RobotSpec` variant with no arm (a new hand-only spec in the package; `hand_sampler/robot_spec.py` itself stays as is);
  - objects start as procedural primitives (cube, cuboids, cylinders; no downloads).
- **Task:**
  - object spawned just above the palm; goal = a target orientation (a random orientation or a delta rotation);
  - success = `quat_error_magnitude` ≤ tolerance for N steps;
  - reward = rotation-error progress, a goal bonus, action and velocity penalties, and a drop penalty;
  - termination on a drop (object farther than a set distance from the palm) or timeout;
  - curriculum on the angular tolerance.
- **Observations:** per-joint tokens as today, in the palm frame, plus object rotation relative to the goal.
- **Hands for calibration:** SHARPA (hand only), cut at `left_hand_C_MC` using our `load_urdf(hand_root)` and `to_urdf` adapters. Commercial hands load from their own URDFs through the existing fixed-robot converter (`_convert_fixed_robot`, `fix_base=True`).
- **Resume:** resume mode from `last/model.pth`, tested locally by killing and restarting a run.
- **Acceptance (local, on the 4090):** a SHARPA-only policy learns to reorient a cube above chance within a short local run, and a killed run resumes with the same curriculum state.

## Phase 2: grammar to simulator adapter
New Isaac-side module, `isaacsimenvs/inhand_reorient/scene/author_grammar.py` (it stays out of `hand_sampler`, which never imports the simulator):
- **Envelope:** authors a padded envelope directly from `derive()` output.
  - The envelope is designed for the grammar: up to 5 digits × 6 joints, plus 2 palm-joint slots, with revolute joints and one motor per joint.
  - It reuses `grammar/envelope.fits_envelope` for admission.
- **Geometry:**
  - capsules per body with the per-hand radius;
  - palm cells as convex colliders;
  - ghost slots as today.
- **Per-design tables:** token boxes from capsule geometry, limits, validity masks, fingertip frames.
- **Populations:** stored as derivation JSON (`derive.derivation_to_json`) with sha256 provenance.
- **Checks:**
  - authored joint frames equal `grammar.fk.forward_kinematics` at zero pose (checked in Isaac locally);
  - grammar-projected commercial hands (from E13) load through the same path, so the controller sees them in-distribution.

## Phase 3: experiments (≤ 200 GPU-hours, main tier, ≤ 4 GPUs in flight)
- **E-R0, calibration (about 25 GPU-h).**
  - Train from scratch on single hands: SHARPA, Allegro, DClaw, XHand, Wuji, Tesollo DG-5F, plus one grammar seed hand. 1 GPU each.
  - First a 20-minute smoke per hand, then one run.
  - Measure success versus GPU-hours. This fixes honest time limits and run sizes, and it gives the meeting's "baseline: learn from scratch on any hand" numbers.
  - A 3-finger Tesollo URDF is not available locally; using one needs a download you approve.
- **E-R1, which grammar distribution trains a controller that generalises (about 150 GPU-h).**
  - One shared policy per grammar variant, trained on a fixed sampled population with no evolution. Variants:
    1. old-sampler-like (G_SERIAL, no palm joints);
    2. grammar 0.5 default, revolute only;
    3. insertion prior;
    4. atlas-informed priors: mount offsets, short knuckles, small bends, limits within real ranges.
  - 2 seeds each. Size and time come from E-R0.
  - Evaluate zero-shot on the commercial hands (original URDFs) and on held-out grammar samples.
  - Metrics: success rate, final rotation error, per-hand retention.
  - This is the grammar-tuning signal, with evolution not yet confounding it.
- **Remaining budget (about 25 GPU-h):** reruns and failures.
- **Accounting:** GPU-hours tracked in `project-notes/grammar/STATE.json` from `cluster hist`. Stop at 200 and report.
- **Next phase (separate approval, not in this budget): E-R2.**
  - MAP-Elites (phenotype: fingers × motors; fitness: reorientation success plus a second objective) versus truncation, on the best E-R1 variant, with weights carried across generations.
  - Track zero-shot commercial-hand success per generation (the SHARPA-collapse question), archive coverage, and complexity.

## Cluster etiquette (built into every job)
- **Main tier by default:**
  - 1 GPU per run, 24–48 GB RTX cards. The `cluster` default suits this; Isaac Sim needs RT cores, so avoid A100/H100.
  - 8 CPUs; RAM of 48 GB set from the measured peak rather than 128 GB.
  - At most 4 GPUs at once.
- **Honest limits:** calibrated runtime + 20%.
- **Order of submission:** `--dry-run`, then `--check`, then a 20-minute smoke, then the real run.
- **No self-resubmitting chains:**
  - sweeps go as one throttled job array, with `--dependency` for stages;
  - preemptible tiers only after resume from `last/model.pth` is verified, and only with your OK.
- **Files:** outputs, caches and checkpoints go under `/data/pulkitag/users/mpeticco`. Node-local Kit and inductor caches are removed at job end.
- **Other rules:**
  - no compute on the login node;
  - poll at most once a minute, from a bounded wait;
  - cancel anything wrong or no longer needed;
  - never touch other users' jobs or files, or other people's branches;
  - keep limits clear of the 2026-10-05 08:00 to 10-06 02:00 maintenance window.

## Files
- New:
  - `isaacsimenvs/inhand_reorient/` (env, cfg, scene, `author_grammar.py`, tests);
  - `experiments/csail_reorient/` (the exact `cluster submit` commands, run manifests, a README);
  - `project-notes/grammar/plan-rl-grammar-tuning.md` and experiment results.
- Edited: `isaacsimenvs/__init__.py` (one import line).
- Untouched: `pose_reaching_6d`, `coevolution/` training logic, collaborators' branches.
- Reused:
  - `coevolution/train.py` (resume, per-design reward logging);
  - `pose_reaching_6d` observation, reward and reset helpers;
  - `scene_utils/assembly._convert_fixed_robot`;
  - `hand_sampler/grammar` (`derive`, `fk`, `geometry`, `envelope`, `adapters.urdf`, `projection`);
  - `grammar_bench/manifest.json` hands.

## Verification
- **Tests:** the existing suite stays green, plus new CPU tests for the adapter tables and envelope admission.
- **Local:**
  - pose-reach smoke on the 4090;
  - reorientation training on SHARPA learns above chance;
  - kill-and-resume restores the curriculum;
  - grammar designs author with FK-matching frames.
- **Cluster:** `cluster submit --dry-run` and `--check` for each job type, a 20-minute smoke, then E-R0.
- **Reporting:** after each phase, a summary in `project-notes/grammar/` with job IDs, GPU-hours used, W&B links and results.
