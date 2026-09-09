# Policy size scaling on two GPUs

The current study uses a **shared actor-critic transformer**. A single trunk
processes the privileged observation fields (the former critic input) and
feeds the action and value heads. `obs_list` equals `state_list`; the existing
observation pipeline is used with observation noise and delay disabled.
The
privileged-state central critic is disabled (`central_value_config=null`).

The launcher defaults to `NCCL_P2P_DISABLE=1`: a minimal 32-element broadcast
timed out with default NCCL on the B0/B1 GPU pair of portal-compute-02 but
passed with P2P disabled. This workaround affects communication performance;
all trials use it consistently. `nccl_probe.py` reproduces the diagnostic.
The earlier asymmetric jobs were cancelled; their logs remain available and
are excluded by the new `scaling_laws_shared_privileged_v1` study ID.

This is an empirical search for the best reward per elapsed hour, not a claim
that parameter count alone determines learning. No jobs are submitted by these
files. Each job uses two GPUs on one node.

## Stage 1: choose throughput settings separately for each model

There are eight model files: `d32_l1`, `d32_l4`, `d64_l1`, `d64_l4`,
`d128_l1`, `d128_l2`, `d256_l1`, and `d256_l4` (all `.sub`). Each defaults to an
five-task Slurm array (`1,3,4,6,7%1`), with **one task active per array**.
Tasks 0, 2, and 5 (global minibatch 16384) were retired at user request;
unfinished instances were cancelled and completed measurements retained for
reference, but excluded from winner selection. Original task IDs stay stable:

| Array task | Environments/GPU | Global environments | Global minibatch | Minibatch/GPU |
|---|---:|---:|---:|---:|
| 0 | 6144 | 12288 | 16384 | 8192 |
| 1 | 6144 | 12288 | 32768 | 16384 |
| 2 | 12288 | 24576 | 16384 | 8192 |
| 3 | 12288 | 24576 | 32768 | 16384 |
| 4 | 12288 | 24576 | 65536 | 32768 |
| 5 | 24576 | 49152 | 16384 | 8192 |
| 6 | 24576 | 49152 | 32768 | 16384 |
| 7 | 24576 | 49152 | 65536 | 32768 |

Every trial trains the real simulator and shared actor-critic for 60 epochs with
seed 123. Discard the first 20 epochs for throughput selection. The score is
global environment transitions / training-loop elapsed time, including
rollouts and PPO updates (`fps total`), not synthetic actor samples/sec.
Compilation/startup is excluded from that score but remains visible in logs.
Select the highest harmonic-mean SPS over the remaining epochs. Repeated
trials are combined using their median score. OOM/failed trials do not win;
unfinished trials and trials with insufficient timing samples block selection.
Inspect failed logs: a software failure is not proof that a configuration
cannot fit. The grid finds the best **measured** setting, not a global optimum.

```bash
# Start with one smoke trial; verify two devices, critic initialization,
# finite losses and global frames before launching the full study.
MAX_EPOCHS=3 sbatch --array=0 experiments/decentralized_control/scaling_laws/d32_l1.sub

# Each file runs its own eight-trial search.
sbatch experiments/decentralized_control/scaling_laws/d32_l1.sub
sbatch experiments/decentralized_control/scaling_laws/d128_l2.sub

python experiments/decentralized_control/scaling_laws/study.py report
python experiments/decentralized_control/scaling_laws/study.py best --model d128_l2
```

The three-epoch smoke run is excluded from throughput ranking. Submitting all
eight arrays means 64 trials and up to 16 GPUs concurrently; `%1` limits each
array, not the whole study. Submit in batches if desired. Rerun the top two
trial IDs for each model to check noise before selecting (for example
`sbatch --array=2,3 ...`). Use a new `STUDY_ID` after changes to the code or
controlled settings so unrelated runs are not combined.

## Stage 2: learning with each model's selected settings

### Optional large-batch pilots

The default eight-task arrays are unchanged. Additional task IDs are available
through an explicit `--array` override on the same model files:

| Task | Environments/GPU | Global minibatch | Minibatch/GPU |
|---|---:|---:|---:|
| 8 | 24576 | 131072 | 65536 |
| 9 | 49152 | 131072 | 65536 |
| 10 | 49152 | 262144 | 131072 |

Start with task 8 for d32 L1 and d128 L1, sequentially, after their baseline
sweeps. Inspect sampled GPU peaks, scheduler RAM accounting, errors, and SPS
before submitting tasks 9 and 10. These larger-env probes are prepared, not
automatically submitted. Jobs remain two GPUs, four CPUs, and 128000 MB RAM.
Do not extrapolate simulator memory linearly from the smallest-env trials.

Completed optional probes participate in `best`; they are not required for
models without pilot submissions. Any submitted probe must be added to its
model's training dependency so selection cannot run before that probe ends.
The staged submission manifest records those extra dependencies and job IDs.
Checkpoint retry handling is unchanged. SPS excludes checkpoint and startup
time; learning comparisons must still consider actual elapsed time.

Once the five retained baseline trials have completed or failed, each file can launch three
long learning runs using its model's measured throughput winner:

```bash
PHASE=train sbatch --array=0-2%1 experiments/decentralized_control/scaling_laws/d128_l2.sub
```

Seeds are 100/200/300. The Slurm limit is 24 hours; checkpoints are saved every
100 epochs and on best reward. Compare common elapsed-time checkpoints and
reward curves, not just whichever checkpoint happens to be last. Periodic
checkpoints provide recovery if Slurm terminates before a final save. To
override the selected pair explicitly, set both `NUM_ENVS_PER_GPU` and
`GLOBAL_MINIBATCH` with `PHASE=train`. These overrides are not used in tuning.

## Controlled architecture and learning settings

- Shared trunk width/depth vary. One head, explicit attention, FF multiplier 2,
  serial pre-LayerNorm blocks with affine parameters, final LayerNorm.
- Shared hand head `[64]`, arm head `[256,128]`; no hybrid context MLP.
- Value head `[512,256]` reads pooled joint tokens, the global token, and raw
  global policy inputs from the same trunk. No separate critic forward/update.
  Policy and value losses both train the trunk using privileged state.
- Compiled networks; FP16 autocast and GradScaler; rl_games gradient all-reduce
  (not the synthetic benchmark's DDP wrapper).
- Initial actor/critic LR 1e-4, existing adaptive actor schedule, horizon 16,
  two PPO mini-epochs, six SAPG exploration blocks **per rank**.
- SHARPA/IIWA, 100 assets/type, same task and disabled domain randomization as
  the tuned reference launcher. Video disabled to keep timing consistent.

Changing environment count also changes rollout size and samples per SAPG
block. Changing minibatches changes optimizer-step count and gradient noise.
This intentionally compares each architecture's throughput-tuned training
configuration, not model size in isolation. Keep reward vs global transitions
alongside reward vs elapsed time, and investigate minibatch/LR effects if a
throughput winner learns poorly. No LR retuning or linear LR scaling is assumed.

## Artifacts and validation

Outputs: `debug_outputs/scaling_laws/<unique run>/`. `result.json` records the
configuration/status/timing score. Each rank has a separate Hydra directory;
rank 0 writes TensorBoard metrics/checkpoints. Torchrun logs, GPU monitoring,
git commit and tracked diff are retained. Wandb is off by default; opt in with
`WANDB_ACTIVATE=1`. Existing reward summaries are rank-0 rollout statistics;
frame/SPS counters are global, so do not interpret them as cross-rank evaluation.

The training entry point routes Kit/simulation/policy to the local rank's
GPU. The RL loop uses torchrun's NCCL rendezvous and initializes both actor
and critic identically across ranks. These are necessary corrections to the
old multi-GPU training path, which forced cuda:0 and omitted critic sync.
The asymmetric two-GPU smoke run passed; the shared version gets its own smoke
run before tuning is released. Synthetic profiling did not validate this RL path.

Run `DRY_RUN=1 bash .../d128_l2.sub` to inspect the launch without starting Kit.
See [HYPOTHESES.md](HYPOTHESES.md) for model-specific hypotheses and provisional SPS ranges.

## Status, 2026-09-09

Stage 1 finished for all eight models. Measured winners (harmonic-mean
`fps total`, epochs 20-60, global minibatch >= 32768):

| Model | envs/GPU | global minibatch | SPS |
|---|---:|---:|---:|
| d32 L1 | 24576 | 65536 | 135k |
| d32 L4 | 24576 | 65536 | 124k |
| d128 L2 | 24576 | 65536 | 106k |
| d64 L1 | 12288 | 65536 | 104k |
| d64 L4 | 24576 | 65536 | 103k |
| d256 L1 | 12288 | 65536 | 95k |
| d128 L1 | 6144 | 32768 | 89k |
| d256 L4 | 24576 | 65536 | 45k |

Read these as a **within-model** ranking only. Four or five of these
two-GPU jobs shared portal-compute-02/04 for the whole sweep, so the
cross-model column carries an unknown amount of co-tenancy. d128 L1 is
the visible symptom: it scores below both d128 L2 and d256 L1, and it is
the one model repeatedly displaced to the other node.

The task-8 pilot (24576 envs, global 131072) **completed and lost** for
d32 L1: 93k against 135k at global 65536. Hypothesis 1's monotonic
reading is rejected at that size; batch has passed the useful point. The
d128 L1 pilot was cancelled six minutes in and never measured.

Stage 2 started for four models at seed 100 and was cancelled after
40-100 minutes of a 24 hour budget. Best reward at matched global
transitions, before the cancellation:

| run | total | @148M | @178M | @250M |
|---|---:|---:|---:|---:|
| d64 L1 | 251M | 237 | 289 | 552 |
| d32 L4 | 424M | 151 | 183 | 279 |
| d32 L1 | 179M | 149 | 149 | - |
| d64 L4 | 148M | 153 | - | - |

Every curve sits near 150 for its first ~150M transitions and then
climbs. d32 L1 and d64 L4 were stopped before that point, so nothing
here separates the models on learning. Their `result.json` files still
say `running`.

## Depth sweep at fixed width: `depth_d64/`

`depth_d64/l{1,2,3,4}.sub` is the follow-up that is actually running:
d_model 64 throughout, depth 1/2/3/4, seed 100, one job each. Every run
uses 12288 envs/GPU and global minibatch 114688, so an epoch is the same
number of transitions and the same number of optimizer steps in all
four, and only depth varies. The settings are deliberately **not** each
model's Stage 1 throughput winner; matched data per epoch is worth more
here than the last few percent of SPS. See `submitted_depth_d64.json`.

`run.sh` now requires the minibatch to divide the SAPG-augmented rollout
only. The separate critic rollout it also used to check no longer exists
under `central_value_config=null`. Every pair in the table above still
passes. The rule matters: at 12288 envs/GPU a global 114688 gives four
equal 57344-row minibatches per rank, where the YAML's historical 98304
would give three of 49152 and one of 81920, because `PPODataset` hands
the remainder to the last minibatch.
