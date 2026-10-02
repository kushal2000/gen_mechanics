r"""MAP-Elites evolution-pilot driver (plan-rl-grammar-tuning.md's
"Revision, 2026-09-27", pilot E-R2'): one shared RL controller whose weights
carry across generations, evolving a `hand_sampler` grammar population
against a `archive.Archive` (digit_count x joint_count, 30 cells).

Runs entirely as a single Python process that shells out to
``coevolution/train.py`` once per generation (never self-resubmits to
SLURM -- the cluster job runs this driver once with an honest time limit,
per this branch's own rule). CPU-only itself (no ``isaaclab``/``torch``
import): it only needs ``hand_sampler`` and this package's own
``scene.grammar_envelope``/``scene.population_file`` (also CPU-only) to
build population files and re-derive designs; the Kit/GPU work happens
entirely inside the ``coevolution/train.py`` subprocess.

CLI:
    python -m isaacsimenvs.inhand_reorient.evolution.driver \
        --variant G_V1 --seed 0 --generations 20 --designs 64 \
        --probes allegro_right,dclaw,sharpa_left_on_iiwa14,leap_right \
        --num-envs 4096 --epochs-per-gen 200 --fitness train_tail \
        --run-dir outputs/evolution_pilot/run0

Resume: rerun the exact same command with the same ``--run-dir``; the
driver loads ``<run-dir>/state.json`` (written atomically after every
COMPLETED generation) and continues from the next one.

Robustness (I41): a generation counts as completed only if ``train.py``
exits 0 AND leaves a checkpoint whose tensors are all finite and whose
GradScaler has not collapsed to 0. Otherwise it is retried once from the
same carried checkpoint (with a different seed); a second failure stops the
driver with a nonzero exit, ``state.json`` still at the last good
generation, and nothing of the failed generation scored or archived. The
checkpoint carried into a generation is written to
``gen_<k>/carry_checkpoint.pth`` after a finiteness check, with the policy's
log-std optionally reset or clamped (``--sigma-on-carry``) and the
GradScaler reset to its initial state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hand_sampler.grammar.derive import (
    EVOLUTION_OPERATORS,
    Derivation,
    VariationImpossible,
    derivation_from_dict,
    derive,
    sample_derivation,
    vary,
)
from hand_sampler.grammar.distributions import Distribution
from hand_sampler.grammar.kinematics import ModelError
from hand_sampler.grammar import variants as grammar_variants

from ..scene import grammar_envelope as ge
from ..scene import population_file as pf
from . import archive as arch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PY = REPO_ROOT / "coevolution" / "train.py"
TASK_ID = "GenMech-InHandReorient-Direct-v0"
AGENT_ENTRY_POINT = "rl_games_sapg_cfg_entry_point"

STATE_SCHEMA = "evolution_driver_state/0.1"
DEFAULT_HORIZON_LENGTH = 16  # coevolution/cfg/train/InHandReorientSAPG.yaml's `horizon_length`
DEFAULT_MINIBATCH_CAP = 16384  # ... its `minibatch_size`

# Task profile (env.task_profile) and the agent configs that go with it. The
# driver defaults to the LEGACY profile and the SAPG agent, so a run started
# before the profile flag existed resumes unchanged; the profile is always
# passed explicitly, so a later change of the env's own default cannot
# silently change a resumed run.
TASK_PROFILES: Tuple[str, ...] = ("legacy", "isaaclab_repose", "anyrotate")
DEFAULT_TASK_PROFILE = "legacy"
SAPG_AGENT_ENTRY_POINTS: Tuple[str, ...] = (
    "rl_games_sapg_cfg_entry_point", "rl_games_sapg_pop_cfg_entry_point")
REPOSE_POP_AGENT_ENTRY_POINT = "rl_games_repose_pop_ppo_cfg_entry_point"  # InHandReposeIsaacLabPopPPO.yaml
ANYROTATE_POP_AGENT_ENTRY_POINT = "rl_games_anyrotate_pop_ppo_cfg_entry_point"  # InHandAnyRotatePopPPO.yaml
# Agents whose YAML rolls out a different horizon than the SAPG configs' 16.
AGENT_HORIZONS: Dict[str, int] = {
    "rl_games_anyrotate_ppo_cfg_entry_point": 8,  # AnyRotate Table 5: rollout 8 steps
    ANYROTATE_POP_AGENT_ENTRY_POINT: 8,
}
PPO_MINIBATCH_CAP = 32768  # InHandReposeIsaacLabPPO.yaml's (NVIDIA's) `minibatch_size`
DEFAULT_PROBES: Tuple[str, ...] = ("allegro_right", "dclaw", "sharpa_left_on_iiwa14", "leap_right")


# --------------------------------------------------------------------------
# Variant resolution (I29/I30 G0 variants live in a registry a concurrent
# worker is still adding to -- resolve by name through it, never hard-code
# a list of variant names here).
# --------------------------------------------------------------------------


def resolve_variant(name: str) -> Distribution:
    registries = (
        getattr(grammar_variants, "NAMED_DISTRIBUTIONS", {}),
        getattr(grammar_variants, "G0_SCREEN_VARIANTS", {}),
    )
    for registry in registries:
        if name in registry:
            dist = registry[name]
            # G_FULL_SMALL-style entries are `(Distribution, operators)` pairs
            # for `vary`'s default operator pool -- the evolution pilot
            # always uses EVOLUTION_OPERATORS regardless, so just take the
            # Distribution half if that's what was registered.
            if isinstance(dist, tuple):
                dist = dist[0]
            return dist
    known = sorted(set(registries[0]) | set(registries[1]))
    raise ValueError(f"unknown --variant {name!r}; known variants: {known}")


# --------------------------------------------------------------------------
# Design bookkeeping
# --------------------------------------------------------------------------


@dataclass
class DesignMeta:
    """One design's identity within a single generation's population --
    parallel to (and matched with, by `source`) a `population_file.
    PopulationEntry`."""

    design_id: str
    founder_id: str
    parent_id: Optional[str]
    generation_born: int
    digit_count: int
    joint_count: int
    role: str  # "elite" | "offspring" | "immigrant" | "probe"
    source: str
    hand_id: Optional[str] = None


def _design_counts(design: "ge.EnvelopeDesign") -> Tuple[int, int]:
    digit_count = int(sum(1 for d in design.finger_digit_id if d is not None))
    joint_count = int(design.slot_valid.sum())
    return digit_count, joint_count


class IdMinter:
    """Monotonic id counter, persisted across resumes (`next_uid` in
    state.json) so ids minted after a resume never collide with ids minted
    before it."""

    def __init__(self, next_uid: int = 0) -> None:
        self.next_uid = int(next_uid)

    def mint(self, generation: int, role: str) -> str:
        uid = self.next_uid
        self.next_uid += 1
        return f"gen{generation}-{role}-{uid:06d}"


def sample_new_founder(
    dist: Distribution, driver_rng: np.random.Generator, max_tries: int = 20000,
) -> Tuple[Derivation, "ge.EnvelopeDesign"]:
    """A freshly sampled, `admit`-ted derivation (generation-0 founder, or
    an immigrant replacing a failed mutation) -- retries with fresh seeds
    (drawn from `driver_rng`, so the whole search is reproducible/resumable
    from the driver's own rng state) until one is admitted."""
    for _ in range(max_tries):
        seed = int(driver_rng.integers(0, 2**31 - 1))
        derivation = sample_derivation(seed, dist)
        try:
            model = derive(derivation)
        except ModelError:
            continue
        result = ge.admit(model)
        if not result.ok:
            continue
        design = ge.canonicalize(model)
        return derivation, design
    raise RuntimeError(f"could not sample an admitted design from this variant in {max_tries} tries")


def try_offspring(
    parent_derivation: Derivation, dist: Distribution, driver_rng: np.random.Generator, max_retries: int = 16,
) -> Optional[Tuple[Derivation, "ge.EnvelopeDesign"]]:
    """Up to `max_retries` attempts at `derive.vary(..., operators=
    EVOLUTION_OPERATORS)` + `grammar_envelope.admit`; `None` if none of them
    produced an admitted design (the caller then draws an immigrant
    instead, per the plan)."""
    for _ in range(max_retries):
        try:
            candidate = vary(parent_derivation, driver_rng, dist, operators=EVOLUTION_OPERATORS)
        except VariationImpossible:
            continue
        try:
            model = derive(candidate)
        except ModelError:
            continue
        result = ge.admit(model)
        if not result.ok:
            continue
        design = ge.canonicalize(model)
        return candidate, design
    return None


# --------------------------------------------------------------------------
# Population assembly
# --------------------------------------------------------------------------


@dataclass
class GenerationPlan:
    entries: List["pf.PopulationEntry"]
    metas: List[DesignMeta]


def build_generation_population(
    generation: int,
    n_designs: int,
    probe_hand_ids: Sequence[str],
    dist: Distribution,
    archive: "arch.Archive",
    driver_rng: np.random.Generator,
    minter: IdMinter,
    max_offspring_retries: int = 16,
) -> GenerationPlan:
    entries: List[pf.PopulationEntry] = []
    metas: List[DesignMeta] = []

    # Probes: fixed identity every generation, never enter the archive.
    for hand_id in probe_hand_ids:
        entry, status, reason = pf.projected_entry(hand_id)
        if status != "admitted":
            raise RuntimeError(f"probe hand {hand_id!r} is not admitted ({status}): {reason}")
        entries.append(entry)
        metas.append(DesignMeta(
            design_id=f"probe:{hand_id}", founder_id=f"probe:{hand_id}", parent_id=None,
            generation_born=0, digit_count=-1, joint_count=-1, role="probe",
            source=entry.source, hand_id=hand_id,
        ))

    slots_for_growth = n_designs - len(probe_hand_ids)
    if slots_for_growth <= 0:
        raise ValueError(
            f"--designs {n_designs} must exceed the number of probes ({len(probe_hand_ids)})"
        )

    elites = archive.elites()
    n_elite_slots = min(len(elites), slots_for_growth)
    if n_elite_slots < len(elites):
        # Only possible with a pathologically small --designs; keep the
        # highest-fitness elites so the population that no longer fits
        # loses its weakest cells first.
        elites = sorted(elites, key=lambda e: e.fitness, reverse=True)[:n_elite_slots]

    for e in elites:
        derivation = derivation_from_dict(e.derivation_dict)
        model = derive(derivation)
        source = f"arch:{e.design_id}"
        entries.append(pf.make_entry(source, derivation, model))
        metas.append(DesignMeta(
            design_id=e.design_id, founder_id=e.founder_id, parent_id=e.parent_id,
            generation_born=e.generation_born, digit_count=e.digit_count, joint_count=e.joint_count,
            role="elite", source=source,
        ))

    n_offspring = slots_for_growth - n_elite_slots
    for _ in range(n_offspring):
        if generation == 0 or not elites:
            derivation, design = sample_new_founder(dist, driver_rng)
            digit_count, joint_count = _design_counts(design)
            design_id = minter.mint(generation, "founder")
            role = "founder"
            founder_id, parent_id = design_id, None
        else:
            parent = archive.sample_parent(driver_rng)
            parent_derivation = derivation_from_dict(parent.derivation_dict)
            result = try_offspring(parent_derivation, dist, driver_rng, max_retries=max_offspring_retries)
            if result is not None:
                derivation, design = result
                digit_count, joint_count = _design_counts(design)
                design_id = minter.mint(generation, "off")
                role = "offspring"
                founder_id, parent_id = parent.founder_id, parent.design_id
            else:
                derivation, design = sample_new_founder(dist, driver_rng)
                digit_count, joint_count = _design_counts(design)
                design_id = minter.mint(generation, "imm")
                role = "immigrant"
                founder_id, parent_id = design_id, None

        source = f"arch:{design_id}"
        entries.append(pf.make_entry(source, derivation, derive(derivation)))
        metas.append(DesignMeta(
            design_id=design_id, founder_id=founder_id, parent_id=parent_id, generation_born=generation,
            digit_count=digit_count, joint_count=joint_count, role=role, source=source,
        ))

    return GenerationPlan(entries=entries, metas=metas)


# --------------------------------------------------------------------------
# Training subprocess + window polling
# --------------------------------------------------------------------------


def _minibatch_and_block_size(num_envs: int, horizon_length: int) -> Tuple[int, int]:
    """`(minibatch_size, expl_coef_block_size)` overrides so training never
    errors for a `--num-envs` other than the yaml's own default 4096:
    `expl_coef_block_size` must divide `num_envs` (SAPG's own constraint,
    see InHandReorientSAPG.yaml's comment) -- setting it equal to
    `num_envs` always keeps exactly 1 exploration block, matching
    `evaluate_population.py`'s own default assumption. `minibatch_size`
    must divide `num_envs * horizon_length` (the rollout batch); the
    yaml's default (16384) only happens to divide the yaml's own default
    4096*16 -- cap at the yaml default but fall back to the exact batch
    size (always a trivial divisor of itself) whenever that is smaller."""
    batch_size = num_envs * horizon_length
    minibatch_size = batch_size if batch_size <= DEFAULT_MINIBATCH_CAP else DEFAULT_MINIBATCH_CAP
    return minibatch_size, num_envs


def agent_horizon(agent_entry_point: str, default: int) -> int:
    """The rollout horizon the agent's YAML uses (``AGENT_HORIZONS``), else
    ``default`` (``--horizon-length``)."""
    return AGENT_HORIZONS.get(agent_entry_point, default)


def is_sapg_agent(agent_entry_point: str) -> bool:
    """SAPG configs carry a central-value block and an exploration block
    size the driver must scale; the plain-PPO (isaaclab_repose) ones do not,
    and Hydra rejects an override of a key the config lacks."""
    return agent_entry_point in SAPG_AGENT_ENTRY_POINTS or "sapg" in agent_entry_point


def build_train_cmd(
    *, train_python: str, population_path: Path, num_envs: int, max_epochs: int, hydra_run_dir: Path,
    checkpoint: Optional[Path], resume_success_tolerance: Optional[float], horizon_length: int,
    agent_entry_point: str = AGENT_ENTRY_POINT, seed: Optional[int] = None,
    extra_overrides: Sequence[str] = (), task_profile: str = DEFAULT_TASK_PROFILE,
) -> List[str]:
    if task_profile not in TASK_PROFILES:
        raise ValueError(f"task_profile={task_profile!r}; expected one of {TASK_PROFILES}")
    sapg = is_sapg_agent(agent_entry_point)
    horizon_length = agent_horizon(agent_entry_point, horizon_length)
    minibatch_size, block_size = _minibatch_and_block_size(num_envs, horizon_length)
    if not sapg:
        batch_size = num_envs * horizon_length
        minibatch_size = batch_size if batch_size <= PPO_MINIBATCH_CAP else PPO_MINIBATCH_CAP
    cmd = [
        str(train_python), str(TRAIN_PY),
        "--task", TASK_ID, "--agent", agent_entry_point, "--headless",
    ]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint), "--checkpoint_load_mode", "weights"]
    cmd += [
        f"env.task_profile={task_profile}",
        f"env.assets.hand_population={population_path}",
        f"env.scene.num_envs={num_envs}",
        f"agent.params.config.max_epochs={max_epochs}",
        f"agent.params.config.minibatch_size={minibatch_size}",
    ]
    if sapg:
        cmd += [
            # The asymmetric-critic block's OWN minibatch_size is a separate
            # hardcoded 16384 in the yaml (central_value_config.minibatch_size);
            # train_central_value() divides by `batch_size // this value` with
            # no guard, so at any --num-envs small enough that
            # num_envs*horizon_length < 16384 (e.g. this pilot's tiny/quick
            # verification runs), the DEFAULT central_value minibatch_size
            # alone gives num_minibatches == 0 -> ZeroDivisionError, even though
            # the OUTER minibatch_size above was already scaled down correctly.
            # Found by reproducing this exact crash at --num-envs 64 while
            # investigating the eval-fitness player mismatch (see this
            # package's README's "eval fitness" section).
            f"agent.params.config.central_value_config.minibatch_size={minibatch_size}",
            f"agent.params.config.expl_coef_block_size={block_size}",
        ]
    cmd += [f"hydra.run.dir={hydra_run_dir}"]
    if resume_success_tolerance is not None:
        cmd += [f"env.termination.resume_success_tolerance={resume_success_tolerance}"]
    if seed is not None:
        cmd += [f"agent.params.seed={seed}"]
    cmd += list(extra_overrides)
    return cmd


def _read_json_safely(path: Path) -> Optional[dict]:
    """Tolerates a read racing the writer's `tmp.replace(path)` (atomic on
    POSIX, but the file can still not exist yet, or -- if we ever read
    mid-``write_text`` on a non-atomic path -- be briefly invalid JSON)."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@dataclass
class TrainRunResult:
    windows: List[dict]
    returncode: int
    boot_s: float
    train_s: float
    fps: float


def run_training_subprocess(
    cmd: Sequence[str], *, cwd: Path, extra_env: Dict[str, str], log_path: Path, score_path: Path,
    timeout_s: int, poll_interval_s: float = 3.0,
) -> TrainRunResult:
    """Launches ``cmd`` (already wrapped in ``timeout -k 30 <timeout_s>`` by
    the caller) and polls ``score_path`` (design_scoring's per-design
    snapshot -- rewritten in place every write window) for mtime changes,
    recording a deep copy of each new window's JSON. `boot_s` is
    wall-clock until the FIRST window appears (Kit/scene boot, before any
    scoring has happened); `train_s` is from there to process exit -- an
    approximation (this module has no other signal for "Kit is up"), noted
    as such in the driver's own report."""
    log_fh = open(log_path, "w")
    t0 = time.time()
    # No `start_new_session`: the subprocess (`timeout -k 30 ... train.py`)
    # inherits the driver's own process group, so a single `kill -- -<pgid>`
    # (the driver's session, if the driver itself was launched via `setsid`)
    # takes down the driver AND this subprocess AND everything `timeout`
    # itself spawned in one shot -- see the README's kill/resume recipe.
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=extra_env, stdout=log_fh, stderr=subprocess.STDOUT)
    pidfile = log_path.with_name(f"train_pid_{proc.pid}")
    pidfile.write_text(f"{proc.pid}\n")

    windows: List[dict] = []
    last_mtime: Optional[float] = None
    boot_s: Optional[float] = None

    def _poll_once() -> None:
        nonlocal last_mtime, boot_s
        if not score_path.exists():
            return
        mtime = score_path.stat().st_mtime
        if last_mtime is not None and mtime <= last_mtime:
            return
        payload = _read_json_safely(score_path)
        if payload is None:
            return
        last_mtime = mtime
        if boot_s is None:
            boot_s = time.time() - t0
        windows.append(payload)

    try:
        while proc.poll() is None:
            time.sleep(poll_interval_s)
            _poll_once()
    finally:
        # One last check in case a final write landed between the last
        # poll and process exit.
        _poll_once()
        log_fh.close()

    t_end = time.time()
    train_s = t_end - t0 - (boot_s or 0.0)
    fps = 0.0
    if windows:
        last = windows[-1]
        elapsed = float(last.get("elapsed_s", 0.0))
        steps = float(last.get("steps", 0.0))
        num_envs_seen = None
        for row in last.get("designs", {}).values():
            num_envs_seen = row.get("envs_per_design")
            break
        # fps is env-steps/second across the whole population: steps here
        # is the number of env.step() CALLS (see design_scoring.py's
        # `_score_total_steps`), so multiply by the run's total env count,
        # not any one design's share of it.
        total_envs = sum(row.get("envs_per_design", 0) for row in last.get("designs", {}).values())
        if elapsed > 0 and steps > 0 and total_envs > 0:
            fps = total_envs * steps / elapsed

    if pidfile.exists():
        pidfile.unlink()

    return TrainRunResult(
        windows=windows, returncode=int(proc.returncode if proc.returncode is not None else -1),
        boot_s=float(boot_s or 0.0), train_s=float(max(train_s, 0.0)), fps=float(fps),
    )


# --------------------------------------------------------------------------
# train_tail fitness
# --------------------------------------------------------------------------


@dataclass
class FitnessResult:
    fitness: float
    episodes: int
    low_confidence: bool


def compute_train_tail_fitness(
    windows: Sequence[dict], *, tail_frac: float = 0.3, min_episodes: int = 10,
) -> Dict[str, FitnessResult]:
    """Mean `graded_fitness` per design (keyed by the population entry's own
    `source` string, stable across windows within one generation), weighted
    by episode count, over the LAST `tail_frac` fraction of write windows
    captured this generation (at least 1). A design with fewer than
    `min_episodes` total episodes in that tail is flagged `low_confidence`
    but still reported (never dropped -- a low-confidence 0 is still
    informative to the archive/probes log)."""
    if not windows:
        return {}
    n_tail = max(1, math.ceil(len(windows) * tail_frac))
    tail = windows[-n_tail:]

    weighted_sum: Dict[str, float] = {}
    episode_sum: Dict[str, int] = {}
    for w in tail:
        for row in w.get("designs", {}).values():
            source = row.get("source")
            if not source:
                continue
            ep = int(row.get("episodes", 0))
            if ep <= 0:
                continue
            gf = float(row.get("graded_fitness", 0.0))
            weighted_sum[source] = weighted_sum.get(source, 0.0) + gf * ep
            episode_sum[source] = episode_sum.get(source, 0) + ep

    out: Dict[str, FitnessResult] = {}
    # every source that ever appeared in ANY window (not just the tail) --
    # a design with zero tail-window episodes still gets an entry (fitness
    # 0.0, low_confidence True) rather than silently vanishing.
    all_sources = {row.get("source") for w in windows for row in w.get("designs", {}).values() if row.get("source")}
    for source in all_sources:
        ep = episode_sum.get(source, 0)
        fitness = (weighted_sum[source] / ep) if ep > 0 else 0.0
        out[source] = FitnessResult(fitness=fitness, episodes=ep, low_confidence=ep < min_episodes)
    return out


CHECKPOINT_RE = re.compile(r"_ep_(\d+)_")


def find_last_checkpoint(train_dir: Path) -> Optional[Path]:
    """`<train_dir>/.../last/model.pth` -- rl_games' vendored agent
    (`a2c_common.py`) writes exactly this path every 3 epochs (independent
    of `save_frequency`), always overwriting IN PLACE, which is what the
    plan's own `--checkpoint <prev gen last/model.pth>` names literally.

    Searched with `rglob`, not a direct child path: rl_games nests its own
    `experiment_dir` one level below the Hydra run dir we pass as
    `train_dir` (`os.path.join(train_dir, experiment_name)`, where
    `experiment_name` is `agent.params.config.full_experiment_name` /
    `name` -- "0_inhand_reorient_sapg" by default, never overridden by
    `build_train_cmd`, but not worth hard-coding here either).

    Falls back to the newest `nn/last_<name>_ep_<N>_rew_<R>.pth` (highest
    epoch `<N>` wins, ties by mtime), then the single best-checkpoint file
    (`nn/<name>.pth`) by mtime, for a run too short to have reached epoch 3
    (`last/model.pth` is never written at all in that case)."""
    direct = sorted(train_dir.rglob("last/model.pth"))
    if direct:
        return max(direct, key=lambda p: p.stat().st_mtime)
    last_candidates = sorted(train_dir.rglob("nn/last_*.pth"))
    if last_candidates:
        def _epoch(p: Path) -> Tuple[int, float]:
            m = CHECKPOINT_RE.search(p.name)
            return (int(m.group(1)) if m else -1, p.stat().st_mtime)
        return max(last_candidates, key=_epoch)
    other = sorted(train_dir.rglob("nn/*.pth"))
    if other:
        return max(other, key=lambda p: p.stat().st_mtime)
    return None


# --------------------------------------------------------------------------
# Checkpoint health and carry-over (I41)
# --------------------------------------------------------------------------

SIGMA_KEY = "a2c_network.sigma"
SIGMA_MODES = ("keep", "reset", "clamp")
# torch.cuda.amp.GradScaler's own initial state (rl_games builds its scaler
# with the defaults). rl_games' weights-mode load restores the scaler from the
# checkpoint, so a scale that collapsed to 0 in one generation (every step
# skipped after a non-finite loss) would otherwise freeze the next one too.
FRESH_GRAD_SCALER = {
    "scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000, "_growth_tracker": 0,
}
MIN_HEALTHY_GRAD_SCALE = 1e-3


class NonFiniteCheckpoint(ValueError):
    """A checkpoint holds NaN/Inf tensors and must not be carried forward."""


class GenerationFailed(RuntimeError):
    """Both attempts at one generation's training failed."""


def _load_checkpoint(path: Path) -> dict:
    import torch  # lazy: the driver itself stays importable without torch

    return torch.load(str(path), map_location="cpu", weights_only=False)


def _policy_state(ck: dict) -> dict:
    """rl_games keys a checkpoint by policy/rank index (`ck[0]["model"]`)."""
    if isinstance(ck, dict) and 0 in ck and isinstance(ck[0], dict):
        return ck[0]
    return ck


def _iter_float_tensors(state: dict):
    import torch

    for group in ("model", "assymetric_vf_nets"):
        sub = state.get(group)
        if not isinstance(sub, dict):
            continue
        for name, value in sub.items():
            if torch.is_tensor(value) and value.is_floating_point():
                yield f"{group}/{name}", value


def _tensor_summary(t) -> Dict[str, float]:
    t = t.detach().double()
    return {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max())}


def checkpoint_stats(path: Path) -> Dict[str, Any]:
    """Health summary of an rl_games checkpoint: whether every model/critic
    tensor is finite (and which are not), the policy log-std (`sigma`) and
    std = exp(sigma) statistics (the network's `sigma_activation` is None,
    so the parameter IS the log-std), the GradScaler's scale, and the epoch."""
    import torch

    state = _policy_state(_load_checkpoint(Path(path)))
    nonfinite = [name for name, t in _iter_float_tensors(state) if not bool(torch.isfinite(t).all())]
    out: Dict[str, Any] = {
        "path": str(path), "finite": not nonfinite, "nonfinite_tensors": nonfinite,
        "epoch": state.get("epoch"), "grad_scale": None, "sigma": None, "std": None,
    }
    scaler = state.get("scaler")
    if isinstance(scaler, dict) and "scale" in scaler:
        out["grad_scale"] = float(scaler["scale"])
    sigma = state.get("model", {}).get(SIGMA_KEY)
    if sigma is not None and bool(torch.isfinite(sigma).all()):
        out["sigma"] = _tensor_summary(sigma)
        out["std"] = _tensor_summary(sigma.double().exp())
    return out


def prepare_carry_checkpoint(
    src: Path, dst: Path, *, sigma_mode: str = "keep", sigma_reset_value: float = 0.0,
    sigma_clamp_max: Optional[float] = None, reset_grad_scaler: bool = True,
    norm_count_cap: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate `src` and write the checkpoint the next generation starts
    from to `dst` (`src` itself is never modified). Raises
    `NonFiniteCheckpoint` if any model/critic tensor is NaN/Inf.

    `sigma_mode`: "keep" leaves the policy log-std as trained; "reset" sets
    every entry to `sigma_reset_value` (0.0 = std 1, the value a fresh
    policy starts from); "clamp" caps it at `sigma_clamp_max`.
    `norm_count_cap` (if > 0) caps the sample count of every running
    observation/value normalizer, so the next generation's statistics move
    it within a few epochs instead of being outweighed by every earlier
    generation's samples. Returns `checkpoint_stats(dst)`."""
    import torch

    if sigma_mode not in SIGMA_MODES:
        raise ValueError(f"sigma_mode must be one of {SIGMA_MODES}, got {sigma_mode!r}")
    ck = _load_checkpoint(Path(src))
    state = _policy_state(ck)
    bad = [name for name, t in _iter_float_tensors(state) if not bool(torch.isfinite(t).all())]
    if bad:
        raise NonFiniteCheckpoint(f"checkpoint {src} has non-finite tensors: {', '.join(bad)}")
    sigma = state.get("model", {}).get(SIGMA_KEY)
    if sigma is not None:
        if sigma_mode == "reset":
            state["model"][SIGMA_KEY] = torch.full_like(sigma, float(sigma_reset_value))
        elif sigma_mode == "clamp":
            if sigma_clamp_max is None:
                raise ValueError("sigma_mode='clamp' needs sigma_clamp_max")
            state["model"][SIGMA_KEY] = sigma.clamp(max=float(sigma_clamp_max))
    if norm_count_cap is not None and norm_count_cap > 0:
        for group in ("model", "assymetric_vf_nets"):
            sub = state.get(group)
            if not isinstance(sub, dict):
                continue
            for name, value in sub.items():
                if name.endswith("mean_std.count") and torch.is_tensor(value) and float(value) > norm_count_cap:
                    sub[name] = torch.full_like(value, float(norm_count_cap))
    if reset_grad_scaler and isinstance(state.get("scaler"), dict):
        state["scaler"] = dict(FRESH_GRAD_SCALER)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(ck, str(tmp))
    tmp.replace(dst)
    return checkpoint_stats(dst)


# --------------------------------------------------------------------------
# Per-generation training-health aggregates
# --------------------------------------------------------------------------

TB_TAGS: Tuple[str, ...] = (
    "successes", "rot_error_mean", "episode_lengths/iter", "episode_final/done_drop",
    "episode_final/done_nonfinite", "losses/entropy", "info/last_lr", "info/kl", "rewards/iter",
    # n_nonfinite of a loss is the early warning for the fp16-overflow failure (I41): NaN
    # losses, GradScaler backing off to 0, then NaN weights on the next finite-loss step.
    "losses/a_loss",
)


def training_scalars(train_dir: Path, tags: Sequence[str] = TB_TAGS, frac: float = 0.1) -> Dict[str, Dict[str, float]]:
    """Mean of each TensorBoard scalar over the first (`head`) and last
    (`tail`) `frac` of its logged points, read from rl_games' own
    `<train_dir>/<experiment>/summaries`. Empty if there are no summaries
    or tensorboard is unavailable (never fatal to the driver)."""
    event_files = sorted(Path(train_dir).rglob("summaries/events.out.tfevents.*"))
    if not event_files:
        return {}
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:  # noqa: BLE001 -- optional dependency
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for logdir in sorted({p.parent for p in event_files}):
        ea = event_accumulator.EventAccumulator(str(logdir), size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception:  # noqa: BLE001 -- a truncated event file must not stop the driver
            continue
        available = set(ea.Tags().get("scalars", []))
        for tag in tags:
            if tag not in available:
                continue
            values = [e.value for e in ea.Scalars(tag)]
            finite = [v for v in values if math.isfinite(v)]
            if not finite:
                continue
            k = max(1, int(len(finite) * frac))
            out[tag] = {
                "head": float(np.mean(finite[:k])), "tail": float(np.mean(finite[-k:])),
                "n": len(values), "n_nonfinite": len(values) - len(finite),
            }
    return out


def _aggregate_windows(windows: Sequence[dict]) -> Dict[str, float]:
    episodes = 0
    sums = {"goals_per_episode": 0.0, "time_held_mean_s": 0.0, "graded_fitness": 0.0}
    for w in windows:
        for row in w.get("designs", {}).values():
            ep = int(row.get("episodes", 0))
            if ep <= 0:
                continue
            episodes += ep
            for key in sums:
                sums[key] += float(row.get(key, 0.0)) * ep
    out = {key: (value / episodes if episodes else 0.0) for key, value in sums.items()}
    out["episodes"] = episodes
    return out


def window_metrics(windows: Sequence[dict], *, tail_frac: float = 0.3) -> Dict[str, Dict[str, float]]:
    """Episode-weighted aggregates over ALL designs (goals per episode =
    aggregate successes per episode, mean time held, mean graded fitness),
    for the first and last `tail_frac` of this generation's write windows."""
    if not windows:
        return {}
    k = max(1, math.ceil(len(windows) * tail_frac))
    return {"head": _aggregate_windows(windows[:k]), "tail": _aggregate_windows(windows[-k:])}


def nonfinite_by_source(windows: Sequence[dict]) -> Dict[str, int]:
    """Episodes ended by the env's non-finite physics guard, per design
    `source` (each design's running total, so the latest window wins).
    Only designs with at least one such episode are listed."""
    out: Dict[str, int] = {}
    for w in windows:
        for row in w.get("designs", {}).values():
            source = row.get("source")
            total = int(row.get("nonfinite_resets_total", 0))
            if source and total > 0:
                out[source] = max(out.get(source, 0), total)
    return out


def assess_attempt(result: "TrainRunResult", train_dir: Path) -> Tuple[bool, str, Optional[Path], Optional[dict]]:
    """`(ok, reason, checkpoint, checkpoint_stats)` for one training
    attempt. OK only if train.py exited 0, at least one scoring window was
    captured, a checkpoint exists, every tensor in it is finite, and its
    GradScaler has not collapsed (a scale near 0 means every optimizer step
    after some point was skipped on a non-finite loss)."""
    checkpoint = find_last_checkpoint(train_dir)
    stats = checkpoint_stats(checkpoint) if checkpoint is not None else None
    if result.returncode != 0:
        return False, f"train.py exited {result.returncode}", checkpoint, stats
    if not result.windows:
        return False, "no design-scoring window was written", checkpoint, stats
    if checkpoint is None or stats is None:
        return False, "no checkpoint was written", checkpoint, stats
    if not stats["finite"]:
        return False, f"checkpoint has non-finite tensors: {', '.join(stats['nonfinite_tensors'])}", checkpoint, stats
    if stats["grad_scale"] is not None and stats["grad_scale"] < MIN_HEALTHY_GRAD_SCALE:
        return False, f"GradScaler collapsed (scale {stats['grad_scale']}): training stopped updating", checkpoint, stats
    return True, "ok", checkpoint, stats


# --------------------------------------------------------------------------
# State (archive + rng + generation index + checkpoint + config hash)
# --------------------------------------------------------------------------


def _config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_state(
    path: Path, *, archive: "arch.Archive", driver_rng: np.random.Generator, generation_completed: int,
    last_checkpoint: Optional[str], prev_tolerance: float, minter: IdMinter, config: dict,
) -> None:
    doc = {
        "schema": STATE_SCHEMA,
        "generation_completed": generation_completed,
        "archive": archive.to_dict(),
        "driver_rng_state": driver_rng.bit_generator.state,
        "last_checkpoint": last_checkpoint,
        "prev_tolerance": prev_tolerance,
        "next_uid": minter.next_uid,
        "config": config,
        "config_hash": _config_hash(config),
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True, indent=1))
    tmp.replace(out_path)


def load_state(path: Path) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != STATE_SCHEMA:
        raise ValueError(f"unsupported state schema {doc.get('schema')!r}, expected {STATE_SCHEMA!r}")
    return doc


# --------------------------------------------------------------------------
# CLI / main loop
# --------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="Named Distribution, resolved through "
                    "hand_sampler.grammar.variants' registries (NAMED_DISTRIBUTIONS, G0_SCREEN_VARIANTS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, required=True)
    ap.add_argument("--designs", type=int, default=64, help="Population size N per generation")
    ap.add_argument("--probes", default=",".join(DEFAULT_PROBES),
                     help="Comma-separated manifest hand ids, fixed every generation")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--epochs-per-gen", type=int, default=200, help="agent.params.config.max_epochs per generation")
    ap.add_argument("--fitness", choices=("train_tail", "eval"), default="train_tail")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-offspring-retries", type=int, default=16)
    ap.add_argument("--min-episodes-per-design", type=int, default=10)
    ap.add_argument("--tail-frac", type=float, default=0.3)
    ap.add_argument("--horizon-length", type=int, default=DEFAULT_HORIZON_LENGTH)
    ap.add_argument("--target-windows", type=int, default=15,
                     help="GENMECH_DESIGN_SCORE_WRITE_EVERY is chosen so this many write windows "
                          "(at least 10, per the plan) fall inside one generation")
    ap.add_argument("--gen-timeout-s", type=int, default=1800, help="timeout -k 30 <this> per generation")
    ap.add_argument("--poll-interval-s", type=float, default=3.0)
    ap.add_argument("--train-python", default=str(REPO_ROOT / ".venv_isaacsim" / "bin" / "python3"))
    ap.add_argument("--eval-episodes-per-design", type=int, default=8)
    # --- training config / stability (I41) ---
    ap.add_argument("--task-profile", choices=TASK_PROFILES, default=DEFAULT_TASK_PROFILE,
                    help="env.task_profile for every generation: legacy (default; the spec runs before "
                         "2026-10-01 used), anyrotate (AnyRotate, CoRL 2024; pair it with --agent-entry-point "
                         f"{ANYROTATE_POP_AGENT_ENTRY_POINT}) or isaaclab_repose (NVIDIA's "
                         f"Isaac-Repose-Cube-Allegro spec; --agent-entry-point {REPOSE_POP_AGENT_ENTRY_POINT})")
    ap.add_argument("--agent-entry-point", default=AGENT_ENTRY_POINT,
                    help="gym-registered rl_games config key for train.py's --agent (e.g. "
                         "rl_games_sapg_pop_cfg_entry_point for InHandReorientPopSAPG.yaml, "
                         f"{REPOSE_POP_AGENT_ENTRY_POINT} for InHandReposeIsaacLabPopPPO.yaml)")
    ap.add_argument("--train-override", action="append", default=[], metavar="KEY=VALUE",
                    help="extra Hydra override passed to every generation's train.py (repeatable)")
    ap.add_argument("--train-seed", type=int, default=42,
                    help="agent.params.seed for a generation's first attempt; a retry uses seed+1")
    ap.add_argument("--sigma-on-carry", choices=SIGMA_MODES, default="keep",
                    help="policy log-std handling when carrying weights into the next generation: "
                         "keep as trained, reset to --sigma-reset-value, or clamp at --sigma-clamp-max")
    ap.add_argument("--sigma-reset-value", type=float, default=0.0, help="log-std after reset (0.0 = std 1)")
    ap.add_argument("--sigma-clamp-max", type=float, default=0.0, help="log-std cap for --sigma-on-carry clamp")
    ap.add_argument("--norm-count-cap", type=float, default=0.0,
                    help="cap the carried observation/value normalizers' sample count (0 = keep), so a "
                         "new generation's designs re-shape the normalizer within a few epochs")
    ap.add_argument("--no-reset-grad-scaler", dest="reset_grad_scaler", action="store_false",
                    help="carry the trained GradScaler state forward instead of resetting it")
    ap.add_argument("--train-retries", type=int, default=1,
                    help="retries of a failed generation (same population, same carried checkpoint) "
                         "before the driver stops")
    return ap.parse_args(argv)


def _resolved_config(args: argparse.Namespace) -> dict:
    config = {
        "variant": args.variant, "seed": args.seed, "designs": args.designs,
        "probes": sorted(args.probes.split(",")), "num_envs": args.num_envs,
        "epochs_per_gen": args.epochs_per_gen, "fitness": args.fitness,
        "max_offspring_retries": args.max_offspring_retries,
        "min_episodes_per_design": args.min_episodes_per_design, "tail_frac": args.tail_frac,
        "horizon_length": args.horizon_length,
        "agent_entry_point": args.agent_entry_point, "train_override": list(args.train_override),
        "sigma_on_carry": args.sigma_on_carry, "sigma_reset_value": args.sigma_reset_value,
        "sigma_clamp_max": args.sigma_clamp_max, "reset_grad_scaler": args.reset_grad_scaler,
        "norm_count_cap": args.norm_count_cap,
    }
    # Only when it is not the default, so a run started before the flag
    # existed keeps its config hash on resume.
    if args.task_profile != DEFAULT_TASK_PROFILE:
        config["task_profile"] = args.task_profile
    return config


def _write_env(cache_path: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["OMNI_KIT_CACHE_PATH"] = str(cache_path)
    env["WANDB_MODE"] = "disabled"
    return env


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


CSV_COLUMNS = [
    "generation", "coverage", "n_cells_total", "qd_score", "best_fitness", "mean_fitness", "median_fitness",
    "n_distinct_founders", "max_founder_share", "mean_joint_count", "max_joint_count",
    "boot_s", "train_s", "select_s", "fps", "checkpoint", "n_windows",
    "std_mean", "std_min", "std_max", "agg_goals_per_episode", "agg_time_held_s", "tb_successes",
    "tb_rot_error_mean", "nonfinite_resets", "n_attempts",
]


def _append_csv(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in CSV_COLUMNS})


def _set_aside_failed_attempt(gen_dir: Path, attempt: int) -> None:
    """Move a failed attempt's `train/` and `train.log` out of the way
    (`train_failed_<k>/`, `train_failed_<k>.log`) so the retry writes a
    fresh `train/` and the failed attempt stays inspectable."""
    train_dir = gen_dir / "train"
    if train_dir.exists():
        train_dir.rename(gen_dir / f"train_failed_{attempt}")
    log_path = gen_dir / "train.log"
    if log_path.exists():
        log_path.rename(gen_dir / f"train_failed_{attempt}.log")


def run_generation(
    generation: int, args: argparse.Namespace, run_dir: Path, dist: Distribution, archive: "arch.Archive",
    driver_rng: np.random.Generator, minter: IdMinter, last_checkpoint: Optional[Path], prev_tolerance: float,
    probe_hand_ids: Sequence[str],
) -> Tuple[Optional[Path], float]:
    gen_dir = run_dir / f"gen_{generation}"
    train_dir = gen_dir / "train"
    gen_dir.mkdir(parents=True, exist_ok=True)

    t_select0 = time.time()
    plan = build_generation_population(
        generation, args.designs, probe_hand_ids, dist, archive, driver_rng, minter,
        max_offspring_retries=args.max_offspring_retries,
    )
    population_path = gen_dir / "population.json"
    population_doc = pf.write_population(population_path, plan.entries)
    select_build_s = time.time() - t_select0

    total_steps_budget = max(1, args.epochs_per_gen * agent_horizon(args.agent_entry_point, args.horizon_length))
    write_every = max(1, total_steps_budget // max(10, args.target_windows))

    env_vars = _write_env(Path(f"/tmp/{os.environ.get('USER', 'user')}/ov_cache"))
    env_vars["GENMECH_DESIGN_SCORE_WRITE_EVERY"] = str(write_every)

    # The checkpoint this generation starts from: validated (finite) and
    # prepared once, reused unchanged by a retry.
    carry_path: Optional[Path] = None
    carry_stats: Optional[dict] = None
    if last_checkpoint is not None:
        carry_path = gen_dir / "carry_checkpoint.pth"
        carry_stats = prepare_carry_checkpoint(
            Path(last_checkpoint), carry_path, sigma_mode=args.sigma_on_carry,
            sigma_reset_value=args.sigma_reset_value, sigma_clamp_max=args.sigma_clamp_max,
            reset_grad_scaler=args.reset_grad_scaler, norm_count_cap=args.norm_count_cap,
        )
        print(f"[driver] generation {generation}: carrying {last_checkpoint} -> {carry_path} "
              f"(sigma_on_carry={args.sigma_on_carry}, std mean {carry_stats['std']['mean']:.3f})"
              if carry_stats.get("std") else
              f"[driver] generation {generation}: carrying {last_checkpoint} -> {carry_path}", flush=True)

    score_path = train_dir / "per_design_scores_rank0.json"
    log_path = gen_dir / "train.log"
    attempts: List[dict] = []
    result: Optional[TrainRunResult] = None
    new_checkpoint: Optional[Path] = None
    trained_stats: Optional[dict] = None
    n_attempts = 1 + max(0, int(args.train_retries))
    for attempt in range(n_attempts):
        if attempt > 0:
            _set_aside_failed_attempt(gen_dir, attempt - 1)
        cmd = build_train_cmd(
            train_python=args.train_python, population_path=population_path, num_envs=args.num_envs,
            max_epochs=args.epochs_per_gen, hydra_run_dir=train_dir, checkpoint=carry_path,
            resume_success_tolerance=(prev_tolerance if generation > 0 else None),
            horizon_length=args.horizon_length, agent_entry_point=args.agent_entry_point,
            seed=args.train_seed + attempt, extra_overrides=args.train_override,
            task_profile=args.task_profile,
        )
        cmd = ["timeout", "-k", "30", str(args.gen_timeout_s)] + cmd
        print(f"[driver] generation {generation} attempt {attempt}: {len(plan.entries)} designs "
              f"({args.designs - len(probe_hand_ids)} archive + {len(probe_hand_ids)} probes), "
              f"write_every={write_every} steps, cmd={' '.join(cmd)}", flush=True)
        result = run_training_subprocess(
            cmd, cwd=REPO_ROOT, extra_env=env_vars, log_path=log_path, score_path=score_path,
            timeout_s=args.gen_timeout_s, poll_interval_s=args.poll_interval_s,
        )
        ok, reason, new_checkpoint, trained_stats = assess_attempt(result, train_dir)
        attempts.append({"attempt": attempt, "returncode": result.returncode, "ok": ok, "reason": reason,
                         "n_windows": len(result.windows)})
        print(f"[driver] generation {generation} attempt {attempt}: train.py exited {result.returncode}, "
              f"{len(result.windows)} window(s) captured, boot={result.boot_s:.1f}s train={result.train_s:.1f}s "
              f"fps={result.fps:.0f} -> {'OK' if ok else 'FAILED: ' + reason}", flush=True)
        if ok:
            break
    else:
        raise GenerationFailed(
            f"generation {generation} failed after {n_attempts} attempt(s): "
            + "; ".join(f"attempt {a['attempt']}: {a['reason']}" for a in attempts)
            + f" (logs: {gen_dir}). Nothing from this generation was scored or archived.")
    assert result is not None and new_checkpoint is not None

    t_select1 = time.time()
    fitness_by_source = compute_train_tail_fitness(
        result.windows, tail_frac=args.tail_frac, min_episodes=args.min_episodes_per_design,
    )

    candidates: List[arch.Candidate] = []
    probe_report: Dict[str, dict] = {}
    for meta in plan.metas:
        fr = fitness_by_source.get(meta.source)
        if fr is None:
            fr = FitnessResult(fitness=0.0, episodes=0, low_confidence=True)
            print(f"[driver] WARNING: design {meta.design_id!r} (source {meta.source!r}) got no "
                  f"episodes this generation; recording fitness 0.0, low_confidence", flush=True)
        if meta.role == "probe":
            probe_report[meta.hand_id] = {
                "fitness": fr.fitness, "episodes": fr.episodes, "low_confidence": fr.low_confidence,
            }
            continue
        candidates.append(arch.Candidate(
            design_id=meta.design_id, derivation_dict=_entry_derivation(plan.entries, meta.source),
            sha256=_entry_sha256(plan.entries, meta.source), founder_id=meta.founder_id,
            parent_id=meta.parent_id, generation_born=meta.generation_born, digit_count=meta.digit_count,
            joint_count=meta.joint_count, fitness=fr.fitness, episodes=fr.episodes,
            low_confidence=fr.low_confidence, source=meta.source,
        ))

    archive.update_generation(candidates, generation)
    select_s = select_build_s + (time.time() - t_select1)

    new_tolerance = prev_tolerance
    if result.windows:
        new_tolerance = float(result.windows[-1].get("success_tolerance", prev_tolerance))

    train_metrics = training_scalars(train_dir)
    win_metrics = window_metrics(result.windows, tail_frac=args.tail_frac)
    nonfinite = nonfinite_by_source(result.windows)
    source_to_id = {m.source: m.design_id for m in plan.metas}
    nonfinite_by_design = {source_to_id.get(src, src): n for src, n in sorted(nonfinite.items())}
    if nonfinite_by_design:
        print(f"[driver] generation {generation}: non-finite physics resets by design: {nonfinite_by_design}",
              flush=True)

    summary = archive.summary()
    row = {
        "generation": generation,
        "coverage": summary["coverage"], "n_cells_total": summary["n_cells_total"],
        "qd_score": summary["qd_score"], "best_fitness": summary["best_fitness"],
        "mean_fitness": summary["mean_fitness"], "median_fitness": summary["median_fitness"],
        "n_distinct_founders": summary["n_distinct_founders"], "max_founder_share": summary["max_founder_share"],
        "mean_joint_count": summary["mean_joint_count"], "max_joint_count": summary["max_joint_count"],
        "cells": summary["cells"],
        "probes": probe_report,
        "timings": {"boot_s": result.boot_s, "train_s": result.train_s, "select_s": select_s},
        "fps": result.fps,
        "checkpoint": str(new_checkpoint),
        "success_tolerance": new_tolerance,
        "n_windows": len(result.windows),
        "n_designs": len(plan.entries),
        "returncode": result.returncode,
        "attempts": attempts,
        "population_sha256": population_doc["population_sha256"],
        "sigma_carried_in": carry_stats,
        "sigma_trained": trained_stats,
        "train_metrics": train_metrics,
        "window_metrics": win_metrics,
        "nonfinite_by_design": nonfinite_by_design,
    }
    _append_jsonl(run_dir / "generations.jsonl", row)
    csv_row = {k: row[k] for k in CSV_COLUMNS if k in row}
    std = (trained_stats or {}).get("std") or {}
    tail = win_metrics.get("tail", {})
    csv_row.update(
        boot_s=result.boot_s, train_s=result.train_s, select_s=select_s,
        std_mean=std.get("mean"), std_min=std.get("min"), std_max=std.get("max"),
        agg_goals_per_episode=tail.get("goals_per_episode"), agg_time_held_s=tail.get("time_held_mean_s"),
        tb_successes=train_metrics.get("successes", {}).get("tail"),
        tb_rot_error_mean=train_metrics.get("rot_error_mean", {}).get("tail"),
        nonfinite_resets=sum(nonfinite.values()), n_attempts=len(attempts),
    )
    _append_csv(run_dir / "generations.csv", csv_row)

    return new_checkpoint, new_tolerance


def _entry_derivation(entries: Sequence["pf.PopulationEntry"], source: str) -> dict:
    for e in entries:
        if e.source == source:
            return e.derivation_dict
    raise KeyError(source)


def _entry_sha256(entries: Sequence["pf.PopulationEntry"], source: str) -> str:
    for e in entries:
        if e.source == source:
            return e.sha256
    raise KeyError(source)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    probe_hand_ids = [h for h in args.probes.split(",") if h]

    state_path = run_dir / "state.json"
    config = _resolved_config(args)
    start_generation = 0
    archive = arch.Archive()
    driver_rng = np.random.default_rng(args.seed)
    minter = IdMinter()
    last_checkpoint: Optional[Path] = None
    prev_tolerance = 0.0

    if state_path.exists():
        state = load_state(state_path)
        if state.get("config_hash") != _config_hash(config):
            print("[driver] WARNING: resuming with a config that differs from the run dir's saved "
                  "state (see state.json's own 'config' for what it was run with before) -- continuing "
                  "with the NEW config for generations from here on.", flush=True)
        archive = arch.Archive.from_dict(state["archive"])
        driver_rng = np.random.default_rng(0)
        driver_rng.bit_generator.state = state["driver_rng_state"]
        minter = IdMinter(state.get("next_uid", 0))
        last_checkpoint = Path(state["last_checkpoint"]) if state.get("last_checkpoint") else None
        prev_tolerance = float(state.get("prev_tolerance", 0.0))
        start_generation = int(state["generation_completed"]) + 1
        print(f"[driver] resuming {run_dir} from generation {start_generation} "
              f"(archive coverage {archive.coverage()}/{arch.N_CELLS}, checkpoint={last_checkpoint})",
              flush=True)

    dist = resolve_variant(args.variant)

    if args.fitness == "eval":
        print("[driver] --fitness eval is not implemented in this pilot driver "
              "(evaluate_population.py's own player/checkpoint mismatch, see the branch's "
              "evaluate_population.py fix and worker report) -- falling back to train_tail.",
              flush=True)

    for generation in range(start_generation, args.generations):
        try:
            last_checkpoint, prev_tolerance = run_generation(
                generation, args, run_dir, dist, archive, driver_rng, minter, last_checkpoint, prev_tolerance,
                probe_hand_ids,
            )
        except (GenerationFailed, NonFiniteCheckpoint) as exc:
            # state.json is deliberately NOT rewritten: it still describes the
            # last generation that completed, so a rerun of the same command
            # resumes (retries) this generation from the last good checkpoint.
            print(f"[driver] ERROR: {exc}", flush=True)
            print(f"[driver] stopping; {state_path} is left at generation "
                  f"{generation - 1 if generation > 0 else 'none (no generation completed)'}.", flush=True)
            return 2
        save_state(
            state_path, archive=archive, driver_rng=driver_rng, generation_completed=generation,
            last_checkpoint=str(last_checkpoint) if last_checkpoint else None, prev_tolerance=prev_tolerance,
            minter=minter, config=config,
        )
        print(f"[driver] generation {generation} complete: coverage={archive.coverage()}/{arch.N_CELLS} "
              f"qd_score={archive.qd_score():.3f} best={archive.best_fitness()}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
