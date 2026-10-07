"""One training .sub per Wuji v2 variant, at the XM335's speed cap: every hand joint limited to 5 rad/s.

Identical to experiments/06oct_minimal_embodiment/make_runs.py (and so to 01oct_uniform_dynamics --canon) --
template, canonical palm frame + palm_extents, uniform dynamics, friction 0.5, RL configuration, 5000 epochs --
except for one line: HAND_VELOCITY_LIMIT=5.0, i.e. env.physics.hand_velocity_limit=5.0, which caps every hand
joint in the solver and replaces the 10 rad/s the uniform URDFs carry. Effort stays the spec's 0.5 N.m (already
the XM335-derived ceiling); stiffness, damping and armature stay the uniform gains (3.0, 0.0775, 0.00058).

    .venv_isaacsim/bin/python experiments/06oct_wuji_5rads/make_runs.py full                  # wuji2_full_v5_ema0.1_dact
    .venv_isaacsim/bin/python experiments/06oct_wuji_5rads/make_runs.py --dr --no-pose-noise full
    .venv_isaacsim/bin/python experiments/06oct_wuji_5rads/make_runs.py --kp-scale 200 --hand-pen 0.003 full
    .venv_isaacsim/bin/python experiments/06oct_wuji_5rads/make_runs.py --delay obs full      # + one more delay
    .venv_isaacsim/bin/python experiments/06oct_wuji_5rads/make_runs.py --ema 1 --no-action-delay full   # wuji2_full_v5_ema1

DEFAULTS (since 7 Oct): hand-target moving average 0.1 and action delay ON. The full hand learns at 5 rad/s only
with the moving average (12491 vs 11983), and action delay alone costs little (21667: 35.6 goals/episode at epoch
5000 vs 12491's 40.6), while obs delay (21666) and object-state delay (21668) each break learning on their own.
The runs in README.md generated before 7 Oct used the old defaults (no moving average, no delay); their names
say what they had, and the names still do: every run is _ema<A>, and _dact unless --no-action-delay.

--ema A sets the hand-target moving average (env.action.hand_moving_average=A, default 0.1; 1 is off;
run_rank.sh pins it to 1.0 and a later Hydra override wins) and adds _emaA to the run name. --no-action-delay
leaves use_action_delay at run_rank.sh's false and drops _dact. --dr turns back on the three
latency/noise knobs run_rank.sh switches off -- observation delay, action delay, and object-state delay + noise --
at the task config's own magnitudes, and adds _dr. --no-pose-noise (with --dr) keeps the object-state DELAY but
zeroes its pose noise (object_state_xyz_noise_std=0, object_state_rotation_noise_degrees=0) and adds _nopn.
--delay {obs,action,object} turns on ONE of the --dr delays alone (repeatable): obs = use_obs_delay, action =
use_action_delay, object = use_object_state_delay_noise with its pose noise zeroed (delay only); adds _dobs /
_dact / _dobj.
Values go into Hydra as float literals (200.0, not 200): Isaac Lab's configclass rejects an int for a float field.
--kp-scale K sets reward.keypoint_rew_scale (trained default 2000) and adds _kpK; --hand-pen P sets
reward.hand_actions_penalty_scale (an L1 hand joint-velocity penalty; trained default 0.0003) and adds _penP.

"full" is the unreduced hand (wuji2_left_uniform_handonly); any other tag is a make_missing_fingers.py variant.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/old_experiments/30sep_ten_hands"))
from make_runs import OBS_LIST, TEMPLATE  # noqa: E402

VLIM = 5.0
LOGS = REPO / "debug_outputs/train_logs/06oct_wuji_5rads"
FRICTION = " env.assets.robot_friction=0.5 env.assets.finger_tip_friction=0.5 env.assets.object_friction=0.5"
CANON_OBS = OBS_LIST.replace("object_vel]", "object_vel,palm_extents]")
CANON = " env.obs.canonical_palm_frame=true"
WANDB = "gen_mechanics_minimal_embodiment"

if __name__ == "__main__":
    args, ema, dr, nopn, kp, pen, delays = sys.argv[1:], 0.1, False, False, None, None, ["action"]
    while args[:1] in (["--ema"], ["--dr"], ["--no-pose-noise"], ["--kp-scale"], ["--hand-pen"], ["--delay"],
                       ["--no-action-delay"]):
        if args[0] == "--no-action-delay":
            delays, args = [d for d in delays if d != "action"], args[1:]
        elif args[0] == "--delay":
            if args[1] not in ("obs", "action", "object"):
                raise SystemExit(f"--delay takes obs, action or object, not {args[1]}")
            delays, args = delays + [args[1]] * (args[1] not in delays), args[2:]
        elif args[0] == "--ema":
            ema, args = float(args[1]), args[2:]
        elif args[0] == "--kp-scale":
            kp, args = float(args[1]), args[2:]
        elif args[0] == "--hand-pen":
            pen, args = float(args[1]), args[2:]
        elif args[0] == "--dr":
            dr, args = True, args[1:]
        else:
            nopn, args = True, args[1:]
    if dr:                                # --dr turns on all three delays: _dr says so
        delays = []
    if nopn and not dr:
        raise SystemExit("--no-pose-noise only means something with --dr")
    for tag in args:
        spec = "wuji2_left_uniform_handonly" + ("" if tag == "full" else f"_{tag}")
        if tag != "full" and not (
                REPO / f"assets/urdf/unified_dynamics_commercial_hands/wuji2/wuji2_left_{tag}.spec.json").exists():
            raise SystemExit(f"no variant {tag}: run make_missing_fingers.py first")
        run = (f"wuji2_{tag}_v5" + (f"_ema{ema:g}" if ema is not None else "") + ("_dr" if dr else "") + ("_nopn" if nopn else "")
               + (f"_kp{kp:g}" if kp is not None else "") + (f"_pen{pen:g}" if pen is not None else "")
               + "".join({"obs": "_dobs", "action": "_dact", "object": "_dobj"}[d] for d in delays))
        ema_hydra = f" env.action.hand_moving_average={float(ema)!r}" if ema is not None else ""
        if dr:
            ema_hydra += "".join(f" env.domain_randomization.{k}=true"
                                 for k in ("use_obs_delay", "use_action_delay", "use_object_state_delay_noise"))
        for d in delays:
            ema_hydra += {"obs": " env.domain_randomization.use_obs_delay=true",
                          "action": " env.domain_randomization.use_action_delay=true",
                          "object": (" env.domain_randomization.use_object_state_delay_noise=true"
                                     " env.domain_randomization.object_state_xyz_noise_std=0.0"
                                     " env.domain_randomization.object_state_rotation_noise_degrees=0.0")}[d]
        if kp is not None:
            ema_hydra += f" env.reward.keypoint_rew_scale={float(kp)!r}"
        if pen is not None:
            ema_hydra += f" env.reward.hand_actions_penalty_scale={float(pen)!r}"
        if nopn:
            ema_hydra += (" env.domain_randomization.object_state_xyz_noise_std=0.0"
                          " env.domain_randomization.object_state_rotation_noise_degrees=0.0")
        text = TEMPLATE.format(hand=f"wuji2 {tag}", run=run, spec=spec, obs=CANON_OBS, logs=LOGS, repo=REPO,
                               hydra_extra=FRICTION + CANON + ema_hydra, vlim=f"export HAND_VELOCITY_LIMIT={VLIM}\n")
        text = text.replace("gen_mechanics_ten_hands", WANDB).replace(
            "default configuration, 5000 epochs. Generated by make_runs.py.",
            f"Wuji v2 '{tag}', uniform dynamics with every hand joint capped at {VLIM:g} rad/s"
            + (f", hand-target moving average {ema:g}" if ema is not None else "")
            + ((", obs/action delay and object-state delay on, no object pose noise" if nopn else
                ", obs/action delay and object-state delay+noise on") if dr else "")
            + (f", keypoint reward scale {kp:g}" if kp is not None else "")
            + (f", hand joint-velocity penalty {pen:g}" if pen is not None else "")
            + "".join({"obs": ", obs delay only", "action": ", action delay only",
                       "object": ", object-state delay only (no pose noise)"}[d] for d in delays) + ", canonical palm "
            "frame + palm_extents, 5000 epochs. Generated by 06oct_wuji_5rads/make_runs.py.")
        p = HERE / f"left_{run}.sub"
        p.write_text(text)
        print(p)
