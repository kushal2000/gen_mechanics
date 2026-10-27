"""Write one left-hand training .sub per hand: the default configuration, 5000 epochs.

Default (user, 30 Sep): joint transformer d64 / 4 layers / 1 head / ff_mult 2, mu head [64], value head
[512,256], no ghost-action mask, no activation checkpointing; trimmed observation (with the
keypoints_rel_ee fix); no fall penalty; gamma 0.998, adaptive lr from 5e-4, 5 mini-epochs, horizon 16;
5 deg tolerance. Budget 5000 epochs, no chained continuation.

    .venv_isaacsim/bin/python experiments/old_experiments/30sep_ten_hands/make_runs.py [--vlim V] <hand> [...]

--uniform uses the uniform-dynamics spec (<hand>_left_uniform_handonly: 1.0 N.m, 10 rad/s, gen-SHARPA gains,
1750 kg/m^3 links) and 0.5 friction on every contact, and suffixes the run _uniform.
--vlim V caps every hand joint's speed at V rad/s in the solver (physics.hand_velocity_limit) and
suffixes the run _v<V>. Without it the vendor URDF's per-joint caps apply (2.0 Shadow .. 15 SHARPA).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "experiments/_shared"))
from ten_hands import LEFT, LOGS, OBS_LIST, TEMPLATE, UNIFORM, UNIFORM_HYDRA  # noqa: E402,F401

if __name__ == "__main__":
    args, vlim, uniform = sys.argv[1:], None, False
    while args[:1] in (["--vlim"], ["--uniform"]):
        if args[0] == "--vlim":
            vlim, args = args[1], args[2:]
        else:
            uniform, args = True, args[1:]
    for hand in args or (UNIFORM if uniform else LEFT):
        run = hand + ("_uniform" if uniform else "") + (f"_v{vlim}" if vlim else "")
        spec = f"{hand}_left_uniform_handonly" if uniform else LEFT[hand]
        p = HERE / f"left_{run}.sub"
        p.write_text(TEMPLATE.format(hand=hand, run=run, spec=spec, obs=OBS_LIST, logs=LOGS, repo=REPO,
                                     hydra_extra=UNIFORM_HYDRA if uniform else "",
                                     vlim=f"export HAND_VELOCITY_LIMIT={float(vlim)}\n" if vlim else ""))
        print(p)
