"""Watch a pose-reaching policy: one env, one checkpoint, Isaac's render in viser.

The original task -- iiwa14 plus a generated hand, lift a tool off the table and
drive it through SE(3) goals. Everything about the env comes from the training
run's own saved config, so what you watch is what the checkpoint trained in.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m coevolution.eval.play_pose_reach \
        --checkpoint debug_outputs/train_logs/17sep_coevolution/<run>/rank_0/<name>/nn/<ckpt>.pth \
        --port 8080

Watch ONE design out of a population by pointing --population at a one-design
file; the policy is multi-embodiment, so it drives whatever it is given:

    ... --population assets/populations/sharpa_capsule.json

Then open the printed URL. Kit boot plus scene build is a minute or so at one
env. Run one Isaac Sim instance per GPU.
"""
from coevolution.eval.viser_play import build_parser, play

TASK = "GenMech-PoseReach-Direct-v0"

if __name__ == "__main__":
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda:0")
    play(TASK, parser.parse_args())
