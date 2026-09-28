"""Watch an in-hand reorientation policy: one env, one checkpoint, render in viser.

The hand-only task -- a fixed hand, no arm, one cube already in it, goal a
uniform random SO(3) orientation resampled on every success. The palm is tilted
45.25 degrees off vertical -- the IsaacLab inhand reference's own palm tilt --
about the palm WIDTH, so gravity's in-plane pull runs at the fingertips.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m coevolution.eval.play_inhand \
        --checkpoint debug_outputs/train_logs/24sep_inhand_reorientation/<run>/rank_0/<name>/nn/<ckpt>.pth \
        --port 8081

The robot_spec comes from the run and already carries its "handonly:" prefix,
which is what drops the arm joints -- so do NOT pass a bare population to
--population here, or the env will build a 37-joint robot for a policy that
expects 30 and the articulation check will fail. Prefix it:

    ... --population handonly:assets/populations/sharpa_capsule.json

What to watch for, in the status panel:

  keypoint dist  is a pure ROTATION residual under this task (object and goal
                 keypoints are compared about their own centres), so it does not
                 move when the cube slides around the palm -- only when it turns.
  lifted         is latched 1 from reset here. If it ever reads 0 the keypoint
                 reward is dead and the run is not learning what you think.
  tolerance      is fixed at 0.0039 (0.1 rad on a 45 mm cube). No curriculum.
"""
from coevolution.eval.viser_play import build_parser, play

TASK = "GenMech-InHandReorient-Direct-v0"

if __name__ == "__main__":
    parser = build_parser(__doc__.splitlines()[0])
    known, hydra_args = parser.parse_known_args()
    play(TASK, known, hydra_args)
