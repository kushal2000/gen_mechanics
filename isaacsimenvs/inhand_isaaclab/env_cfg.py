"""A ladder from IsaacLab's in-hand task toward ours, one delta per rung.

Each class changes exactly ONE thing from the class above it. The agent config stays
IsaacLab's at every rung (see this package's ``__init__``), so a rung that stops
learning is a fact about the environment change rather than about a different learner.

Why a ladder and not a port: our task differs from the reference in a dozen ways at
once -- decimation, episode length, reward shape, success threshold, object placement,
observation, hand -- and it does not learn. A single diff cannot say which difference
matters. Changing one at a time can.

The rungs are ordered cheapest-to-suspect first: timing before reward, reward before
geometry. Where a delta cannot be expressed as a config change on their cfg it stops
being a rung and becomes a note in the README; that boundary is itself informative,
because it marks the differences that are structural rather than parametric.

Read the README in this folder for the measured reference values and what each rung is
expected to cost.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.allegro_env_cfg import (
    AllegroCubeEnvCfg,
)


@configclass
class InHandIsaacLabReferenceCfg(AllegroCubeEnvCfg):
    """Rung 0: the reference, UNCHANGED.

    Present as a rung of its own rather than pointing at their task id, so that every
    rung including the baseline is launched by the same script with the same agent
    config. If rung 0 does not learn here, nothing below it is worth reading and the
    problem is our install or our launcher, not our task.

    Verified in this install: 16 joints, 21 bodies, 72-d observation, 16-d action,
    decimation 4, dt 1/120, episode 20 s.
    """


@configclass
class InHandIsaacLabDecimationCfg(InHandIsaacLabReferenceCfg):
    """Rung 1: our control rate. decimation 4 -> 2 at sim dt 1/120, so 60 Hz control instead of 30 Hz.

    Cheap to suspect and cheap to test. It doubles the policy's decisions per second
    and halves the torque applied per decision, and it is the difference most likely to
    be invisible in a reward curve while changing what the policy can do.
    """

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        # Their cfg ties the render interval to decimation; keep that relationship
        # rather than leaving it at the parent's value.
        self.sim.render_interval = self.decimation


@configclass
class InHandIsaacLabEpisodeCfg(InHandIsaacLabDecimationCfg):
    """Rung 2: our episode budget. 20 s -> 10 s per goal.

    Both theirs and ours reset the clock on a goal hit, so this is time to reach EACH
    goal, not the whole episode. Halving it halves the exploration a policy gets per
    goal before truncation.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0


@configclass
class InHandIsaacLabToleranceCfg(InHandIsaacLabEpisodeCfg):
    """Rung 3: our success threshold, as an angle.

    Theirs is 0.1 rad on the orientation command's success term. Ours is set in degrees
    and converted; 20 degrees is what the current runs use, and the 5-vs-20 comparison
    on our own task showed 66x more goals per episode at 20. So this rung is expected
    to make the task EASIER, not harder -- which is the point of putting it on the
    ladder: if the reference's learning collapses anyway, the threshold is not what our
    task's difficulty rests on.
    """

    SUCCESS_THRESHOLD_DEG = 20.0

    def __post_init__(self):
        super().__post_init__()
        threshold = math.radians(self.SUCCESS_THRESHOLD_DEG)
        # The command term owns the threshold in their formulation, and the reward and
        # the termination both read it from there, so setting it in one place is enough.
        self.commands.object_pose.orientation_success_threshold = threshold


__all__ = [
    "InHandIsaacLabDecimationCfg",
    "InHandIsaacLabEpisodeCfg",
    "InHandIsaacLabReferenceCfg",
    "InHandIsaacLabToleranceCfg",
]


@configclass
class InHandIsaacLabSapgCfg(InHandIsaacLabReferenceCfg):
    """Rung 0b: the reference env run by OUR learner (SAPG), not an environment delta.

    This is the control the rest of the ladder cannot provide. Every rung below changes
    the ENV and holds the learner at IsaacLab's PPO, so none of them can say whether SAPG
    -- which all seven of our in-hand jobs use and which the reference run does not --
    costs anything. Here the env is the reference's, and only the learner changes
    (``train.py --sapg``: expl_type mixed_expl_learn_param, entropy exploration reward at
    coef scale 0.002, four blocks of 2048, use_others_experience lf, off_policy_ratio 1.0,
    fixed_sigma coef_cond -- the values in coevolution/cfg/train/PoseReachSAPG*.yaml).

    THE ONE ENV CHANGE, and why it is inert. Under mixed_expl the fork concatenates the
    block embedding onto BOTH obs and states (a2c_common.py:604-615) with no check that
    states exist, so the reference env raises ``KeyError: 'states'`` on the first epoch:
    it declares only a ``policy`` observation group, where our env returns policy and
    critic. IsaacLab's wrapper exposes ``states`` exactly when a ``critic`` group is
    present (rl_games.py:131), so one is added here as a copy of the policy group.

    It cannot change the learning problem: IsaacLab's agent config has no
    ``central_value_config``, so rl_games builds no central value network and nothing
    reads these states as a critic input. They exist only to be concatenated with the
    embedding and dropped. A copy of the policy group -- noise terms included -- is used
    rather than a privileged clean-state group precisely so that this stays true; a
    different observation here would be a second delta smuggled in beside the learner.
    """

    def __post_init__(self):
        super().__post_init__()
        import copy

        self.observations.critic = copy.deepcopy(self.observations.policy)


# ---------------------------------------------------------------------------
# Independent single-delta probes off rung 0.
#
# The numbered rungs above CHAIN (each one keeps the deltas below it), which is
# right when the question is "how far toward our config can we get". These are
# different: each one changes exactly one thing FROM THE REFERENCE and nothing
# else, so they run in parallel and attribute cleanly. Mixing the two styles in
# one LADDER is deliberate; the docstrings say which is which.
# ---------------------------------------------------------------------------

# 0.045 / 0.060. Their cube is a USD (dex_cube_instanceable.usd), so size is a spawn
# scale rather than a number in the cfg.
CUBE_SCALE = 45.0 / 60.0


@configclass
class InHandIsaacLabCube45Cfg(InHandIsaacLabReferenceCfg):
    """Probe: their 60 mm cube shrunk to OUR 45 mm, at constant density.

    Why it is a suspect: a 60 mm cube on a ~100x117 mm Allegro palm is easy to cage --
    it bridges between the fingertips and the palm, and the fingers do not have to curl
    far. At 45 mm the same hand must close much further and the object can slip between
    fingers. The reference solves possession in ~140 epochs (episode length 19 -> 560)
    while every one of our runs still drops the cube in 77-100% of episodes after 5000,
    so anything that makes the object harder to hold is worth isolating.

    MASS IS PINNED EXPLICITLY, and that is not a second delta -- it is what keeps this
    one honest. The cfg declares density 400, which at 60 mm would imply 0.086 kg, but
    the body actually reports 0.264 kg: the USD carries its own mass and inertia and wins
    over the declared density. So scaling the geometry alone would leave the mass at
    0.264 kg, turning a 45 mm cube into a 2900 kg/m3 one -- a size change AND a nearly
    3x density change in a single rung. Setting the mass to the measured value times
    scale^3 keeps the material constant, which is what "smaller cube" should mean.
    Whether this override actually takes (the same USD-wins question) is checked at
    startup by train.py, which prints the measured edge and mass.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.object.spawn.scale = (CUBE_SCALE,) * 3
        # 0.21412 kg is the reference's NOMINAL mass at 60 mm -- the MEAN over 512 envs,
        # because object_scale_mass multiplies each env's mass by U(0.4, 1.6) and a single
        # env reads anywhere in 0.086-0.345. An earlier version of this line used 0.264,
        # which was one env's draw, and produced 1226 kg/m3 against the reference's 991:
        # a 24% density increase riding along inside a size rung. Same density, 0.75x edge.
        self.scene.object.spawn.mass_props = sim_utils.MassPropertiesCfg(
            mass=0.21412 * CUBE_SCALE**3)


@configclass
class InHandIsaacLabObjInitCfg(InHandIsaacLabReferenceCfg):
    """Probe: the object starts at a RANDOM orientation instead of identity.

    Their reset_object jitters position by +/-1 cm and never touches rotation, so the
    cube begins every episode at rot=(1,0,0,0) and the required rotation IS the sampled
    goal -- one 2-parameter family, from one canonical pose, every episode. Ours draws a
    uniform start and a uniform goal, so the relative rotation covers the full group from
    an arbitrary pose. That is a large difference in how much a policy can reuse between
    episodes, and it bites hardest early, which is exactly where ours is stuck.

    NOTE WHAT THIS DISTRIBUTION IS. reset_root_state_uniform samples roll/pitch/yaw
    independently and uniformly (events.py:1100), which is NOT Haar-uniform on SO(3) --
    it concentrates near the poles. Ours uses random_orientation, which is Haar. This
    rung is therefore "identity -> random", not "identity -> exactly our distribution";
    closing that second gap needs a custom event and is only worth a rung if this one
    moves the needle. Measured earlier: their goal family averages 127.9 deg of required
    rotation against 126.4 for Haar-uniform, so the distributions are close in angle
    even where they differ in shape.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_object.params["pose_range"] = {
            "x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01],
            "roll": [-math.pi, math.pi], "pitch": [-math.pi, math.pi],
            "yaw": [-math.pi, math.pi],
        }


@configclass
class InHandIsaacLabProgressRewardCfg(InHandIsaacLabReferenceCfg):
    """Probe: OUR reward's shape -- progress against a record -- instead of their dense term.

    The last untested suspect. On the reference env, SAPG, gamma (0.99 and 0.98), cube size
    and random object init all came out neutral or BETTER than the reference's own choice,
    so nothing parametric explains why our task never learns possession. What is left is the
    representation (tested separately by ARCH=mlp on our env) and this: their reward pays
    1/(dtheta+0.1) every step, ours pays only for beating a record.

    ONE DELTA. track_orientation_inv_l2 is removed and track_orientation_progress replaces
    it on the SAME quantity (their quaternion angle, not our keypoint residual), at a weight
    chosen so the per-episode magnitude matches what it replaces. success_bonus stays 250,
    the penalties are untouched, and the env is otherwise rung 0. See progress_reward.py for
    why the weight is 267 and why the record resets on a goal resample.

    The prediction, if the reward shape is the cause: episode length stops climbing to ~560
    and instead wanders in a band the way ours does (577945 has oscillated between 60 and 190
    of 600 for 7800 epochs without trending), because a reward that pays nothing for holding
    a cube it cannot currently improve on gives no gradient toward keeping it.
    """

    def __post_init__(self):
        super().__post_init__()
        from .progress_reward import track_orientation_progress

        self.rewards.track_orientation_inv_l2 = None
        self.rewards.track_orientation_progress = RewTerm(
            func=track_orientation_progress,
            # 28 * 30 / pi: the dt multiplication inside RewardManager applies to a delta
            # too, and the deltas over one goal sum to at most the initial error (~pi rad).
            weight=267.0,
            params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
        )


@configclass
class InHandIsaacLabCombinedCfg(InHandIsaacLabSapgCfg):
    """All the deltas that individually came out neutral-or-better, applied AT ONCE.

    Every probe so far changes one thing and each was fine on its own, measured on the
    reference env at matched epochs: SAPG >= PPO, gamma 0.98 and 0.99 both far ahead on goal
    rate, cube 45 mm ahead, random object init ahead. Individually clean does not imply
    jointly clean -- a combination can interact where the parts do not, and the possession
    cost of a short horizon (drop 0.075 -> 0.432 as gamma goes 0.998 -> 0.98) might compound
    with a smaller cube that is harder to cage.

    So this is four deltas toward our configuration in one env:
      * OUR learner      -- SAPG, inherited from InHandIsaacLabSapgCfg (which also supplies
                            the critic observation group the fork needs; see that docstring)
      * OUR horizon      -- gamma 0.98, passed by the .sub, not set here: 50 steps = 1.67 s
                            at their 30 Hz, matching what our policies actually see
      * OUR cube         -- 45 mm at the reference's own density, as in Cube45Cfg
      * OUR object start -- random orientation rather than identity, as in ObjInitCfg

    What is deliberately NOT here: the reward shape, the keypoint success metric and the
    joint-token observation. Those are the three differences that remain untested-or-implicated,
    and this run is the control that says how much of the gap they have to account for. If
    this configuration still reaches ~500 episode length and climbing goals, then everything
    except reward, metric and observation is exonerated jointly as well as singly.
    """

    def __post_init__(self):
        super().__post_init__()
        # --- our cube: 45 mm at their density (see InHandIsaacLabCube45Cfg) -------------
        self.scene.object.spawn.scale = (CUBE_SCALE,) * 3
        self.scene.object.spawn.mass_props = sim_utils.MassPropertiesCfg(
            mass=0.21412 * CUBE_SCALE**3)

        # --- our object start: random orientation (see InHandIsaacLabObjInitCfg) --------
        self.events.reset_object.params["pose_range"] = {
            "x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01],
            "roll": [-math.pi, math.pi], "pitch": [-math.pi, math.pi],
            "yaw": [-math.pi, math.pi],
        }


@configclass
class InHandIsaacLabKeypointRewardCfg(InHandIsaacLabReferenceCfg):
    """Probe: OUR REWARD on their env -- progress against a record, on the keypoint residual.

    The paired arm of progressrew. That rung put our progress SHAPE on their quaternion
    angle and slowed possession learning ~15x without stopping it (episode length still
    climbing past 400). This one also swaps the METRIC to our keypoint residual, so the two
    together separate shape from metric:
      progressrew  : progress shape, true angle
      keypointrew  : progress shape, keypoint residual   <- this
    Weight 267 and the record-keeping are identical; the residual is divided by the keypoint
    radius so its small-angle scale equals theta. Success and termination stay on their
    quaternion angle. See keypoint_reward.py.
    """

    def __post_init__(self):
        super().__post_init__()
        from .keypoint_reward import track_orientation_keypoint_progress

        self.rewards.track_orientation_inv_l2 = None
        self.rewards.track_orientation_keypoint_progress = RewTerm(
            func=track_orientation_keypoint_progress,
            weight=267.0,       # same as progressrew; see progress_reward.py for why 267
            params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
        )


@configclass
class InHandIsaacLabOurSpawnCfg(InHandIsaacLabReferenceCfg):
    """Probe: their cube spawned where OUR env spawns it, relative to the same palm_link.

    Measured with the same probe in both envs, in palm_link's frame (+x out the fingers,
    +z out of the palm):
        reference  cube centre (102, 9, 57) mm -- over the fingertip centroid (97, -19, 11),
                   resting on the curled finger pads
        ours       cube centre (65, -14, 85) mm -- 32 mm back toward the palm centre
                   (in_hand_fingertip_fraction 0.5) and ~60 mm above the tips, so it drops
                   onto the palm slab
    Both are stable under a do-nothing hand (99.6% vs 98.0% still held at 10 s), so this is
    not "starts already falling". The question is whether starting on the PALM rather than on
    the FINGERS is what keeps our policies from ever holding on, while every reward/metric
    change on this env only slowed possession learning.

    ONE DELTA: init_state.pos only. Their 60 mm cube, identity start orientation, +-1 cm reset
    jitter, hand reset pose and everything else stay rung 0. In-plane position is ours
    exactly; height keeps our cube's BOTTOM at the same 51.3 mm above the palm (our 84.8 mm
    centre minus our 33.5 mm mean half-reach), plus their 30 mm half-edge.

    Converted with palm_link's measured env-frame pose (-0.008776, -0.082421, 0.512699),
    wxyz (0.706322, 0.033309, 0.033308, -0.706322). Sanity check on that transform: their own
    (0, -0.19, 0.56) maps back to (102.7, 8.8, 57.2) mm, matching the probe's (102.2, 8.6, 56.8).
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.object.init_state.pos = (-0.022576, -0.154883, 0.587512)


@configclass
class InHandIsaacLabAllOursCfg(InHandIsaacLabSapgCfg):
    """EVERY difference expressible on their env, at once. The bisection's far end.

    Eight single deltas have each failed to reproduce our failure here (SAPG, gamma 0.99/0.98,
    45 mm cube, random start orientation, our progress shape, our keypoint metric, our spawn).
    So either the cause is a COMBINATION, or it is something this env cannot express (the hand
    model, the observation, sim parameters). This rung decides which: it stacks all of ours --

      learner   SAPG (inherited; supplies the critic obs group the fork needs), gamma 0.98 and
                OUR PPO hyperparameters (both passed by the .sub: --gamma 0.98 --our-hparams)
      timing    decimation 2 (60 Hz) and a 10 s episode = 600 steps, as ours
      action    no EMA (alpha 1.0), as ours after run_rank.sh's override
      object    45 mm cube at their density; random start orientation
      spawn     OUR palm-frame placement for a 45 mm cube, centre (65.1, -13.8, 84.8) mm
      reward    OUR shape and metric: progress on the keypoint residual, weight 267

    -- on their hand, their observation, their sim. If this still learns possession, the answer
    lies in those three; if it stalls, bisect back toward the reference from here.
    """

    def __post_init__(self):
        super().__post_init__()
        from .keypoint_reward import track_orientation_keypoint_progress

        # timing -- mirrors InHandIsaacLabDecimationCfg / EpisodeCfg
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        # action: no smoothing
        self.actions.joint_pos.alpha = 1.0
        # object: 45 mm at their density, random start orientation
        self.scene.object.spawn.scale = (CUBE_SCALE,) * 3
        self.scene.object.spawn.mass_props = sim_utils.MassPropertiesCfg(
            mass=0.21412 * CUBE_SCALE**3)
        self.events.reset_object.params["pose_range"] = {
            "x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01],
            "roll": [-math.pi, math.pi], "pitch": [-math.pi, math.pi],
            "yaw": [-math.pi, math.pi],
        }
        # spawn: our palm-frame centre for the 45 mm cube, through palm_link's measured pose
        self.scene.object.init_state.pos = (-0.022576, -0.155212, 0.590996)
        # reward: ours, shape and metric
        self.rewards.track_orientation_inv_l2 = None
        self.rewards.track_orientation_keypoint_progress = RewTerm(
            func=track_orientation_keypoint_progress, weight=267.0,
            params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
        )


# The exact USD our env builds from assets/urdf/unified_commercial_hands/allegro/allegro.urdf
# (assembly._convert_fixed_robot -> flatten_robot_usd): self-collision filters and our 2 mm
# contact offset baked in, flattened, no external references. Committed under assets/ because
# our env writes it only to a per-run temp dir.
OUR_ALLEGRO_USD = str(__import__("pathlib").Path(__file__).resolve().parents[2]
                      / "assets/usd/allegro_handonly/robot_flat.usd")


@configclass
class InHandIsaacLabOurHandCfg(InHandIsaacLabReferenceCfg):
    """Probe: OUR Allegro -- the physical hand our policies train on -- in their env.

    Read live from PhysX in both envs, the two Allegros are not the same object:
                              reference              ours
        joint torque cap      0.50 N.m               0.35 N.m (the URDF's effort)
        fingertip link mass   0.10-0.11 kg each      0.021 kg (index/middle/ring), 0.10 thumb
        hand total mass       2.14 kg                1.67 kg
        armature              0                      0.001
        contact offset        USD default            2 mm on every robot shape
        collision geometry    their USD              our unified URDF (palm box, biotac meshes)
    Joint limits, stiffness, damping and joint friction agree.

    ONE DELTA: the robot asset and its drive settings. Spawned from OUR_ALLEGRO_USD, rooted at
    palm_link, placed at the palm_link pose measured in this env -- so the palm, and therefore
    the cube's spawn relative to it, is where rung 0 has it. Actuators are ours (stiffness 3.0,
    damping 0.1, armature 0.001, friction 0, torque cap from the asset). Their rigid/articulation
    props and their robot DR events are kept; note their friction DR (U(0.7, 1.3) on every robot
    body) therefore replaces the 1.5 tip / 0.5 palm materials our env applies at runtime.

    One side effect worth knowing: their object_out_of_reach measures distance to the robot
    ROOT, which is now palm_link rather than allegro_mount (~84 mm apart) -- small against the
    0.3 m threshold.
    """

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.actuators import ImplicitActuatorCfg

        rob = self.scene.robot
        rob.spawn = sim_utils.UsdFileCfg(
            usd_path=OUR_ALLEGRO_USD,
            activate_contact_sensors=False,
            rigid_props=rob.spawn.rigid_props,
            articulation_props=rob.spawn.articulation_props,
        )
        rob.init_state.pos = (-0.008776, -0.082421, 0.512699)
        rob.init_state.rot = (0.706322, 0.033309, 0.033308, -0.706322)
        rob.actuators = {
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=3.0, damping=0.1, armature=0.001,
                friction=0.0),
        }


@configclass
class InHandIsaacLabKeypointSpawnCfg(InHandIsaacLabKeypointRewardCfg):
    """Probe: OUR reward (progress on the keypoint residual) + OUR spawn. Two deltas, on purpose.

    Each alone has been run: our reward alone learns possession slowly but steadily
    (keypointrew, 328/600 by epoch 500), our spawn alone changes nothing (ourspawn tracks the
    baseline). The mechanism found in our own env says the damage is the INTERACTION -- a cube
    dropped onto the palm tumbles, a tumbling cube banks progress on the way out, so under a
    progress reward dropping out-earns holding. Here that is tested on their env: prediction,
    possession stalls or collapses. Spawn is the 60 mm version of our placement, as ourspawn.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.object.init_state.pos = (-0.022576, -0.154883, 0.587512)


# OUR palm_link world orientation, hand_only_base_rot(allegro PALM_FRAME) (wxyz): 5.00 deg
# off vertical, same tilt direction as theirs (0.706322, 0.033309, 0.033308, -0.706322) = 5.40.
OUR_PALM_QUAT = (0.7064337722128922, 0.030843564597231896, 0.030843564597231896,
                 -0.7064337722128922)
THEIR_PALM_POS = (-0.008776, -0.082421, 0.512699)


@configclass
class InHandIsaacLabOurDefaultCfg(InHandIsaacLabSapgCfg):
    """Their env moved ALL THE WAY to our default in-hand setup, except observation and network.

    Target is the Allegro control 585759 (our env, 20 deg, no fall penalty). Launch with
    ``--sapg --sapg-block-size 2048 --num-envs 12288 --gamma 0.99 --our-hparams
    --mixed-precision`` for the learner half. Terms live in ourdefault.py, each citing the code
    in our env it reproduces. Deltas from rung 0:

      learner    SAPG with 6 blocks of 2048 (inherited critic group), gamma 0.99, our PPO
                 hyperparameters, mixed precision, 12288 envs                 [flags]
      timing     decimation 2 at dt 1/120 = 60 Hz, 10 s episode = 600 steps
      action     no EMA (alpha 1.0)
      hand       OUR physical Allegro USD + our drives (0.35 N.m cap, light tips, armature),
                 palm at OUR 5.00 deg orientation (theirs 5.40), at their palm position
      friction   ours, fixed: robot 0.5, fingertips 1.5, object 1.0; none of their DR
                 (friction, object mass x U(0.4,1.6), robot mass, gain log-U(0.3,3))
      object     45 mm cube at OUR mass 0.0455625 kg (500 kg/m3)
      spawn      our palm-frame centre (65.1, -13.8, 84.8) mm; Haar-uniform start orientation;
                 +-5 mm in-plane jitter
      hand reset our lerp 0.1 toward U(lo, hi), joint velocity U(-0.5, 0.5)
      goals      Haar-uniform SO(3); success = keypoint residual <= 0.0203 m (our 20 deg);
                 the per-goal timer reset
      reward     OURS at our magnitudes: keypoint progress x2000/m, 1000 per success,
                 -0.0003 x L1 hand joint velocity; theirs removed
      drop       > 0.3 m from the palm (ours), not from the root
      obs noise  off (ours runs with obs DR off)

    NOT changed: their 72-d observation and their MLP. If this reproduces our collapse, the
    cause is in the env/reward/learner; if it learns, it is in our observation or network.
    """

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        from isaaclab_tasks.manager_based.manipulation.inhand import mdp as ih_mdp

        from . import ourdefault as od
        from .keypoint_reward import KEYPOINT_SCALE

        # --- timing ------------------------------------------------------------------
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        # --- action ------------------------------------------------------------------
        self.actions.joint_pos.alpha = 1.0
        # --- hand: our USD, our drives, our palm orientation -------------------------
        rob = self.scene.robot
        rob.spawn = sim_utils.UsdFileCfg(
            usd_path=OUR_ALLEGRO_USD, activate_contact_sensors=False,
            rigid_props=rob.spawn.rigid_props, articulation_props=rob.spawn.articulation_props)
        rob.init_state.pos = THEIR_PALM_POS
        rob.init_state.rot = OUR_PALM_QUAT
        rob.actuators = {"fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=3.0, damping=0.1, armature=0.001, friction=0.0)}
        # --- object ------------------------------------------------------------------
        self.scene.object.spawn.scale = (CUBE_SCALE,) * 3
        self.scene.object.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.0455625)
        # our palm-frame centre (65.1, -13.8, 84.8) mm through OUR palm orientation
        self.scene.object.init_state.pos = (-0.022576, -0.154664, 0.591502)
        # --- DR off, our fixed frictions ---------------------------------------------
        ev = self.events
        ev.robot_scale_mass = None
        ev.robot_joint_stiffness_and_damping = None
        ev.object_scale_mass = None
        mat = dict(restitution_range=(0.0, 0.0), num_buckets=1)
        ev.robot_physics_material = EventTerm(
            func=ih_mdp.randomize_rigid_body_material, mode="startup",
            params=dict(asset_cfg=SceneEntityCfg("robot", body_names=".*"),
                        static_friction_range=(0.5, 0.5), dynamic_friction_range=(0.5, 0.5), **mat))
        ev.fingertip_physics_material = EventTerm(
            func=ih_mdp.randomize_rigid_body_material, mode="startup",
            params=dict(asset_cfg=SceneEntityCfg("robot", body_names=".*_link_3"),
                        static_friction_range=(1.5, 1.5), dynamic_friction_range=(1.5, 1.5), **mat))
        ev.object_physics_material = EventTerm(
            func=ih_mdp.randomize_rigid_body_material, mode="startup",
            params=dict(asset_cfg=SceneEntityCfg("object", body_names=".*"),
                        static_friction_range=(1.0, 1.0), dynamic_friction_range=(1.0, 1.0), **mat))
        # --- resets ------------------------------------------------------------------
        ev.reset_object = EventTerm(func=od.reset_object_like_ours, mode="reset",
                                    params={"jitter": 0.005})
        ev.reset_robot_joints = EventTerm(func=od.reset_hand_like_ours, mode="reset",
                                          params={"pos_interval": 0.1, "vel_interval": 0.5})
        # --- goals + success ---------------------------------------------------------
        edge = 0.06 * CUBE_SCALE
        tol = edge * math.sqrt(3.0) * math.sin(math.radians(20.0) / 2.0) * KEYPOINT_SCALE
        old = self.commands.object_pose
        self.commands.object_pose = od.KeypointGoalCommandCfg(
            asset_name="object", init_pos_offset=old.init_pos_offset,
            update_goal_on_success=True, orientation_success_threshold=0.1,
            make_quat_unique=False, marker_pos_offset=old.marker_pos_offset, debug_vis=False,
            keypoint_tol_m=tol, reset_timer_on_success=True)
        # --- reward: ours, at our magnitudes (weight = ours / step_dt, step_dt = 1/60) -----
        step_dt = self.sim.dt * self.decimation
        for k in ("track_orientation_inv_l2", "success_bonus", "joint_vel_l2", "action_l2",
                  "action_rate_l2"):
            setattr(self.rewards, k, None)
        cmd = {"command_name": "object_pose"}
        self.rewards.keypoint_progress = RewTerm(func=od.keypoint_progress_m,
                                                 weight=2000.0 / step_dt, params=dict(cmd))
        self.rewards.goal_bonus = RewTerm(func=od.keypoint_success,
                                          weight=1000.0 / step_dt, params=dict(cmd))
        self.rewards.hand_velocity = RewTerm(func=od.hand_joint_vel_l1,
                                             weight=-0.0003 / step_dt)
        # --- drop: from the palm -----------------------------------------------------
        self.terminations.object_out_of_reach = DoneTerm(
            func=od.object_away_from_palm, params={"threshold": 0.3})
        # --- no observation noise ----------------------------------------------------
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False


# ---------------------------------------------------------------------------------------------
# 29sep_what_breaks_reference: LEAVE-ONE-OUT from ourdefault.
#
# ourdefault (their env + everything of ours but observation and network) reproduces our env's
# learn-then-collapse; every single change applied alone to the reference did not. Each class
# below is ourdefault with ONE GROUP of changes reverted to the reference. A group whose revert
# keeps possession is one the break needs. Learner-side groups (gamma, PPO hyperparameters, SAPG)
# are reverted by the .sub's flags, not here. See experiments/old_experiments/29sep_what_breaks_reference/.
# ---------------------------------------------------------------------------------------------


def _reference_cfg():
    """A fresh rung-0 cfg to copy reverted groups from (post_init already run)."""
    return InHandIsaacLabReferenceCfg()


def _set_our_reward_weights(cfg) -> None:
    """Re-derive our reward weights for the cfg's current step_dt (weight = ours / step_dt)."""
    step_dt = cfg.sim.dt * cfg.decimation
    if getattr(cfg.rewards, "keypoint_progress", None) is not None:
        cfg.rewards.keypoint_progress.weight = 2000.0 / step_dt
    if getattr(cfg.rewards, "goal_bonus", None) is not None:
        cfg.rewards.goal_bonus.weight = 1000.0 / step_dt
    if getattr(cfg.rewards, "hand_velocity", None) is not None:
        cfg.rewards.hand_velocity.weight = -0.0003 / step_dt


def _keypoint_tol_for_edge(edge_m: float, deg: float = 20.0) -> float:
    from .keypoint_reward import KEYPOINT_SCALE
    return edge_m * math.sqrt(3.0) * math.sin(math.radians(deg) / 2.0) * KEYPOINT_SCALE


@configclass
class LooRewardCfg(InHandIsaacLabOurDefaultCfg):
    """Revert the REWARD SHAPE: their dense 1/(theta+0.1), their three penalties, and a 250
    success bonus -- fired on OUR success test (the success test is a separate group), at their
    weight. Their terms carry their own dt scaling, now at our 60 Hz."""

    def __post_init__(self):
        super().__post_init__()
        from . import ourdefault as od
        ref = _reference_cfg()
        for k in ("keypoint_progress", "goal_bonus", "hand_velocity"):
            setattr(self.rewards, k, None)
        for k in ("track_orientation_inv_l2", "joint_vel_l2", "action_l2", "action_rate_l2"):
            setattr(self.rewards, k, getattr(ref.rewards, k))
        self.rewards.success_bonus = RewTerm(func=od.keypoint_success, weight=250.0,
                                             params={"command_name": "object_pose"})


@configclass
class LooSuccessCfg(InHandIsaacLabOurDefaultCfg):
    """Revert SUCCESS TEST + GOALS + TIMER: their command term -- true quat angle, rot_x*rot_y
    goals, no per-goal timer reset -- but at OUR 20 deg, not their 0.1 rad: the tolerance is its
    own group (LooToleranceCfg). Our reward's bonus fires on this success, at our weight."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab_tasks.manager_based.manipulation.inhand import mdp as ih_mdp
        ref = _reference_cfg()
        self.commands.object_pose = ref.commands.object_pose
        self.commands.object_pose.debug_vis = False
        self.commands.object_pose.orientation_success_threshold = math.radians(20.0)
        self.rewards.goal_bonus = RewTerm(func=ih_mdp.success_bonus, weight=1.0,
                                          params={"command_name": "object_pose"})
        _set_our_reward_weights(self)


@configclass
class LooTimingCfg(InHandIsaacLabOurDefaultCfg):
    """Revert TIMING: decimation 4 (30 Hz), 20 s episodes (600 steps), action EMA 0.95. Our reward
    weights are re-derived so our per-policy-step magnitudes are unchanged."""

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.actions.joint_pos.alpha = 0.95
        _set_our_reward_weights(self)


@configclass
class LooHandCfg(InHandIsaacLabOurDefaultCfg):
    """Revert the HAND: their Allegro USD, their actuators (0.5 N.m cap), their mount pose (palm
    5.40 deg). Our fixed frictions are still applied to its bodies (DR is a separate group)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _reference_cfg().scene.robot


@configclass
class LooObjectCfg(InHandIsaacLabOurDefaultCfg):
    """Revert the OBJECT: their 60 mm cube at their own (USD) mass. The keypoint threshold is
    re-derived for the 60 mm edge so '20 deg' still means 20 deg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.object.spawn = _reference_cfg().scene.object.spawn
        self.commands.object_pose.keypoint_tol_m = _keypoint_tol_for_edge(0.06)


@configclass
class LooSpawnResetCfg(InHandIsaacLabOurDefaultCfg):
    """Revert SPAWN + RESETS: their object start (over the fingertips, identity orientation, +-1 cm)
    and their hand reset (uniform within 20% of each joint's range about the default)."""

    def __post_init__(self):
        super().__post_init__()
        ref = _reference_cfg()
        self.scene.object.init_state.pos = ref.scene.object.init_state.pos
        self.scene.object.init_state.rot = ref.scene.object.init_state.rot
        self.events.reset_object = ref.events.reset_object
        self.events.reset_robot_joints = ref.events.reset_robot_joints


@configclass
class LooDrCfg(InHandIsaacLabOurDefaultCfg):
    """Revert DOMAIN RANDOMIZATION + FRICTION: their startup DR on friction (U(0.7,1.3), robot and
    object), object mass (x U(0.4,1.6)), robot mass (x U(0.95,1.05)) and gains (log-U(0.3,3))."""

    def __post_init__(self):
        super().__post_init__()
        ref = _reference_cfg()
        self.events.fingertip_physics_material = None
        for k in ("robot_physics_material", "robot_scale_mass", "robot_joint_stiffness_and_damping",
                  "object_physics_material", "object_scale_mass"):
            setattr(self.events, k, getattr(ref.events, k))


@configclass
class LooMiscCfg(InHandIsaacLabOurDefaultCfg):
    """Revert the small things: their observation noise, and the drop test measured from the robot
    ROOT rather than the palm."""

    def __post_init__(self):
        super().__post_init__()
        ref = _reference_cfg()
        self.observations.policy.enable_corruption = True
        self.observations.critic.enable_corruption = True
        self.terminations.object_out_of_reach = ref.terminations.object_out_of_reach


@configclass
class LooToleranceCfg(InHandIsaacLabOurDefaultCfg):
    """Revert the TOLERANCE only: our keypoint success test tightened from 20 deg to their 0.1 rad
    (5.73 deg). A looser target makes accidental goals during a tumble far more likely."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.keypoint_tol_m = _keypoint_tol_for_edge(
            0.06 * CUBE_SCALE, math.degrees(0.1))
