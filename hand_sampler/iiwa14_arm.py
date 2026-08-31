"""The KUKA iiwa14 arm, shared verbatim by every RobotSpec.

The study's central claim is that observed differences are attributable to the
*hand*. That only holds if the arm is identical — same joint limits, same PD
gains, same home pose, same reach. Keeping these tables in one module makes that
structural: a spec imports them rather than restating them, so it cannot quietly
drift, and ``test_robot_spec_invariants.py`` asserts every registered spec's arm
fields are equal.

Values are the ones the simtoolreal SHARPA policy was trained and validated
against (verified against the live articulation by tests/test_env_smoke.py).

The arm links are also byte-identical at the URDF level: the Allegro robot is
built by splicing the Allegro hand onto the *same* iiwa14 chain the SHARPA URDF
uses, rather than using the stock kuka_allegro.urdf, which ships on an iiwa7
with different link lengths and joint limits. See docs/methodology.md §1.
"""

from __future__ import annotations

import math


ARM_NAME = "iiwa14"

ARM_JOINT_NAMES: tuple[str, ...] = (
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
)

ARM_STIFFNESS: dict[str, float] = {
    "iiwa14_joint_1": 600.0,
    "iiwa14_joint_2": 600.0,
    "iiwa14_joint_3": 500.0,
    "iiwa14_joint_4": 400.0,
    "iiwa14_joint_5": 200.0,
    "iiwa14_joint_6": 200.0,
    "iiwa14_joint_7": 200.0,
}

ARM_DAMPING: dict[str, float] = {
    "iiwa14_joint_1": 27.027026473513512,
    "iiwa14_joint_2": 27.027026473513512,
    "iiwa14_joint_3": 24.672186769721083,
    "iiwa14_joint_4": 22.067474708266914,
    "iiwa14_joint_5": 9.752538131173853,
    "iiwa14_joint_6": 9.147747263670984,
    "iiwa14_joint_7": 9.147747263670984,
}

ARM_DEFAULT_JOINT_POS: dict[str, float] = {
    "iiwa14_joint_1": -1.571,
    "iiwa14_joint_2": 1.571,
    "iiwa14_joint_3": 0.0,
    "iiwa14_joint_4": 1.376,
    "iiwa14_joint_5": 0.0,
    "iiwa14_joint_6": 1.485,
    "iiwa14_joint_7": 1.308,
}

# Applied on top of the home pose when reset.start_arm_higher is set, which the
# DexToolBench evaluation uses to clear the table.
START_ARM_HIGHER_DELTAS: dict[str, float] = {
    "iiwa14_joint_2": -math.radians(10.0),
    "iiwa14_joint_4": +math.radians(10.0),
}

# Base placement on the table. Held constant with the arm: moving it would shift
# the reachable workspace relative to the goal volume.
BASE_POS: tuple[float, float, float] = (0.0, 0.8, 0.0)
BASE_ROT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

# Serial chain, so only consecutive links need filtering. Hand specs extend this
# with their own palm-to-finger pairs; the last arm link (whichever the palm
# merges into) also gains the finger-base pairs there.
ARM_ADJACENT_LINKS: dict[str, list[str]] = {
    "iiwa14_link_0": ["iiwa14_link_1"],
    "iiwa14_link_1": ["iiwa14_link_0", "iiwa14_link_2"],
    "iiwa14_link_2": ["iiwa14_link_1", "iiwa14_link_3"],
    "iiwa14_link_3": ["iiwa14_link_2", "iiwa14_link_4"],
    "iiwa14_link_4": ["iiwa14_link_3", "iiwa14_link_5"],
    "iiwa14_link_5": ["iiwa14_link_4", "iiwa14_link_6"],
    "iiwa14_link_6": ["iiwa14_link_5", "iiwa14_link_7"],
}

# The link the hand's palm merges into under merge_fixed_joints.
ARM_TIP_LINK = "iiwa14_link_7"

ARM_LINK_PRIM_REGEX = "/World/envs/env_.*/Robot/iiwa14_link_.*/visuals"


assert len(ARM_JOINT_NAMES) == 7
assert set(ARM_STIFFNESS) == set(ARM_JOINT_NAMES)
assert set(ARM_DAMPING) == set(ARM_JOINT_NAMES)
assert set(ARM_DEFAULT_JOINT_POS) == set(ARM_JOINT_NAMES)


__all__ = [
    "ARM_NAME",
    "ARM_JOINT_NAMES",
    "ARM_STIFFNESS",
    "ARM_DAMPING",
    "ARM_DEFAULT_JOINT_POS",
    "START_ARM_HIGHER_DELTAS",
    "BASE_POS",
    "BASE_ROT",
    "ARM_ADJACENT_LINKS",
    "ARM_TIP_LINK",
    "ARM_LINK_PRIM_REGEX",
]
