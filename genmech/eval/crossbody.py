"""Drive one robot with a policy trained on a different one.

The immediate use is a sharp test of the generated hand: ``gen_sharpa_like`` is
supposed to reproduce SHARPA's kinematics, and it does to 0.002 mm on every
fingertip. If that reproduction is faithful in the way that *matters*, then
SHARPA's own pretrained policy should be able to drive it — zero-shot, no
fine-tuning. Geometry checks say the shapes agree; this says whether the
agreement is good enough to be functionally the same robot.

It is a strictly harder test than any measurement so far, and a cheap one: no
training, one rollout.

**Why the dimensions do not match.** A generated hand always emits the full
template — 7 arm + 30 hand joints — whatever it looks like, because PhysX reads
one ``dof_count`` per articulation view. SHARPA has 7 + 22. So a SHARPA policy
expects 29 actions and an observation built from 29 joints, while the generated
env offers 37.

**Why the mapping is exact anyway.** Only three observation fields scale with
joint count (``joint_pos``, ``joint_vel``, ``prev_action_targets``). Both hands
have five fingertips, so every other field — including all the object and
keypoint state — is identical in width and meaning. The 22 active slots of the
generated hand correspond one-to-one with SHARPA's 22 hand joints by
construction, so the adapter is a gather on those three fields and a scatter on
the action. Ghosted joints receive zero, which is also where their limits hold
them.

Nothing here is approximate: it is a permutation, not a projection.
"""

from __future__ import annotations

import torch

from genmech.robots.generated import params as P


# SHARPA joint name -> template slot. The bare pinky "CMC" is a flexion joint
# despite the name, and the thumb's single interphalangeal "IP" occupies the DIP
# slot -- ghosting PIP instead of DIP is what puts the thumb's distal segment in
# the right mass tier (see params.SHARPA_LIKE).
_SLOT_ALIASES = {
    "CMC_FE": "CMC_FE",
    "CMC_AA": "CMC_AA",
    "MCP_FE": "MCP_FE",
    "MCP_AA": "MCP_AA",
    "PIP": "PIP",
    "DIP": "DIP",
    "IP": "DIP",
    "CMC": "CMC_FE",
}

# Slot order of SHARPA_LIKE's finger list.
_FINGER_SLOT = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}


def _slot_of(sharpa_joint: str) -> tuple[int, str]:
    finger = next((f for f in _FINGER_SLOT if f in sharpa_joint), None)
    if finger is None:
        raise KeyError(f"cannot place {sharpa_joint!r}: no finger in the name")
    # Longest alias first, so "MCP_FE" is not matched by "FE" and "CMC" does not
    # shadow "CMC_FE".
    for alias in sorted(_SLOT_ALIASES, key=len, reverse=True):
        if sharpa_joint.endswith(alias):
            return _FINGER_SLOT[finger], _SLOT_ALIASES[alias]
    raise KeyError(f"cannot place {sharpa_joint!r}: no slot suffix matched")


def joint_index_map(policy_spec, env_spec) -> torch.Tensor:
    """For each of the policy robot's joints, its index in the env robot.

    Returns a LongTensor of length ``policy_spec.num_joints``. Gathering the
    env's joint vector with it yields the policy's joint vector; scattering the
    policy's action with it yields the env's action.
    """
    env_names = list(env_spec.joint_names_canonical)
    index_of = {n: i for i, n in enumerate(env_names)}

    out: list[int] = []
    for name in policy_spec.joint_names_canonical:
        if name in index_of:                     # the arm, named identically
            out.append(index_of[name])
            continue
        slot_idx, slot = _slot_of(name)
        target = f"gen_f{slot_idx}_{slot}"
        if target not in index_of:
            raise KeyError(f"{name!r} maps to {target!r}, absent from {env_spec.name}")
        out.append(index_of[target])

    if len(set(out)) != len(out):
        raise ValueError("joint map is not injective; two policy joints share a slot")
    return torch.tensor(out, dtype=torch.long)


def obs_index_map(policy_spec, env_spec, field_list) -> torch.Tensor:
    """Indices into the env's observation that rebuild the policy's observation.

    Walks the field list in order, gathering the mapped joints for the three
    joint-width fields and passing everything else through. Fails loudly if a
    field's width differs for any other reason -- a silent mismatch here would
    feed the policy scrambled observations that still have the right shape.
    """
    from genmech.tasks.pose_reach.utils.obs_utils import obs_field_sizes

    jmap = joint_index_map(policy_spec, env_spec)
    env_sizes = obs_field_sizes(env_spec)
    pol_sizes = obs_field_sizes(policy_spec)

    idx: list[int] = []
    cursor = 0
    for field in field_list:
        env_w, pol_w = env_sizes[field], pol_sizes[field]
        if env_w == pol_w:
            idx.extend(range(cursor, cursor + env_w))
        elif pol_w == policy_spec.num_joints and env_w == env_spec.num_joints:
            idx.extend((cursor + jmap).tolist())
        else:
            raise ValueError(
                f"field {field!r} is {env_w} wide in {env_spec.name} but {pol_w} "
                f"in {policy_spec.name}, and is not a joint vector"
            )
        cursor += env_w

    if len(idx) != sum(pol_sizes[f] for f in field_list):
        raise ValueError("obs map length does not match the policy's obs width")
    return torch.tensor(idx, dtype=torch.long)


class CrossBodyAdapter:
    """Gathers observations down to the policy's robot, scatters actions back up."""

    def __init__(self, policy_spec, env_spec, field_list, device):
        self.policy_spec = policy_spec
        self.env_spec = env_spec
        self.obs_idx = obs_index_map(policy_spec, env_spec, field_list).to(device)
        self.act_idx = joint_index_map(policy_spec, env_spec).to(device)
        self.n_env_actions = env_spec.num_joints
        self.device = device

    def observation(self, env_obs: torch.Tensor) -> torch.Tensor:
        return env_obs.index_select(-1, self.obs_idx)

    def action(self, policy_action: torch.Tensor) -> torch.Tensor:
        """Ghosted joints get zero, which is where their limits hold them."""
        out = torch.zeros(*policy_action.shape[:-1], self.n_env_actions,
                          dtype=policy_action.dtype, device=policy_action.device)
        out[..., self.act_idx] = policy_action
        return out

    def describe(self) -> str:
        ghost = self.n_env_actions - len(self.act_idx)
        return (f"cross-body: {self.policy_spec.name} policy "
                f"({self.policy_spec.num_joints} act) -> {self.env_spec.name} "
                f"({self.n_env_actions} act); {ghost} joint(s) driven to zero, "
                f"obs {len(self.obs_idx)} gathered")


__all__ = ["CrossBodyAdapter", "joint_index_map", "obs_index_map"]
