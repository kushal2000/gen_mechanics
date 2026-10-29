"""Several hands in one olympics scene, each on its dummy wrist over its own envs (round-robin), presented to the env
as one articulation: the hand sets of common/robots/multi_hand.py, presented as pose_reaching_6d's multi-hand scene
presents them (scene_utils/multi_articulation.py), with the 6 wrist joints ahead of the padded finger slots.

Selected by ``hand.robot_spec = "multi:<spec>+<spec>+..."``. Joint order (Isaac Lab order = canonical order here):
the wrist joints, then finger slots s0..s{J-1}; which slot a hand's joint takes is the HandSet's ``joint_slots`` (its
k-th joint for a dense ``multi:`` ref, its finger-major slot for ``multi:finger:``); a ghost slot reads 0 with limits
(0, 0), so ``joint_enabled`` masks it. Bodies are virtual -- ``palm``, ``tip0..``,
``link_s0..`` -- a ghost tip or link reading the palm's state (masked downstream by ``tip_valid`` /
``joint_geometry_valid``). Nothing is padded in the simulator: each hand keeps its own links and joints.
"""
from __future__ import annotations

import re

import torch

from isaacsimenvs.common.robots.multi_hand import PALM, slot_link, slot_tip
from .wrist import WRIST_JOINTS


class _Data:
    """``data``: every field (N, ...) in slot layout, gathered from the hands' articulations."""

    def __init__(self, owner: "OlympicsMultiHand"):
        self._o = owner

    def _joint(self, name: str, fill: float = 0.0, tail=()) -> torch.Tensor:
        o = self._o
        out = torch.full((o.num_envs, o.num_joints, *tail), fill, device=o.device)
        for h, art in enumerate(o.arts):
            out[o.blocks[h][:, None], o.slot_cols[h]] = getattr(art.data, name)[:, o.joint_ids[h]]
        return out

    @property
    def joint_pos(self):
        return self._joint("joint_pos")

    @property
    def joint_vel(self):
        return self._joint("joint_vel")

    @property
    def default_joint_pos(self):
        return self._joint("default_joint_pos")

    @property
    def soft_joint_pos_limits(self):
        return self._joint("soft_joint_pos_limits", tail=(2,))       # a ghost slot: (0, 0)

    @property
    def joint_pos_limits(self):
        return self._joint("joint_pos_limits", tail=(2,))

    @property
    def body_state_w(self):
        o = self._o
        out = torch.empty((o.num_envs, len(o.body_names), 13), device=o.device)
        for h, art in enumerate(o.arts):
            out[o.blocks[h]] = art.data.body_state_w[:, o.body_ids[h]]
        return out

    @property
    def body_pos_w(self):
        return self.body_state_w[..., 0:3]

    @property
    def body_quat_w(self):
        return self.body_state_w[..., 3:7]

    def _root(self, name, width):
        o = self._o
        out = torch.empty((o.num_envs, width), device=o.device)
        for h, art in enumerate(o.arts):
            out[o.blocks[h]] = getattr(art.data, name)
        return out

    @property
    def root_pos_w(self):
        return self._root("root_pos_w", 3)

    @property
    def root_quat_w(self):
        return self._root("root_quat_w", 4)

    @property
    def default_root_state(self):
        return self._root("default_root_state", 13)


class OlympicsMultiHand:
    def __init__(self, hand_set, arts, hand_of_env: torch.Tensor, device):
        self.hand_set, self.arts, self.device = hand_set, list(arts), device
        self.num_envs = int(hand_of_env.numel())
        self.hand_of_env = hand_of_env.to(device)
        self.blocks = [(self.hand_of_env == h).nonzero(as_tuple=True)[0] for h in range(len(self.arts))]
        if any(b.numel() == 0 for b in self.blocks):
            raise RuntimeError("a hand has no envs")
        self.local_of_env = torch.empty(self.num_envs, dtype=torch.long, device=device)
        for b in self.blocks:
            self.local_of_env[b] = torch.arange(b.numel(), device=device)
        n_w = len(WRIST_JOINTS)
        self.joint_names = list(WRIST_JOINTS) + list(hand_set.template.hand_joint_names)
        self.num_joints = len(self.joint_names)
        self.body_names = [PALM] + [slot_tip(i) for i in range(hand_set.n_tips)] + \
                          [slot_link(k) for k in range(hand_set.n_joints)]
        self.n_real = [n_w + s.num_hand_joints for s in hand_set.specs]
        # Each hand's columns in slot space: the wrist, then its joints' slots (HandSet.joint_slots).
        self.slot_cols = [torch.tensor(list(range(n_w)) + [n_w + s for s in hand_set.joint_slots[h]],
                                       dtype=torch.long, device=device) for h in range(hand_set.n_hands)]
        self.joint_ids: list[list[int]] = []
        self.body_ids: list[list[int]] = []
        self.data = _Data(self)
        self._ALL_INDICES = torch.arange(self.num_envs, device=device)

    def resolve(self) -> None:
        """Map each hand's real joints and bodies onto the slots. Needs the started sim."""
        self.joint_ids, self.body_ids = [], []
        F, J = self.hand_set.n_tips, self.hand_set.n_joints
        for h, (spec, art) in enumerate(zip(self.hand_set.specs, self.arts)):
            names = list(WRIST_JOINTS) + list(spec.hand_joint_names)
            jid, jnames = art.find_joints(names, preserve_order=True)
            if list(jnames) != names:
                raise RuntimeError(f"{spec.name}: joints {jnames} != {names}")
            palm = art.find_bodies(spec.palm_body_name)[0][0]
            tips = art.find_bodies(list(spec.fingertip_body_names), preserve_order=True)[0]
            links = art.find_bodies(list(spec.joint_link_bodies), preserve_order=True)[0]
            if len(tips) != spec.num_fingertips or len(links) != spec.num_hand_joints:
                raise RuntimeError(f"{spec.name}: found {len(tips)} tips / {len(links)} joint links")
            self.joint_ids.append(list(jid))
            link_of_slot = [palm] * J
            for j, s in enumerate(self.hand_set.joint_slots[h]):
                link_of_slot[s] = links[j]
            self.body_ids.append([palm] + list(tips) + [palm] * (F - len(tips)) + link_of_slot)
            block = self.blocks[h]
            if art.num_instances != block.numel():
                raise RuntimeError(f"{spec.name}: articulation covers {art.num_instances} envs, block {block.numel()}")
            # Row r of the view must be the r-th env of the block, or every gather lands on another env.
            env_of_row = [int(re.search(r"/env_(\d+)/", p).group(1)) for p in art.root_physx_view.prim_paths]
            if env_of_row != block.tolist():
                raise RuntimeError(f"{spec.name}: view rows are envs {env_of_row[:8]}..., expected {block.tolist()[:8]}...")

    # -- lookups, in slot space ------------------------------------------------------------------------
    def find_bodies(self, names, preserve_order: bool = False):
        keys = [names] if isinstance(names, str) else list(names)
        if preserve_order:
            ids = [i for k in keys for i, n in enumerate(self.body_names) if re.fullmatch(k, n)]
        else:
            ids = [i for i, n in enumerate(self.body_names) if any(re.fullmatch(k, n) for k in keys)]
        return ids, [self.body_names[i] for i in ids]

    def real_joint_names(self, env_id: int) -> list[str]:
        """Env ``env_id``'s own joint names, wrist first; their slot columns are ``real_slots(env_id)``."""
        return list(WRIST_JOINTS) + list(self.hand_set.specs[int(self.hand_of_env[env_id])].hand_joint_names)

    def real_slots(self, env_id: int) -> list[int]:
        """The slot columns holding env ``env_id``'s own joints, in ``real_joint_names`` order."""
        return self.slot_cols[int(self.hand_of_env[env_id])].tolist()

    def _split(self, env_ids):
        env_ids = self._ALL_INDICES if env_ids is None else torch.as_tensor(env_ids, device=self.device,
                                                                            dtype=torch.long)
        hands = self.hand_of_env[env_ids]
        for h in range(len(self.arts)):
            pick = (hands == h).nonzero(as_tuple=True)[0]
            if pick.numel():
                yield h, pick, self.local_of_env[env_ids[pick]]

    # -- writes ------------------------------------------------------------------------------------------
    def set_joint_position_target(self, target: torch.Tensor) -> None:
        for h, art in enumerate(self.arts):
            art.set_joint_position_target(target[self.blocks[h][:, None], self.slot_cols[h]],
                                          joint_ids=self.joint_ids[h])

    def write_joint_state_to_sim(self, position, velocity, env_ids=None) -> None:
        """``position`` / ``velocity``: (len(env_ids), J) slot layout, rows in env_ids order."""
        for h, pick, local in self._split(env_ids):
            cols = self.slot_cols[h]
            self.arts[h].write_joint_state_to_sim(position[pick[:, None], cols], velocity[pick[:, None], cols],
                                                  joint_ids=self.joint_ids[h], env_ids=local)

    def write_root_pose_to_sim(self, pose, env_ids=None) -> None:
        for h, pick, local in self._split(env_ids):
            self.arts[h].write_root_pose_to_sim(pose[pick], env_ids=local)

    # -- the scene's hooks -------------------------------------------------------------------------------
    def reset(self, env_ids=None) -> None:
        for h, _, local in self._split(env_ids):
            self.arts[h].reset(local)

    def update(self, dt: float) -> None:
        for art in self.arts:
            art.update(dt)

    def write_data_to_sim(self) -> None:
        for art in self.arts:
            art.write_data_to_sim()

    @property
    def num_instances(self) -> int:
        return self.num_envs

    @property
    def root_physx_view(self):
        return self.arts[0].root_physx_view


__all__ = ["OlympicsMultiHand"]
