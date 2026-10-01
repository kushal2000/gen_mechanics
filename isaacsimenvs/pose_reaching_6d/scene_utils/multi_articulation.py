"""One articulation's interface over several hands' articulations, each on its own contiguous env block.

``env.robot`` is this object in a multi-hand scene (robots/multi_hand.py). It exposes what the env reads
-- ``data.joint_pos`` / ``joint_vel`` / ``joint_pos_limits`` / ``body_state_w`` / ..., ``find_joints``,
``find_bodies``, ``set_joint_position_target``, ``write_joint_state_to_sim`` -- in the template's padded
SLOT layout, gathering from and scattering to each hand's real articulation. It is registered with the
scene in place of the hands, so the scene's ``reset(env_ids)`` / ``update`` / ``write_data_to_sim`` reach
every hand with its env ids made local to its block (the hands' own articulations cover only their block,
and handing them global ids is the indexing bug this avoids).

Slots: joint slot k is the hand's k-th joint in its spec's canonical order; a ghost slot reads 0 with
limits (0, 0), so ``joint_enabled`` masks it. Bodies are virtual: ``palm``, ``tip0..tip{F-1}``,
``link_s0..link_s{J-1}``; a ghost tip or link reads the palm's state (and is masked downstream by
``fingertip_valid`` / ``joint_geometry_valid``).
"""

from __future__ import annotations

import re

import torch

from .robots.multi_hand import PALM, slot_link, slot_tip


class _MultiData:
    """``data`` of the multi-hand articulation: every field (N, ...) in slot layout."""

    def __init__(self, owner: "MultiHandArticulation"):
        self._o = owner

    def _joint(self, name: str, fill: float = 0.0) -> torch.Tensor:
        o = self._o
        out = torch.full((o.num_envs, o.n_joints), fill, device=o.device)
        for h, art in enumerate(o.arts):
            out[o.blocks[h], : o.n_real[h]] = getattr(art.data, name)[:, o.joint_ids[h]]
        return out

    @property
    def joint_names(self) -> list[str]:
        return list(self._o.joint_names)

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
    def joint_vel_limits(self):
        return self._joint("joint_vel_limits", 1.0)

    @property
    def joint_effort_limits(self):
        return self._joint("joint_effort_limits")

    @property
    def joint_stiffness(self):
        return self._joint("joint_stiffness")

    @property
    def joint_armature(self):
        return self._joint("joint_armature")

    @property
    def joint_pos_limits(self):
        o = self._o
        out = torch.zeros((o.num_envs, o.n_joints, 2), device=o.device)   # ghost slot: (0, 0)
        for h, art in enumerate(o.arts):
            out[o.blocks[h], : o.n_real[h]] = art.data.joint_pos_limits[:, o.joint_ids[h]]
        return out

    @property
    def body_state_w(self):
        o = self._o
        out = torch.empty((o.num_envs, len(o.body_names), 13), device=o.device)
        for h, art in enumerate(o.arts):
            out[o.blocks[h]] = art.data.body_state_w[:, o.body_ids[h]]
        return out

    @property
    def root_pos_w(self):
        return self._root("root_pos_w", 3)

    @property
    def root_quat_w(self):
        return self._root("root_quat_w", 4)

    def _root(self, name, width):
        o = self._o
        out = torch.empty((o.num_envs, width), device=o.device)
        for h, art in enumerate(o.arts):
            out[o.blocks[h]] = getattr(art.data, name)
        return out

    @property
    def default_mass(self):
        """(N, 1): each env's whole-hand mass (the only use is a log line)."""
        o = self._o
        out = torch.empty((o.num_envs, 1))
        for h, art in enumerate(o.arts):
            out[o.blocks[h].cpu()] = art.data.default_mass.sum(-1, keepdim=True)
        return out


class MultiHandArticulation:
    def __init__(self, hand_set, arts, hand_of_env: torch.Tensor, device):
        self.hand_set, self.arts, self.device = hand_set, list(arts), device
        self.num_envs = int(hand_of_env.numel())
        self.hand_of_env = hand_of_env.to(device)
        self.n_joints, self.n_tips = hand_set.n_joints, hand_set.n_tips
        self.blocks = [(self.hand_of_env == h).nonzero(as_tuple=True)[0] for h in range(len(self.arts))]
        for h, b in enumerate(self.blocks):
            if b.numel() == 0 or not torch.equal(b, torch.arange(int(b[0]), int(b[0]) + b.numel(), device=device)):
                raise RuntimeError(f"hand {h}: envs are not one contiguous block")
        self.block_start = [int(b[0]) for b in self.blocks]
        self.joint_names = template_joints = list(hand_set.template.hand_joint_names)
        self.body_names = [PALM] + [slot_tip(i) for i in range(self.n_tips)] + \
                          [slot_link(k) for k in range(self.n_joints)]
        self.n_real = [s.num_hand_joints for s in hand_set.specs]
        self.joint_ids: list[list[int]] = []
        self.body_ids: list[list[int]] = []
        self.data = _MultiData(self)
        self._resolved = False
        del template_joints

    # -- indices: resolved once the articulations have their PhysX views -----------------
    def resolve(self) -> None:
        """Map each hand's real joints and bodies onto the slots. Needs the started sim."""
        self.joint_ids, self.body_ids = [], []
        for spec, art in zip(self.hand_set.specs, self.arts):
            jid, jnames = art.find_joints(list(spec.hand_joint_names), preserve_order=True)
            if list(jnames) != list(spec.hand_joint_names):
                raise RuntimeError(f"{spec.name}: joints {jnames} != spec {spec.hand_joint_names}")
            palm = art.find_bodies(spec.palm_body_name)[0][0]
            tips = art.find_bodies(list(spec.fingertip_body_names), preserve_order=True)[0]
            links = art.find_bodies(list(spec.joint_link_bodies), preserve_order=True)[0]
            if len(tips) != spec.num_fingertips or len(links) != spec.num_hand_joints:
                raise RuntimeError(f"{spec.name}: found {len(tips)} tips / {len(links)} joint links")
            ids = [palm] + list(tips) + [palm] * (self.n_tips - len(tips)) + \
                  list(links) + [palm] * (self.n_joints - len(links))
            self.joint_ids.append(list(jid))
            self.body_ids.append(ids)
            if art.num_instances != self.blocks[len(self.joint_ids) - 1].numel():
                raise RuntimeError(f"{spec.name}: articulation covers {art.num_instances} envs, "
                                   f"its block is {self.blocks[len(self.joint_ids) - 1].numel()}")
        self._resolved = True

    # -- name lookups, in slot space -------------------------------------------------------
    @staticmethod
    def _find(keys, pool, preserve_order):
        keys = [keys] if isinstance(keys, str) else list(keys)
        if preserve_order:
            ids, names = [], []
            for k in keys:
                hit = [i for i, n in enumerate(pool) if re.fullmatch(k, n)]
                ids += hit
                names += [pool[i] for i in hit]
            return ids, names
        ids = [i for i, n in enumerate(pool) if any(re.fullmatch(k, n) for k in keys)]
        return ids, [pool[i] for i in ids]

    def find_joints(self, names, preserve_order: bool = False):
        return self._find(names, self.joint_names, preserve_order)

    def find_bodies(self, names, preserve_order: bool = False):
        return self._find(names, self.body_names, preserve_order)

    # -- writes ----------------------------------------------------------------------------
    def set_joint_position_target(self, target: torch.Tensor, joint_ids=None, env_ids=None) -> None:
        if joint_ids is not None or env_ids is not None:
            raise NotImplementedError("multi-hand targets are set for every env and slot at once")
        for h, art in enumerate(self.arts):
            art.set_joint_position_target(target[self.blocks[h], : self.n_real[h]], joint_ids=self.joint_ids[h])

    def _split(self, env_ids):
        """Global env ids -> per hand (global ids in the block, local ids)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        hands = self.hand_of_env[env_ids]
        for h in range(len(self.arts)):
            sel = env_ids[hands == h]
            if sel.numel():
                yield h, sel, sel - self.block_start[h]

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None) -> None:
        """``position`` / ``velocity`` are (len(env_ids), J) in slot layout, rows in env_ids order."""
        if joint_ids is not None:
            raise NotImplementedError("multi-hand joint state is written for every slot")
        env_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else \
            torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        row = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
        row[env_ids] = torch.arange(env_ids.numel(), device=self.device)
        for h, glob, local in self._split(env_ids):
            r = row[glob]
            self.arts[h].write_joint_state_to_sim(
                position[r, : self.n_real[h]], velocity[r, : self.n_real[h]],
                joint_ids=self.joint_ids[h], env_ids=local)

    # -- the scene's hooks -----------------------------------------------------------------
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
        """The first hand's view -- only ever asked for a shape count (``_verify_articulation_view``)."""
        return self.arts[0].root_physx_view


__all__ = ["MultiHandArticulation"]
