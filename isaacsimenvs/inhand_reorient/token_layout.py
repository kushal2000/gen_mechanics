"""Per-joint token observation for ``task_profile: hora`` (``hora.token_obs``).

The team's joint-token method (``coevolution/networks/joint_transformer.py``,
``pose_reaching_6d/obs_utils/layout.py``) on the in-hand envelope: one token
per articulation column of the 36-slot envelope (no arm), one global token.
Kit-free: the env computes the per-joint arrays and calls ``assemble``; the
network (``token_policy.py``) reads the same layout.

Token features, in order (``TOKEN_FIELDS``):

- ``q_hist`` (3): HORA's joint position, unscaled to [-1, 1] by the joint's
  limits, at t, t-1, t-2 (with HORA's observation noise).
- ``target_hist`` (3): the PD target at t, t-1, t-2 (rad).
- ``link_box`` (12): the slot's child link as 4 ordered box points
  (``grammar_envelope.token_boxes``), in the palm frame at the current q (m).
- ``limits`` (2): lower and upper joint limit (rad).
- ``enabled`` (1): 1 for a joint the policy controls (a real finger joint or
  a leader palm joint), 0 for a ghost slot and for a locked or follower
  carrier (a follower is tied to its leader). Read RAW by the network as the
  attention mask; never normalised.
- ``object_kp_rel`` (12): 4 object keypoints (origin and +x, +y, +z at the
  keypoint distance) relative to the link's origin, in the palm frame (m).

A ghost token is all zeros (``enabled`` 0). Global token (``GLOBAL_FIELDS``):
HORA's 9 privileged values (object position in the palm frame, scale, mass,
friction, centre of mass) and 3 hand scalars (digit count, hand scale,
capsule radius). An optional design one-hot follows (``hora.design_id_obs``);
the network uses it only to pick per-design normalisation statistics, never
as an input feature.
"""

from __future__ import annotations

import torch

TOKEN_FIELDS: tuple[tuple[str, int], ...] = (
    ("q_hist", 3),
    ("target_hist", 3),
    ("link_box", 12),
    ("limits", 2),
    ("enabled", 1),
    ("object_kp_rel", 12),
)
TOKEN_DIM: int = sum(w for _, w in TOKEN_FIELDS)  # 33
ENABLED_COL: int = sum(w for name, w in TOKEN_FIELDS[: [n for n, _ in TOKEN_FIELDS].index("enabled")])  # 20
GLOBAL_FIELDS: tuple[tuple[str, int], ...] = (("hora_priv", 9), ("hand_scalars", 3))
GLOBAL_DIM: int = sum(w for _, w in GLOBAL_FIELDS)  # 12
NUM_KEYPOINTS: int = 4


def slot_body_name(slot: int) -> str:
    """The child body of envelope slot ``slot`` (``author_grammar`` naming)."""
    from .scene import grammar_envelope as ge

    return ge.slot_body(slot)


def layout(n_tokens: int, design_id_width: int = 0) -> dict:
    """Column map of the flat observation: tokens token-major, then the
    global block, then the optional design one-hot."""
    tok_end = n_tokens * TOKEN_DIM
    glob_end = tok_end + GLOBAL_DIM
    return {
        "n_tokens": n_tokens,
        "token_dim": TOKEN_DIM,
        "enabled_col": ENABLED_COL,
        "token_columns": [list(range(i * TOKEN_DIM, (i + 1) * TOKEN_DIM)) for i in range(n_tokens)],
        "global_slice": (tok_end, glob_end),
        "design_slice": (glob_end, glob_end + int(design_id_width)),
        "obs_dim": glob_end + int(design_id_width),
    }


def assemble(q_hist: torch.Tensor, target_hist: torch.Tensor, link_box: torch.Tensor, limits: torch.Tensor,
             enabled: torch.Tensor, object_kp_rel: torch.Tensor) -> torch.Tensor:
    """``(n, J, 3) x2, (n, J, 12), (n, J, 2), (n, J) bool, (n, J, 12)`` ->
    ``(n, J * TOKEN_DIM)``, ghost tokens zeroed."""
    en = enabled.to(q_hist.dtype).unsqueeze(-1)
    feats = torch.cat([q_hist, target_hist, link_box, limits, en, object_kp_rel], dim=-1)
    if feats.shape[-1] != TOKEN_DIM:
        raise ValueError(f"token width {feats.shape[-1]} != {TOKEN_DIM}")
    feats = feats * en
    return feats.reshape(feats.shape[0], -1)


__all__ = ["ENABLED_COL", "GLOBAL_DIM", "GLOBAL_FIELDS", "NUM_KEYPOINTS", "TOKEN_DIM", "TOKEN_FIELDS",
           "assemble", "layout", "slot_body_name"]
