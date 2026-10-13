"""Pen keypoints and goal-pose helpers."""
from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """(N,) yaw about world z -> (N, 4) wxyz: the pen flat, its tip (local +x) pointing at that yaw."""
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0], q[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return q


def pen_keypoint_offsets(length: float, device) -> torch.Tensor:
    """(4, 3): four evenly spaced points along the pen's long axis, tip (+x end) to tail, in the pen's frame. Four,
    so the keypoint observations keep in-hand reorientation's width (4 x 3); distinct ends, so a pen turned end for
    end is a pen length away from its goal at the tips; roll about the long axis moves none of them."""
    h = length / 2
    return torch.tensor([[h, 0.0, 0.0], [h / 3, 0.0, 0.0], [-h / 3, 0.0, 0.0], [-h, 0.0, 0.0]], device=device)


def keypoints(pos: torch.Tensor, quat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """(N, 3) positions and (N, 4) wxyz -> (N, K, 3) keypoints."""
    n, k = pos.shape[0], offsets.shape[0]
    q = quat.unsqueeze(1).expand(n, k, 4).reshape(-1, 4)
    return pos.unsqueeze(1) + quat_apply(q, offsets.unsqueeze(0).expand(n, k, 3).reshape(-1, 3)).view(n, k, 3)
