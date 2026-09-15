# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Follower cost: pulls frame_b toward the pose implied by frame_a and a fixed relative transform.

    T_b_desired = T_a(q) . T_AB_rigid

frame_a (leader) is detached from the computation graph so only frame_b's joints receive the
gradient -- the leader is free to pursue its own trajectory goal while the follower is attracted
to track it rigidly.

It reads both frames straight from ``state.tool_poses`` (the same state the tool-pose cost uses)
so it needs no goal buffer and no kinematics change -- both grippers are already tool frames, so
their live poses are computed by the rollout every step. The target is a stored constant, updated
in place so enabling/retargeting it stays CUDA-graph safe (weight and target tensors keep their
identity; only their values change).

Quaternion convention is cuRobo's wxyz. The rotation error uses the vector part of the relative
error quaternion (``sin`` of the half-angle), which is smooth and differentiable everywhere
(no ``acos`` gradient blow-up at perfect alignment) and correct under quaternion double-cover.
"""
from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Optional

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a wxyz quaternion, shape [..., 4]."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two wxyz quaternions, shape [..., 4]. Functional (autograd-safe)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v ([..., 3]) by wxyz quaternion q ([..., 4]) via the fast formula."""
    qv = q[..., 1:]
    w = q[..., 0:1]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w * t + torch.cross(qv, t, dim=-1)


class RelativePoseCost(BaseCost):
    """Cost keeping frame_b's pose relative to frame_a at a fixed target transform."""

    def __init__(self, config: "RelativePoseCostCfg"):
        self.config: "RelativePoseCostCfg" = config
        super().__init__(config)  # sets self.device_cfg, self._weight, enabled/disabled from weight

        self.frame_a = config.frame_a
        self.frame_b = config.frame_b
        self._pos_w = float(config.position_weight)
        self._rot_w = float(config.rotation_weight)

        # Resolved lazily on the first forward() from current_tool_poses.tool_frames.
        self._idx_a: Optional[int] = None
        self._idx_b: Optional[int] = None

        # Persistent target buffers -- updated in place (copy_) so an already-captured CUDA graph
        # keeps referencing the same tensors. Default: identity relative pose (harmless while the
        # cost is disabled, i.e. weight == 0).
        dev, dt = self.device_cfg.device, self.device_cfg.dtype
        tp = config.target_rel_position
        tq = config.target_rel_quaternion
        self._target_pos = torch.tensor(
            list(tp) if tp is not None else [0.0, 0.0, 0.0], device=dev, dtype=dt
        )
        self._target_quat = torch.tensor(
            list(tq) if tq is not None else [1.0, 0.0, 0.0, 0.0], device=dev, dtype=dt
        )

    def update_target(self, rel_position, rel_quaternion) -> None:
        """Set T_AB_rigid live (per-solve). In-place, so it is safe to call after CUDA-graph
        capture.

        Args:
            rel_position: (3,) target translation of relpose(a, b) = R_a^T (t_b - t_a).
            rel_quaternion: (4,) wxyz target rotation of relpose(a, b) = R_a^T R_b.
        """
        self._target_pos.copy_(
            torch.as_tensor(rel_position, device=self._target_pos.device, dtype=self._target_pos.dtype)
        )
        self._target_quat.copy_(
            torch.as_tensor(rel_quaternion, device=self._target_quat.device, dtype=self._target_quat.dtype)
        )

    def _resolve_frame_indices(self, current_tool_poses) -> None:
        """Resolve (and cache) the tool-frame indices of frame_a / frame_b."""
        if self._idx_a is not None:
            return
        tf = current_tool_poses.tool_frames
        if self.frame_a not in tf or self.frame_b not in tf:
            log_and_raise(
                f"RelativePoseCost: frames ({self.frame_a}, {self.frame_b}) not both in "
                f"tool_frames {tf}"
            )
        self._idx_a = tf.index(self.frame_a)
        self._idx_b = tf.index(self.frame_b)

    def _errors(self, current_tool_poses, detach_leader: bool):
        """Squared position / rotation error of frame_b against T_a . T_AB_target.

        Args:
            current_tool_poses: ToolPose with position [B,H,L,3], quaternion [B,H,L,4].
            detach_leader: When True, frame_a is detached so only frame_b's joints
                receive gradient (leader/follower asymmetry used by the cost path).

        Returns:
            ``(pos_err_sq, rot_err_sq)``, each (B, H). ``rot_err_sq`` is
            ``sin^2(theta/2)`` of the orientation error.
        """
        self._resolve_frame_indices(current_tool_poses)

        pos = current_tool_poses.position  # [B, H, L, 3]
        quat = current_tool_poses.quaternion  # [B, H, L, 4]
        p_b = pos[:, :, self._idx_b, :]
        q_b = quat[:, :, self._idx_b, :]

        p_a = pos[:, :, self._idx_a, :]
        q_a = quat[:, :, self._idx_a, :]
        if detach_leader:
            p_a = p_a.detach()
            q_a = q_a.detach()

        # Desired pose of B in world frame: T_b_desired = T_a . T_AB_target
        # Expand scalar target buffers to [B, H, 3/4] so torch.cross/unbind broadcast correctly.
        target_pos = self._target_pos.expand(p_a.shape)
        target_quat = self._target_quat.expand(q_a.shape)
        p_b_desired = p_a + _quat_rotate(q_a, target_pos)
        q_b_desired = _quat_multiply(q_a, target_quat)

        pos_err_sq = ((p_b - p_b_desired) ** 2).sum(dim=-1)  # (B, H)

        # Rotation error via the vector part of q_b^{-1} . q_b_desired:
        # smooth everywhere, 0 at alignment, and identical for q and -q (double-cover safe).
        q_err = _quat_multiply(_quat_conjugate(q_b), q_b_desired)  # [B, H, 4]
        rot_err_sq = (q_err[..., 1:] ** 2).sum(dim=-1)  # (B, H)
        return pos_err_sq, rot_err_sq

    def is_active(self) -> torch.Tensor:
        """0/1 scalar tensor: is the coupling currently weighted in?

        Tensor-valued on purpose -- callers gate on it by multiplication rather than by
        Python control flow, so the decision stays valid inside a captured CUDA graph.
        """
        return (self._weight.abs().sum() > 0).to(self._weight.dtype)

    def forward(self, current_tool_poses, **kwargs) -> torch.Tensor:
        """Compute the loop-closure cost from a ToolPose (position [B,H,L,3], quaternion [B,H,L,4]).

        Returns:
            cost: (B, H, 1) tensor. Identically zero while the weight is zero, so this is
            safe (and cheap enough) to evaluate unconditionally every step.
        """
        pos_err_sq, rot_err_sq = self._errors(current_tool_poses, detach_leader=True)
        cost = self._weight * (self._pos_w * pos_err_sq + self._rot_w * rot_err_sq)
        return cost.unsqueeze(-1)  # (B, H, 1)

    @torch.no_grad()
    def compute_drift(self, current_tool_poses):
        """Loop-closure drift in physical units, for convergence/validation checks.

        Unlike :meth:`forward` this is weight-independent in magnitude (it is an error, not
        a cost) but is masked to zero while the coupling is inactive, so a plan made without
        coupling is not judged against a drift tolerance.

        The values are a **running max over the horizon**: element ``t`` holds the worst
        drift seen on steps ``0..t``. Solvers check convergence at a single timestep, so
        this makes the terminal entry report the worst drift of the whole trajectory rather
        than only the final node -- which is what matters for a rigid co-grasp.

        Returns:
            ``(position_drift_m, rotation_drift_rad)``, each (B, H, 1).
        """
        pos_err_sq, rot_err_sq = self._errors(current_tool_poses, detach_leader=False)
        active = self.is_active()
        position_drift = torch.sqrt(torch.clamp(pos_err_sq, min=0.0)) * active
        # |vec(q_err)| = sin(theta/2) -> geodesic angle = 2 * asin(|vec|), in [0, pi].
        sin_half = torch.sqrt(torch.clamp(rot_err_sq, min=0.0, max=1.0))
        rotation_drift = 2.0 * torch.asin(torch.clamp(sin_half, max=1.0)) * active
        position_drift = torch.cummax(position_drift, dim=1)[0]
        rotation_drift = torch.cummax(rotation_drift, dim=1)[0]
        return position_drift.unsqueeze(-1), rotation_drift.unsqueeze(-1)
