# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RelativePoseCost (follower cost: B tracks A * T_AB_target).

Runs on CPU (no GPU needed): the cost's forward is pure torch. Verifies the follower math
(including a non-identity frame_a, so the R_a rotation is actually exercised), the enable/
disable weight toggle, and the in-place retarget.
"""
import torch

from curobo._src.cost.cost_relative_pose import (
    RelativePoseCost,
    _quat_conjugate,
    _quat_multiply,
    _quat_rotate,
)
from curobo._src.cost.cost_relative_pose_cfg import RelativePoseCostCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import ToolPose

CPU = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)


def _norm(q):
    return q / torch.linalg.norm(q)


def _make_tool_pose(p_a, q_a, p_b, q_b):
    """Build a [B=1, H=1, L=2, .] ToolPose for frames A, B."""
    position = torch.stack((p_a, p_b), dim=0).view(1, 1, 2, 3)
    quaternion = torch.stack((q_a, q_b), dim=0).view(1, 1, 2, 4)
    return ToolPose(tool_frames=["A", "B"], position=position, quaternion=quaternion)


def _build_cost(target_pos, target_quat, weight=1.0):
    cfg = RelativePoseCostCfg(
        weight=[weight], device_cfg=CPU, frame_a="A", frame_b="B",
        target_rel_position=target_pos.tolist(), target_rel_quaternion=target_quat.tolist(),
    )
    return RelativePoseCost(cfg)


def test_zero_cost_when_relative_pose_matches_target():
    # Arbitrary (non-identity) frame_a, so R_a^T actually matters.
    p_a = torch.tensor([0.3, -0.2, 0.7])
    q_a = _norm(torch.tensor([0.5, 0.5, 0.5, 0.5]))          # 120 deg about the diagonal
    # A known target relative transform.
    t_rel = torch.tensor([0.0, 0.15, 0.0])                    # 15 cm apart along a-frame +y
    q_rel = _norm(torch.tensor([0.9, 0.1, 0.2, 0.3]))
    # Compose frame_b = frame_a * relpose, so relpose(a, b) == (t_rel, q_rel) exactly.
    q_b = _quat_multiply(q_a, q_rel)
    p_b = p_a + _quat_rotate(q_a, t_rel)

    cost = _build_cost(t_rel, q_rel)
    tp = _make_tool_pose(p_a, q_a, p_b, q_b)
    value = cost.forward(tp)
    assert value.shape == (1, 1, 1)
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-6), value


def test_cost_grows_with_translation_error():
    p_a = torch.tensor([0.0, 0.0, 0.0])
    q_a = _norm(torch.tensor([1.0, 0.0, 0.0, 0.0]))           # identity: relpose == frame_b pose
    t_rel = torch.tensor([0.1, 0.0, 0.0])
    q_rel = _norm(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    cost = _build_cost(t_rel, q_rel, weight=1.0)

    # frame_b displaced 5 cm beyond the target along x -> pos error 0.05 m.
    p_b = t_rel + torch.tensor([0.05, 0.0, 0.0])
    tp = _make_tool_pose(p_a, q_a, p_b, q_rel)
    value = cost.forward(tp)
    # cost = weight * position_weight * ||err||^2 = 1 * 1 * 0.05^2
    assert torch.allclose(value.view(()), torch.tensor(0.05 ** 2), atol=1e-6), value


def test_disable_zeroes_cost_and_enable_restores():
    t_rel = torch.tensor([0.1, 0.0, 0.0])
    q_id = _norm(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    cost = _build_cost(t_rel, q_id, weight=1.0)
    p_a = torch.zeros(3)
    p_b = t_rel + torch.tensor([0.05, 0.0, 0.0])
    tp = _make_tool_pose(p_a, q_id, p_b, q_id)

    assert cost.forward(tp).view(()) > 0
    cost.disable_cost()
    assert torch.allclose(cost.forward(tp), torch.zeros(1, 1, 1)), "disabled cost must be 0"
    cost.enable_cost()
    assert cost.forward(tp).view(()) > 0, "re-enabled cost must be > 0 again"


def test_update_target_in_place_drives_cost_to_zero():
    q_id = _norm(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    cost = _build_cost(torch.tensor([0.1, 0.0, 0.0]), q_id, weight=1.0)
    p_a = torch.zeros(3)
    p_b = torch.tensor([0.25, 0.0, 0.0])          # actual relative translation
    tp = _make_tool_pose(p_a, q_id, p_b, q_id)

    assert cost.forward(tp).view(()) > 0          # target 0.1 != actual 0.25
    # Buffer identity must be preserved (CUDA-graph safety) while the value changes.
    tgt_ptr = cost._target_pos.data_ptr()
    cost.update_target([0.25, 0.0, 0.0], q_id.tolist())
    assert cost._target_pos.data_ptr() == tgt_ptr, "update_target must be in place"
    assert torch.allclose(cost.forward(tp), torch.zeros(1, 1, 1), atol=1e-6)


def test_rotation_error_is_double_cover_safe():
    # q and -q are the same rotation: relpose matching -target must still cost ~0.
    q_id = _norm(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    q_tgt = _norm(torch.tensor([0.6, 0.8, 0.0, 0.0]))
    cost = _build_cost(torch.zeros(3), q_tgt, weight=1.0)
    p = torch.zeros(3)
    tp = _make_tool_pose(p, q_id, p, -q_tgt)      # frame_b rotation is -q_tgt (same rotation)
    assert torch.allclose(cost.forward(tp), torch.zeros(1, 1, 1), atol=1e-6), cost.forward(tp)


def test_drift_reports_metres_and_radians():
    q_id = _norm(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    t_rel = torch.tensor([0.1, 0.0, 0.0])
    cost = _build_cost(t_rel, q_id, weight=1.0)
    # frame_b off target by 5 cm and by 30 deg about x.
    angle = torch.tensor(30.0) * torch.pi / 180.0
    q_off = torch.tensor([torch.cos(angle / 2), torch.sin(angle / 2), 0.0, 0.0])
    tp = _make_tool_pose(torch.zeros(3), q_id, t_rel + torch.tensor([0.05, 0.0, 0.0]), q_off)

    pos_drift, rot_drift = cost.compute_drift(tp)
    assert pos_drift.shape == (1, 1, 1) and rot_drift.shape == (1, 1, 1)
    assert torch.allclose(pos_drift.view(()), torch.tensor(0.05), atol=1e-6), pos_drift
    assert torch.allclose(rot_drift.view(()), angle, atol=1e-5), rot_drift


def test_drift_is_zero_while_coupling_inactive():
    """A zero weight must mask the drift, so an uncoupled plan is not judged against it."""
    q_id = _norm(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    cost = _build_cost(torch.tensor([0.1, 0.0, 0.0]), q_id, weight=0.0)
    tp = _make_tool_pose(torch.zeros(3), q_id, torch.tensor([0.9, 0.0, 0.0]), q_id)

    assert cost.is_active().item() == 0.0
    pos_drift, rot_drift = cost.compute_drift(tp)
    assert torch.count_nonzero(pos_drift) == 0 and torch.count_nonzero(rot_drift) == 0

    # Re-weighting (the in-place path used by update_params) must revive the metric.
    cost._weight.copy_(cost.config.weight.new_tensor([1.0]))
    assert cost.is_active().item() == 1.0
    pos_drift, _ = cost.compute_drift(tp)
    assert torch.allclose(pos_drift.view(()), torch.tensor(0.8), atol=1e-6), pos_drift


def test_drift_last_step_carries_horizon_max():
    """Solvers check a single timestep, so the terminal value must see the worst drift."""
    q_id = _norm(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    cost = _build_cost(torch.zeros(3), q_id, weight=1.0)
    # H = 3: frame_b drifts 0.02 -> 0.30 -> 0.01 m while frame_a sits at the origin.
    p_a = torch.zeros(3, 3)
    p_b = torch.tensor([[0.02, 0.0, 0.0], [0.30, 0.0, 0.0], [0.01, 0.0, 0.0]])
    position = torch.stack((p_a, p_b), dim=1).view(1, 3, 2, 3)
    quaternion = q_id.view(1, 1, 1, 4).expand(1, 3, 2, 4).contiguous()
    tp = ToolPose(tool_frames=["A", "B"], position=position, quaternion=quaternion)

    pos_drift, _ = cost.compute_drift(tp)
    assert pos_drift.shape == (1, 3, 1)
    assert torch.allclose(
        pos_drift.view(-1), torch.tensor([0.02, 0.30, 0.30]), atol=1e-6
    ), pos_drift


def test_forward_is_zero_when_weight_is_zero():
    """The cost is gated by its weight alone -- it is evaluated unconditionally every step."""
    q_id = _norm(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    cost = _build_cost(torch.zeros(3), q_id, weight=0.0)
    tp = _make_tool_pose(torch.zeros(3), q_id, torch.tensor([0.5, 0.0, 0.0]), q_id)
    assert torch.allclose(cost.forward(tp), torch.zeros(1, 1, 1))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All relative-pose cost tests passed.")
