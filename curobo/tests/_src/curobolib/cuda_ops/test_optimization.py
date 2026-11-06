# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optimization CUDA ops."""

from __future__ import annotations

from typing import List

import pytest
import torch

from curobo._src.curobolib.cuda_ops.optimization import wolfe_line_search
from curobo._src.optim.gradient.line_search_context import LineSearchContext
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.types.device_cfg import DeviceCfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def run_wolfe_line_search(
    costs: List[float], slopes: List[float], c_1: float, c_2: float
) -> int:
    """Run the CUDA Wolfe line search on one problem and return the selected index.

    Args:
        costs: Cost at each line-search point; point 0 is the current point.
        slopes: Gradient at each point, as a multiple of the step direction.
        c_1: Armijo constant.
        c_2: Curvature constant.
    """
    device = torch.device("cuda:0")
    action_horizon, action_dim, n_linesearch = 4, 7, len(costs)
    opt_dim = action_horizon * action_dim
    context = LineSearchContext(
        device_cfg=DeviceCfg(device=device, dtype=torch.float32),
        line_search_scale=[0.0, 0.1, 0.5, 1.0],
        line_search_c_1=c_1,
        line_search_c_2=c_2,
        num_problems=1,
        opt_dim=opt_dim,
        action_horizon=action_horizon,
        action_dim=action_dim,
        step_scale=0.98,
        fix_terminal_action=False,
        action_horizon_step_max=None,
        use_cuda_kernel_line_search=True,
        compute_costs_and_gradients=None,
        convergence_iteration=0,
        cost_delta_threshold=0.0,
        cost_relative_threshold=0.0,
    )

    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(*shape, device=device, dtype=dtype)

    direction = torch.ones(1, action_horizon, action_dim, device=device)
    state = OptimizationIterationState(
        action=zeros(1, action_horizon, action_dim),
        cost=zeros(1),
        gradient=zeros(1, action_horizon, action_dim),
        exploration_action=zeros(1, action_horizon, action_dim),
        exploration_gradient=zeros(1, action_horizon, action_dim),
        exploration_cost=zeros(1),
        step_direction=direction,
        best_action=zeros(1, action_horizon, action_dim),
        best_cost=zeros(1),
        best_iteration=zeros(1, dtype=torch.int32),
        current_iteration=zeros(1, dtype=torch.int32),
        converged=zeros(1, dtype=torch.uint8),
    )
    slope = torch.tensor(slopes, device=device).view(1, n_linesearch, 1)
    selected_idx = zeros(1, n_linesearch, dtype=torch.int32)
    wolfe_line_search(
        state,
        context,
        zeros(1, n_linesearch, dtype=torch.int32),
        selected_idx,
        torch.tensor(costs, device=device).view(1, n_linesearch, 1),
        zeros(1, n_linesearch, opt_dim),
        slope * direction.view(1, 1, opt_dim),
        direction.view(1, 1, opt_dim),
        False,
        True,
    )
    return int(selected_idx[0, 0].item())


def test_wolfe_line_search_uses_armijo_constant():
    """Points 1 and 3 raise the cost, so with c_1 = 1e-3 the Armijo condition keeps point 2.

    A c_1 read wrongly by the kernel (e.g. a double read as a float) accepts point 3.
    """
    selected = run_wolfe_line_search(
        costs=[0.25, 0.55, -0.85, 0.55], slopes=[-0.5, -0.5, -0.5, -0.5], c_1=1e-3, c_2=0.9
    )
    assert selected == 2


def test_wolfe_line_search_uses_curvature_constant():
    """Every point decreases the cost; with c_2 = 0.9 only points 1 and 2 satisfy the
    curvature condition, so the line search selects point 2, not point 3."""
    selected = run_wolfe_line_search(
        costs=[0.0, -1.0, -2.0, -3.0], slopes=[-1.0, -0.5, -0.5, -0.95], c_1=1e-3, c_2=0.9
    )
    assert selected == 2
