# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for QuasiNewtonBuffers Gram matrix bookkeeping."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.optim.components.quasi_newton_buffers import QuasiNewtonBuffers
from curobo._src.types.device_cfg import DeviceCfg


def expected_gram(buffers: QuasiNewtonBuffers) -> torch.Tensor:
    s = buffers.s.squeeze(-1)
    y = buffers.y.squeeze(-1)
    # Elementwise products: a matmul would follow the global TF32 setting.
    s_y = (s.transpose(0, 1).unsqueeze(2) * y.transpose(0, 1).unsqueeze(1)).sum(-1)
    y_y = (y.transpose(0, 1).unsqueeze(2) * y.transpose(0, 1).unsqueeze(1)).sum(-1)
    return torch.stack([s_y, y_y], dim=1)


@pytest.fixture
def buffers() -> QuasiNewtonBuffers:
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    buffers = QuasiNewtonBuffers(device_cfg, history=5)
    buffers.resize(num_problems=3, opt_dim=4 * 7)
    gen = torch.Generator(device=device_cfg.device).manual_seed(0)
    buffers.s.copy_(torch.randn(buffers.s.shape, device=device_cfg.device, generator=gen))
    buffers.y.copy_(torch.randn(buffers.y.shape, device=device_cfg.device, generator=gen))
    buffers.refresh_gram()
    return buffers


def test_resize_allocates_zero_gram(buffers: QuasiNewtonBuffers):
    buffers.resize(num_problems=2, opt_dim=8)
    assert buffers.gram.shape == (2, 2, 5, 5)
    assert torch.count_nonzero(buffers.gram) == 0


def test_refresh_gram_matches_history(buffers: QuasiNewtonBuffers):
    assert torch.allclose(buffers.gram, expected_gram(buffers), rtol=1e-5, atol=1e-5)


def test_clear_mask_zeroes_gram_of_masked_problems(buffers: QuasiNewtonBuffers):
    before = buffers.gram.clone()
    mask = torch.tensor([True, False, True], device=buffers.gram.device)
    buffers.clear(mask)
    assert torch.count_nonzero(buffers.gram[mask]) == 0
    assert torch.equal(buffers.gram[~mask], before[~mask])
    assert torch.allclose(buffers.gram, expected_gram(buffers), rtol=1e-5, atol=1e-5)


def test_shift_keeps_gram_consistent(buffers: QuasiNewtonBuffers):
    gram_ptr = buffers.gram.data_ptr()
    buffers.shift(shift_steps=1, action_dim=7)
    assert buffers.gram.data_ptr() == gram_ptr
    assert torch.allclose(buffers.gram, expected_gram(buffers), rtol=1e-5, atol=1e-5)


def test_refresh_gram_ignores_tf32(buffers: QuasiNewtonBuffers):
    reference = buffers.gram.clone()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        buffers.refresh_gram()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    assert torch.equal(buffers.gram, reference)
