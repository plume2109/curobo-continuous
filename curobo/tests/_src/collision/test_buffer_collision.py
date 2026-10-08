# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for CollisionBuffer storage layout."""

# Standard Library
import dataclasses

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture
def device_cfg() -> DeviceCfg:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)


def test_buffer_shares_one_storage(device_cfg: DeviceCfg):
    """Distance and gradient are views of one allocation, gradient 16-byte aligned."""
    buffer = CollisionBuffer.from_shape(torch.Size([2, 3, 5, 4]), device_cfg)
    assert buffer.distance.shape == (2, 3, 5)
    assert buffer.gradient.shape == (2, 3, 5, 4)
    assert buffer.distance.is_contiguous() and buffer.gradient.is_contiguous()
    assert buffer.gradient.data_ptr() % 16 == 0
    assert buffer._storage is not None
    assert buffer.distance.untyped_storage().data_ptr() == buffer._storage.data_ptr()
    assert buffer.gradient.untyped_storage().data_ptr() == buffer._storage.data_ptr()


def test_zero_clears_distance_and_gradient(device_cfg: DeviceCfg):
    buffer = CollisionBuffer.from_shape(torch.Size([2, 3, 5, 4]), device_cfg)
    buffer.distance.fill_(1.0)
    buffer.gradient.fill_(2.0)
    buffer.zero_()
    assert torch.count_nonzero(buffer.distance) == 0
    assert torch.count_nonzero(buffer.gradient) == 0


def test_distance_and_gradient_do_not_overlap(device_cfg: DeviceCfg):
    buffer = CollisionBuffer.from_shape(torch.Size([1, 3, 7, 4]), device_cfg)
    buffer.gradient.fill_(2.0)
    assert torch.count_nonzero(buffer.distance) == 0
    buffer.distance.fill_(1.0)
    assert torch.all(buffer.gradient == 2.0)


def test_clone_is_independent(device_cfg: DeviceCfg):
    buffer = CollisionBuffer.from_shape(torch.Size([2, 3, 5, 4]), device_cfg)
    buffer.distance.fill_(1.0)
    buffer.gradient.fill_(2.0)
    copy = buffer.clone()
    buffer.zero_()
    assert torch.all(copy.distance == 1.0)
    assert torch.all(copy.gradient == 2.0)
    copy.zero_()
    assert torch.count_nonzero(copy.distance) == 0
    assert torch.count_nonzero(copy.gradient) == 0


def test_resize_keeps_single_storage(device_cfg: DeviceCfg):
    buffer = CollisionBuffer.from_shape(torch.Size([2, 3, 5, 4]), device_cfg)
    buffer.resize(torch.Size([4, 6, 9, 4]), device_cfg)
    assert buffer.distance.shape == (4, 6, 9)
    assert buffer.gradient.shape == (4, 6, 9, 4)
    buffer.distance.fill_(1.0)
    buffer.gradient.fill_(2.0)
    buffer.zero_()
    assert torch.count_nonzero(buffer.distance) == 0
    assert torch.count_nonzero(buffer.gradient) == 0


def test_mixed_dtypes_use_separate_tensors(device_cfg: DeviceCfg):
    mixed_cfg = dataclasses.replace(device_cfg, collision_gradient_dtype=torch.float16)
    buffer = CollisionBuffer.from_shape(torch.Size([2, 3, 5, 4]), mixed_cfg)
    assert buffer._storage is None
    assert buffer.distance.dtype == torch.float32
    assert buffer.gradient.dtype == torch.float16
    buffer.distance.fill_(1.0)
    buffer.gradient.fill_(2.0)
    buffer.zero_()
    assert torch.count_nonzero(buffer.distance) == 0
    assert torch.count_nonzero(buffer.gradient) == 0
