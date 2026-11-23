# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Clearing from depth seen past the range (``CameraObservation.clearing_depth_image``).

Scene: a camera at the origin looking down +z at a wall within range. The wall
leaves; behind it the camera only sees farther than ``depth_maximum_distance``.
The integrated depth is 0 there (out of range), the clearing depth is ``inf``.
"""

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.util.warp import init_warp
from curobo.tests._src.perception.mapper.conftest import make_observation

IMG_H, IMG_W = 96, 128
FOCAL = 100.0
VOXEL = 0.005
MAX_DEPTH = 0.9
WALL_M = 0.6


@pytest.fixture(scope="module")
def warp_init():
    init_warp()
    return True


def _integrator() -> BlockSparseTSDFIntegrator:
    return BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=8000,
            voxel_size=VOXEL,
            block_size=8,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            truncation_distance=0.02,
            depth_minimum_distance=0.03,
            depth_maximum_distance=MAX_DEPTH,
            device="cuda:0",
            time_decay=1.0,
            frustum_decay=0.8,
            frustum_decay_depth_margin=0.01,
            frustum_decay_per_voxel=True,
            minimum_tsdf_weight=2.5,
            tsdf_weight_max=30.0,
            tsdf_coverage_weighting=False,
            image_height=IMG_H,
            image_width=IMG_W,
        )
    )


def _observe(integrator, depth_m: float, frames: int, clearing_m=None, z=0.0) -> None:
    depth = torch.full((IMG_H, IMG_W), depth_m, dtype=torch.float32, device="cuda:0")
    rgb = torch.full((IMG_H, IMG_W, 3), 128, dtype=torch.uint8, device="cuda:0")
    intrinsics = torch.tensor(
        [[FOCAL, 0.0, IMG_W / 2], [0.0, FOCAL, IMG_H / 2], [0.0, 0.0, 1.0]], device="cuda:0"
    )
    position = torch.tensor([0.0, 0.0, z], device="cuda:0")
    quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0")
    for _ in range(frames):
        obs = make_observation(depth, rgb, position, quaternion, intrinsics)
        if clearing_m is not None:
            obs.clearing_depth_image = torch.full_like(obs.depth_image, clearing_m)
        integrator.integrate(obs)


def _occupied(integrator) -> int:
    return len(integrator.extract_occupied_voxels(surface_only=False).centers)


def _with_wall() -> BlockSparseTSDFIntegrator:
    integrator = _integrator()
    _observe(integrator, WALL_M, frames=10)
    assert _occupied(integrator) > 0
    return integrator


def test_seen_past_the_range_clears_within_it(warp_init):
    integrator = _with_wall()
    _observe(integrator, 0.0, frames=30, clearing_m=float("inf"))
    assert _occupied(integrator) == 0


def test_without_clearing_depth_out_of_range_is_no_evidence(warp_init):
    integrator = _with_wall()
    before = _occupied(integrator)
    _observe(integrator, 0.0, frames=30)
    assert _occupied(integrator) == before


def test_seen_past_the_range_never_clears_beyond_it(warp_init):
    """From 0.5 m back the wall is 1.1 m away, beyond the 0.9 m range."""
    integrator = _with_wall()
    before = _occupied(integrator)
    _observe(integrator, 0.0, frames=30, clearing_m=float("inf"), z=-0.5)
    assert _occupied(integrator) == before


def test_clearing_depth_must_match_the_depth_shape(warp_init):
    integrator = _with_wall()
    depth = torch.zeros((IMG_H, IMG_W), device="cuda:0")
    obs = make_observation(
        depth,
        torch.zeros((IMG_H, IMG_W, 3), dtype=torch.uint8, device="cuda:0"),
        torch.zeros(3, device="cuda:0"),
        torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0"),
        torch.tensor([[FOCAL, 0, IMG_W / 2], [0, FOCAL, IMG_H / 2], [0, 0, 1.0]], device="cuda:0"),
    )
    obs.clearing_depth_image = torch.zeros((1, IMG_H, IMG_W + 1), device="cuda:0")
    with pytest.raises(Exception, match="clearing_depth_image"):
        integrator.integrate(obs)
