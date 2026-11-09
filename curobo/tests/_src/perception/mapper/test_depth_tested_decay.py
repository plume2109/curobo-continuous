# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Depth-tested frustum decay: keep occluded geometry, forget seen-through geometry.

Scene: a camera at the origin looking down +z. A back wall stands at 1.0 m.
An occluder wall is placed at 0.5 m and hides the back wall completely. The
occluder is then removed, so the camera sees the back wall again.
"""

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.util.warp import init_warp
from curobo.tests._src.perception.mapper.conftest import make_observation

IMG_H, IMG_W = 48, 64
BACK_WALL_M = 1.0
OCCLUDER_M = 0.5


@pytest.fixture(scope="module")
def warp_init():
    init_warp()
    return True


def _integrator(depth_margin, voxel_size=0.01, **weighting):
    return BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=4000,
            voxel_size=voxel_size,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(256, 256, 256),
            truncation_distance=voxel_size * 4,
            depth_minimum_distance=0.03,
            depth_maximum_distance=2.0,
            device="cuda:0",
            time_decay=1.0,
            frustum_decay=0.5,
            frustum_decay_depth_margin=depth_margin,
            image_height=IMG_H,
            image_width=IMG_W,
            **weighting,
        )
    )


def _observe(integrator, wall_depth_m, frames, holes=False):
    """Integrate ``frames`` Frames of a flat wall filling the whole image.

    ``holes`` zeroes every 4th pixel column, like stereo dropouts.
    """
    depth = torch.full((IMG_H, IMG_W), wall_depth_m, dtype=torch.float32, device="cuda:0")
    if holes:
        depth[:, ::4] = 0.0
    rgb = torch.full((IMG_H, IMG_W, 3), 128, dtype=torch.uint8, device="cuda:0")
    intrinsics = torch.tensor(
        [[50.0, 0.0, IMG_W / 2], [0.0, 50.0, IMG_H / 2], [0.0, 0.0, 1.0]], device="cuda:0"
    )
    position = torch.zeros(3, device="cuda:0")
    quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0")
    for _ in range(frames):
        integrator.integrate(make_observation(depth, rgb, position, quaternion, intrinsics))


def _surface_voxels_near(integrator, z_m, tolerance_m=0.03):
    """Count central surface voxels within ``tolerance_m`` of depth ``z_m``."""
    centers = integrator.extract_occupied_voxels(surface_only=True).centers
    central = (centers[:, 0].abs() < 0.2) & (centers[:, 1].abs() < 0.15)
    near = (centers[:, 2] - z_m).abs() < tolerance_m
    return int((central & near).sum().item())


def test_occluded_wall_is_retained(warp_init):
    integrator = _integrator(depth_margin=0.04)
    _observe(integrator, BACK_WALL_M, frames=5)
    before = _surface_voxels_near(integrator, BACK_WALL_M)
    _observe(integrator, OCCLUDER_M, frames=30)
    assert before > 0
    assert _surface_voxels_near(integrator, BACK_WALL_M) == before


def test_occluded_wall_is_lost_without_depth_test(warp_init):
    integrator = _integrator(depth_margin=None)
    _observe(integrator, BACK_WALL_M, frames=5)
    _observe(integrator, OCCLUDER_M, frames=30)
    assert _surface_voxels_near(integrator, BACK_WALL_M) == 0


def test_removed_occluder_is_cleared(warp_init):
    integrator = _integrator(depth_margin=0.04)
    _observe(integrator, BACK_WALL_M, frames=5)
    _observe(integrator, OCCLUDER_M, frames=5)
    assert _surface_voxels_near(integrator, OCCLUDER_M) > 0
    _observe(integrator, BACK_WALL_M, frames=20)
    assert _surface_voxels_near(integrator, OCCLUDER_M) == 0
    assert _surface_voxels_near(integrator, BACK_WALL_M) > 0


@pytest.mark.parametrize("holes", [False, True])
def test_occluder_close_to_the_camera_is_cleared(warp_init, holes):
    """An Occluder 7 cm away: its blocks project far past the image border.

    The see-through test must still clear them from the pixels that are in
    view and have valid depth, including when the depth has holes.
    """
    integrator = _integrator(depth_margin=0.005, voxel_size=0.005)
    _observe(integrator, 0.24, frames=5)
    _observe(integrator, 0.07, frames=5)
    assert _surface_voxels_near(integrator, 0.07, tolerance_m=0.01) > 0
    _observe(integrator, 0.24, frames=20, holes=holes)
    assert _surface_voxels_near(integrator, 0.07, tolerance_m=0.01) == 0
    assert _surface_voxels_near(integrator, 0.24, tolerance_m=0.01) > 0


def test_occluded_wall_survives_occluder_with_holes(warp_init):
    """Holes in the Occluder's depth are not evidence that the wall behind is gone."""
    integrator = _integrator(depth_margin=0.005, voxel_size=0.005)
    _observe(integrator, 0.24, frames=5)
    before = _surface_voxels_near(integrator, 0.24, tolerance_m=0.01)
    _observe(integrator, 0.07, frames=30, holes=True)
    assert _surface_voxels_near(integrator, 0.24, tolerance_m=0.01) == before


def _observed_weights(integrator):
    n = int(integrator.tsdf.data.num_allocated.item())
    w = integrator.tsdf.data.block_data[:n, :, 1].float()
    return w[w > 0]


def test_tsdf_weight_is_capped_and_stays_finite(warp_init):
    integrator = _integrator(depth_margin=None, voxel_size=0.005, tsdf_weight_max=10.0)
    _observe(integrator, 0.07, frames=10)
    weights = _observed_weights(integrator)
    assert torch.isfinite(weights).all()
    assert weights.max().item() <= 10.0
    assert _surface_voxels_near(integrator, 0.07, tolerance_m=0.01) > 0


def test_without_coverage_weighting_the_weight_counts_frames(warp_init):
    # The depth test keeps the visible wall from being faded, so only integration counts.
    integrator = _integrator(depth_margin=0.005, voxel_size=0.005, tsdf_coverage_weighting=False)
    _observe(integrator, 0.07, frames=7)
    assert _observed_weights(integrator).max().item() == 7.0
