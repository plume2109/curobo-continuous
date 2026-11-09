# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Per-voxel depth-tested frustum decay (``frustum_decay_per_voxel``).

Scene: a camera at the origin looking down +z at a plane (a table seen from
above). A box rests on the plane, so its top is nearer to the camera. Blocks
are 4 cm and the plane sits 2 cm into its block, so the box's lower voxels
share blocks with the plane and fail the whole-block see-through test.
"""

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.util.warp import init_warp
from curobo.tests._src.perception.mapper.conftest import make_observation

IMG_H, IMG_W = 96, 128
FOCAL = 100.0
VOXEL = 0.005
BLOCK = 8
MARGIN = 0.01
TRUNCATION = 0.02
MIN_WEIGHT = 0.5
PLANE_M = 0.30
BOX_TOP_M = 0.27
BOX_HALF_M = 0.04


@pytest.fixture(scope="module")
def warp_init():
    init_warp()
    return True


def _integrator(per_voxel: bool) -> BlockSparseTSDFIntegrator:
    return BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=4000,
            voxel_size=VOXEL,
            block_size=BLOCK,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(256, 256, 256),
            truncation_distance=TRUNCATION,
            depth_minimum_distance=0.03,
            depth_maximum_distance=2.0,
            device="cuda:0",
            time_decay=1.0,
            frustum_decay=0.8,
            frustum_decay_depth_margin=MARGIN,
            frustum_decay_per_voxel=per_voxel,
            minimum_tsdf_weight=MIN_WEIGHT,
            tsdf_weight_max=30.0,
            tsdf_coverage_weighting=False,
            image_height=IMG_H,
            image_width=IMG_W,
        )
    )


def _depth(background_m: float, boxes: list[tuple[float, float]]) -> torch.Tensor:
    """Depth image of a plane at ``background_m`` with square boxes facing the camera.

    ``boxes`` holds ``(top_depth_m, half_width_m)``; each box is centred on the
    optical axis and nearer boxes hide farther ones.
    """
    v, u = torch.meshgrid(
        torch.arange(IMG_H, device="cuda:0", dtype=torch.float32) + 0.5,
        torch.arange(IMG_W, device="cuda:0", dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    ray_x = (u - IMG_W / 2) / FOCAL
    ray_y = (v - IMG_H / 2) / FOCAL
    depth = torch.full((IMG_H, IMG_W), background_m, dtype=torch.float32, device="cuda:0")
    for top_m, half_m in sorted(boxes, reverse=True):
        inside = ((ray_x * top_m).abs() < half_m) & ((ray_y * top_m).abs() < half_m)
        depth[inside] = top_m
    return depth


def _observe(integrator: BlockSparseTSDFIntegrator, depth: torch.Tensor, frames: int) -> None:
    rgb = torch.full((IMG_H, IMG_W, 3), 128, dtype=torch.uint8, device="cuda:0")
    intrinsics = torch.tensor(
        [[FOCAL, 0.0, IMG_W / 2], [0.0, FOCAL, IMG_H / 2], [0.0, 0.0, 1.0]], device="cuda:0"
    )
    position = torch.zeros(3, device="cuda:0")
    quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0")
    for _ in range(frames):
        integrator.integrate(make_observation(depth, rgb, position, quaternion, intrinsics))


def _voxels(integrator: BlockSparseTSDFIntegrator):
    """Return voxel centres ``(N, 3)``, SDF ``(N,)`` and weight ``(N,)`` of live blocks.

    With the grid centred on the origin and ``grid_shape`` a multiple of the
    block size, a voxel centre is ``(block_key * BLOCK + local + 0.5) * VOXEL``.
    """
    data = integrator.tsdf.data
    n = int(data.num_allocated.item())
    live = data.block_to_hash_slot[:n] >= 0
    keys = data.block_coords[: n * 3].view(n, 3)[live].float()
    block = data.block_data[:n][live].float()
    local = torch.arange(BLOCK**3, device=keys.device)
    local_xyz = torch.stack(
        [local % BLOCK, (local // BLOCK) % BLOCK, local // (BLOCK * BLOCK)], dim=-1
    ).float()
    centers = ((keys[:, None, :] * BLOCK + local_xyz[None] + 0.5) * VOXEL).view(-1, 3)
    weight = block[:, :, 1].reshape(-1)
    sdf = block[:, :, 0].reshape(-1) / weight.clamp_min(1e-6)
    return centers, sdf, weight


def _in_box_column(centers: torch.Tensor, half_m: float) -> torch.Tensor:
    """Voxels well inside the box footprint, away from its edges."""
    inner = half_m - 2 * VOXEL
    return (centers[:, 0].abs() < inner) & (centers[:, 1].abs() < inner)


def _box_residue(integrator: BlockSparseTSDFIntegrator):
    """Box voxels at least ``MARGIN`` above the plane (nearer the camera)."""
    centers, sdf, weight = _voxels(integrator)
    box = _in_box_column(centers, BOX_HALF_M) & (centers[:, 2] <= PLANE_M - MARGIN)
    occupied = box & (weight >= MIN_WEIGHT) & (sdf <= 0.0)
    # Voxels beyond the plane's truncation band are never re-integrated, so
    # only fading can clear them.
    beyond_band = box & (centers[:, 2] < PLANE_M - TRUNCATION) & (weight > 0.0)
    return occupied, beyond_band, weight


def _plane_surface_voxels(integrator: BlockSparseTSDFIntegrator) -> int:
    centers = integrator.extract_occupied_voxels(surface_only=True).centers
    return int(((centers[:, 2] - PLANE_M).abs() < 2 * VOXEL).sum().item())


def _remove_box(per_voxel: bool) -> BlockSparseTSDFIntegrator:
    integrator = _integrator(per_voxel)
    _observe(integrator, _depth(PLANE_M, [(BOX_TOP_M, BOX_HALF_M)]), frames=30)
    occupied, beyond_band, _ = _box_residue(integrator)
    assert occupied.any()
    assert beyond_band.any()
    _observe(integrator, _depth(PLANE_M, []), frames=20)
    return integrator


def test_removed_object_near_a_surface_is_cleared(warp_init):
    integrator = _remove_box(per_voxel=True)
    occupied, beyond_band, weight = _box_residue(integrator)
    assert not occupied.any()
    assert weight[beyond_band].max().item() < MIN_WEIGHT
    assert _plane_surface_voxels(integrator) > 0


def test_faded_blocks_are_recycled(warp_init):
    """Blocks emptied voxel by voxel are freed by the block_sums / recycle path."""
    integrator = _remove_box(per_voxel=True)
    _observe(integrator, _depth(PLANE_M, []), frames=60)
    centers, _, _ = _voxels(integrator)
    beyond_band = _in_box_column(centers, BOX_HALF_M) & (centers[:, 2] < PLANE_M - TRUNCATION)
    assert not beyond_band.any()
    assert _plane_surface_voxels(integrator) > 0


def test_without_per_voxel_the_residue_stays(warp_init):
    """The whole-block test alone leaves the box voxels sharing blocks with the plane."""
    integrator = _remove_box(per_voxel=False)
    _, beyond_band, weight = _box_residue(integrator)
    assert weight[beyond_band].max().item() >= MIN_WEIGHT


def test_occluded_object_is_retained(warp_init):
    """A back box hidden behind a nearer front box keeps its voxel weights."""
    back_top_m, back_half_m = 0.34, 0.03
    integrator = _integrator(per_voxel=True)
    _observe(integrator, _depth(0.40, [(back_top_m, back_half_m)]), frames=30)
    centers, _, weight_before = _voxels(integrator)
    back = (
        _in_box_column(centers, back_half_m)
        & ((centers[:, 2] - back_top_m).abs() <= TRUNCATION)
        & (weight_before > 0.0)
    )
    assert back.any()
    before = dict(zip(map(tuple, centers[back].tolist()), weight_before[back].tolist()))

    _observe(integrator, _depth(0.40, [(back_top_m, back_half_m), (0.26, 0.05)]), frames=30)
    centers_after, _, weight_after = _voxels(integrator)
    after = dict(zip(map(tuple, centers_after.tolist()), weight_after.tolist()))
    assert all(after.get(c) == w for c, w in before.items())


def test_per_voxel_requires_depth_margin():
    with pytest.raises(ValueError):
        MapperCfg(
            extent_meters_xyz=(1.0, 1.0, 1.0),
            frustum_decay_factor=0.8,
            frustum_decay_per_voxel=True,
        )
