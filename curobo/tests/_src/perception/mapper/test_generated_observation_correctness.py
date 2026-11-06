# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Generated-observation correctness tests for the high-level Mapper API."""

from __future__ import annotations

import math

import pytest
import torch

from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
from curobo._src.util.warp import init_warp


IMAGE_H = 32
IMAGE_W = 40
VOXEL_SIZE = 0.02
PLANE_Z = 1.0


@pytest.fixture(scope="module")
def warp_init():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for block-sparse mapper integration tests")
    init_warp()
    return True


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for block-sparse mapper integration tests")
    return "cuda:0"


def _intrinsics(
    device: str,
    image_height: int = IMAGE_H,
    image_width: int = IMAGE_W,
    focal: float = 80.0,
    cx: float | None = None,
    cy: float | None = None,
) -> torch.Tensor:
    if cx is None:
        cx = image_width / 2.0
    if cy is None:
        cy = image_height / 2.0
    return torch.tensor(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


def _identity_pose(device: str, num_cameras: int = 1) -> Pose:
    position = torch.zeros((num_cameras, 3), dtype=torch.float32, device=device)
    quaternion = torch.zeros((num_cameras, 4), dtype=torch.float32, device=device)
    quaternion[:, 0] = 1.0
    return Pose(position=position, quaternion=quaternion)


def _make_mapper(
    device: str,
    *,
    image_height: int = IMAGE_H,
    image_width: int = IMAGE_W,
    num_cameras: int = 1,
    feature_dim: int = 0,
    feature_integration_kernel: str = "auto",
    feature_channels_per_thread: int = 8,
    support_capacity: int = 8,
    decay_factor: float = 1.0,
    frustum_decay_factor: float = 1.0,
    profile_kernel_timings: bool = False,
    grid_center: tuple[float, float, float] = (0.0, 0.0, PLANE_Z),
) -> Mapper:
    feature_grid_kwargs = {}
    if feature_dim > 0:
        feature_grid_kwargs = {
            "feature_grid_height": 7,
            "feature_grid_width": 11,
        }
    return Mapper(
        MapperCfg(
            extent_meters_xyz=(1.0, 0.8, 0.8),
            extent_esdf_meters_xyz=(1.0, 0.8, 0.8),
            voxel_size=VOXEL_SIZE,
            esdf_voxel_size=0.04,
            grid_center=torch.tensor(grid_center, dtype=torch.float32),
            truncation_distance=0.04,
            depth_minimum_distance=0.1,
            depth_maximum_distance=3.0,
            decay_factor=decay_factor,
            frustum_decay_factor=frustum_decay_factor,
            image_height=image_height,
            image_width=image_width,
            num_cameras=num_cameras,
            block_size=2,
            max_support_pixels_per_block_camera=support_capacity,
            feature_dim=feature_dim,
            **feature_grid_kwargs,
            feature_channels_per_thread=feature_channels_per_thread,
            feature_integration_kernel=feature_integration_kernel,
            profile_integration_kernel_timings=profile_kernel_timings,
            device=device,
        )
    )


def _observation(
    *,
    device: str,
    depth: torch.Tensor,
    rgb: torch.Tensor,
    intrinsics: torch.Tensor | None = None,
    feature_grid: torch.Tensor | None = None,
    pose: Pose | None = None,
) -> CameraObservation:
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if intrinsics is None:
        intrinsics = _intrinsics(device, depth.shape[-2], depth.shape[-1])
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if pose is None:
        pose = _identity_pose(device, depth.shape[0])
    return CameraObservation(
        depth_image=depth,
        rgb_image=rgb,
        pose=pose,
        intrinsics=intrinsics,
        feature_grid=feature_grid,
    )


def _constant_plane_observation(
    device: str,
    *,
    image_height: int = IMAGE_H,
    image_width: int = IMAGE_W,
    rgb_value: tuple[int, int, int] = (128, 128, 128),
    feature_grid: torch.Tensor | None = None,
    num_cameras: int = 1,
) -> CameraObservation:
    depth = torch.full(
        (num_cameras, image_height, image_width),
        PLANE_Z,
        dtype=torch.float32,
        device=device,
    )
    rgb = torch.empty(
        (num_cameras, image_height, image_width, 3),
        dtype=torch.uint8,
        device=device,
    )
    rgb[..., 0] = rgb_value[0]
    rgb[..., 1] = rgb_value[1]
    rgb[..., 2] = rgb_value[2]
    return _observation(
        device=device,
        depth=depth,
        rgb=rgb,
        feature_grid=feature_grid,
        pose=_identity_pose(device, num_cameras),
        intrinsics=_intrinsics(device, image_height, image_width).view(1, 3, 3)
        .expand(num_cameras, 3, 3)
        .contiguous(),
    )


def _render_identity_plane_depth(
    device: str,
    *,
    normal: torch.Tensor,
    point: torch.Tensor,
    image_height: int = IMAGE_H,
    image_width: int = IMAGE_W,
) -> torch.Tensor:
    intr = _intrinsics(device, image_height, image_width)
    u = torch.arange(image_width, dtype=torch.float32, device=device)
    v = torch.arange(image_height, dtype=torch.float32, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    x_norm = (uu - intr[0, 2]) / intr[0, 0]
    y_norm = (vv - intr[1, 2]) / intr[1, 1]
    rays = torch.stack(
        (x_norm, y_norm, torch.ones_like(x_norm)),
        dim=-1,
    )
    normal = normal.to(device=device, dtype=torch.float32)
    normal = normal / normal.norm().clamp(min=1e-6)
    point = point.to(device=device, dtype=torch.float32)
    numerator = torch.dot(normal, point)
    denominator = (rays * normal.view(1, 1, 3)).sum(dim=-1).clamp(min=1e-4)
    return (numerator / denominator).contiguous()


def _constant_feature_grid(
    device: str,
    feature_vector: torch.Tensor,
    *,
    num_cameras: int = 1,
    feature_height: int = 7,
    feature_width: int = 11,
) -> torch.Tensor:
    feature_vector = feature_vector.to(device=device, dtype=torch.float16)
    return (
        feature_vector.view(1, 1, 1, -1)
        .expand(num_cameras, feature_height, feature_width, feature_vector.numel())
        .contiguous()
    )


def _spatial_feature_grid(
    device: str,
    *,
    feature_dim: int,
    feature_height: int = 7,
    feature_width: int = 11,
    num_cameras: int = 1,
) -> torch.Tensor:
    gy = torch.arange(feature_height, dtype=torch.float32, device=device).view(
        feature_height, 1
    )
    gx = torch.arange(feature_width, dtype=torch.float32, device=device).view(
        1, feature_width
    )
    gx_norm = gx / float(max(feature_width - 1, 1))
    gy_norm = gy / float(max(feature_height - 1, 1))
    channels = []
    for ch in range(feature_dim):
        if ch % 4 == 0:
            values = gx_norm.expand(feature_height, feature_width)
        elif ch % 4 == 1:
            values = gy_norm.expand(feature_height, feature_width)
        elif ch % 4 == 2:
            values = 0.5 * (gx_norm + gy_norm)
        else:
            values = torch.full(
                (feature_height, feature_width),
                -0.25 + 0.05 * ch,
                dtype=torch.float32,
                device=device,
            )
        channels.append(values)
    grid = torch.stack(channels, dim=-1).to(torch.float16)
    return grid.view(1, feature_height, feature_width, feature_dim).expand(
        num_cameras, feature_height, feature_width, feature_dim
    ).contiguous()


def _active_rgb(mapper: Mapper) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = mapper.tsdf.data
    n = int(data.num_allocated.item())
    block_grid_rgb = data.block_grid_rgb[:n, 0].float()
    active = block_grid_rgb[:, 3] > 0
    pool_idx = torch.arange(n, dtype=torch.int64, device=block_grid_rgb.device)[active]
    normalized = block_grid_rgb[active, :3] / block_grid_rgb[active, 3:4].clamp(
        min=1e-6
    )
    return pool_idx, normalized, block_grid_rgb[active, 3]


def _active_features(mapper: Mapper) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = mapper.tsdf.data
    n = int(data.num_allocated.item())
    weights = data.block_feature_weight[:n].float().sum(dim=1)
    active = weights > 0
    pool_idx = torch.arange(n, dtype=torch.int64, device=weights.device)[active]
    feature_sum = data.block_features[:n].float().sum(dim=1)
    normalized = feature_sum[active] / weights[active].view(-1, 1)
    return pool_idx, normalized, weights[active]


def _sort_active_features_by_coord(mapper: Mapper) -> tuple[torch.Tensor, torch.Tensor]:
    pool_idx, normalized, _ = _active_features(mapper)
    coords = mapper.tsdf.data.block_coords.view(-1, 3)[pool_idx].long()
    sort_key = coords[:, 0] * (2**40) + coords[:, 1] * (2**20) + coords[:, 2]
    order = sort_key.argsort()
    return coords[order], normalized[order]


def _grid_node_world(mapper: Mapper, pool_idx: torch.Tensor) -> torch.Tensor:
    data = mapper.tsdf.data
    coords = data.block_coords.view(data.max_blocks, 3)[pool_idx.long()].float()
    block_size = int(data.block_size)
    grid_d, grid_h, grid_w = (int(v) for v in data.grid_shape)
    blocks_x = (grid_w + block_size - 1) // block_size
    blocks_y = (grid_h + block_size - 1) // block_size
    blocks_z = (grid_d + block_size - 1) // block_size
    offsets = torch.tensor(
        [blocks_x // 2, blocks_y // 2, blocks_z // 2],
        dtype=torch.float32,
        device=pool_idx.device,
    )
    center_offset = torch.tensor(
        [grid_w, grid_h, grid_d],
        dtype=torch.float32,
        device=pool_idx.device,
    ) * 0.5
    local = torch.full((3,), block_size * 0.5, dtype=torch.float32, device=pool_idx.device)
    voxel = (coords + offsets) * float(block_size) + local
    origin = data.origin.to(device=pool_idx.device, dtype=torch.float32)
    return origin + (voxel - center_offset) * float(data.voxel_size)


def _pool_to_visible_slot(mapper: Mapper) -> torch.Tensor:
    camera_integrator = mapper.integrator._tsdf_integrator._camera_integrator
    n_visible = int(camera_integrator.visible_count.item())
    max_pool = int(mapper.tsdf.data.num_allocated.item())
    pool_to_vis = torch.full(
        (max_pool,),
        -1,
        dtype=torch.int64,
        device=camera_integrator.pool_indices.device,
    )
    for vis_idx in range(n_visible):
        pool = int(camera_integrator.pool_indices[vis_idx].item())
        if 0 <= pool < max_pool:
            pool_to_vis[pool] = vis_idx
    return pool_to_vis


def _support_rgb_reference(
    mapper: Mapper,
    rgb: torch.Tensor,
    pool_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera_integrator = mapper.integrator._tsdf_integrator._camera_integrator
    cfg = mapper.config
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    support = torch.zeros((pool_idx.numel(), 3), dtype=torch.float32, device=rgb.device)
    support_weight = torch.zeros(pool_idx.numel(), dtype=torch.float32, device=rgb.device)
    pool_to_vis = _pool_to_visible_slot(mapper)
    for out_idx, pool in enumerate(pool_idx.long().tolist()):
        vis_idx = int(pool_to_vis[pool].item())
        if vis_idx < 0:
            continue
        for cam_i in range(rgb.shape[0]):
            count = int(camera_integrator.support_counts[vis_idx, cam_i].item())
            if count <= 0:
                continue
            pixel_idx = int(camera_integrator.support_pixels[vis_idx, cam_i, 0].item())
            py = pixel_idx // cfg.image_width
            px = pixel_idx - py * cfg.image_width
            if 0 <= py < cfg.image_height and 0 <= px < cfg.image_width:
                support[out_idx] += rgb[cam_i, py, px].float() / 255.0
                support_weight[out_idx] += 1.0
    active = support_weight > 0
    support[active] = support[active] / support_weight[active].unsqueeze(-1)
    return support, active


def _support_feature_reference(
    mapper: Mapper,
    feature_grid: torch.Tensor,
    pool_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera_integrator = mapper.integrator._tsdf_integrator._camera_integrator
    cfg = mapper.config
    feature_h = feature_grid.shape[1]
    feature_w = feature_grid.shape[2]
    support = torch.zeros(
        (pool_idx.numel(), feature_grid.shape[-1]),
        dtype=torch.float32,
        device=feature_grid.device,
    )
    support_weight = torch.zeros(pool_idx.numel(), dtype=torch.float32, device=feature_grid.device)
    pool_to_vis = _pool_to_visible_slot(mapper)
    for out_idx, pool in enumerate(pool_idx.long().tolist()):
        vis_idx = int(pool_to_vis[pool].item())
        if vis_idx < 0:
            continue
        for cam_i in range(feature_grid.shape[0]):
            count = int(camera_integrator.support_counts[vis_idx, cam_i].item())
            if count <= 0:
                continue
            pixel_idx = int(camera_integrator.support_pixels[vis_idx, cam_i, 0].item())
            py = pixel_idx // cfg.image_width
            px = pixel_idx - py * cfg.image_width
            if 0 <= py < cfg.image_height and 0 <= px < cfg.image_width:
                gy = max(0, min(feature_h - 1, (py * feature_h) // cfg.image_height))
                gx = max(0, min(feature_w - 1, (px * feature_w) // cfg.image_width))
                support[out_idx] += feature_grid[cam_i, gy, gx].float()
                support_weight[out_idx] += 1.0
    active = support_weight > 0
    support[active] = support[active] / support_weight[active].unsqueeze(-1)
    return support, active


def _expected_rgb_from_grid_node(
    mapper: Mapper,
    depth: torch.Tensor,
    rgb: torch.Tensor,
    pool_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected RGB for the block color-grid node, with support fallback."""
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    assert rgb.shape[0] == depth.shape[0]
    data = mapper.tsdf.data
    cfg = mapper.config
    world = _grid_node_world(mapper, pool_idx).to(device=rgb.device)

    intrinsics = _intrinsics(str(rgb.device), cfg.image_height, cfg.image_width)
    z = world[:, 2]
    u = intrinsics[0, 0] * world[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * world[:, 1] / z + intrinsics[1, 2]
    projected = (
        (z > cfg.depth_minimum_distance)
        & (z <= cfg.depth_maximum_distance)
        & (u >= 0.0)
        & (u <= cfg.image_width - 1)
        & (v >= 0.0)
        & (v <= cfg.image_height - 1)
    )

    u_clamped = u.clamp(0.0, cfg.image_width - 1)
    v_clamped = v.clamp(0.0, cfg.image_height - 1)
    px0 = torch.floor(u_clamped).long()
    py0 = torch.floor(v_clamped).long()
    px1 = (px0 + 1).clamp(max=cfg.image_width - 1)
    py1 = (py0 + 1).clamp(max=cfg.image_height - 1)
    tx = (u_clamped - px0.float()).unsqueeze(-1)
    ty = (v_clamped - py0.float()).unsqueeze(-1)

    image = rgb[0].float() / 255.0
    depth0 = depth[0].float()
    d00 = depth0[py0, px0]
    d10 = depth0[py0, px1]
    d01 = depth0[py1, px0]
    d11 = depth0[py1, px1]
    depths = (d00, d10, d01, d11)
    pixels = (image[py0, px0], image[py0, px1], image[py1, px0], image[py1, px1])
    bilinear = (
        (1.0 - tx.squeeze(-1)) * (1.0 - ty.squeeze(-1)),
        tx.squeeze(-1) * (1.0 - ty.squeeze(-1)),
        (1.0 - tx.squeeze(-1)) * ty.squeeze(-1),
        tx.squeeze(-1) * ty.squeeze(-1),
    )
    coverage = torch.maximum(
        (intrinsics[0, 0] * float(data.voxel_size) / z)
        * (intrinsics[1, 1] * float(data.voxel_size) / z),
        torch.ones_like(z),
    )
    total_rgb = torch.zeros((pool_idx.numel(), 3), dtype=torch.float32, device=rgb.device)
    total_w = torch.zeros(pool_idx.numel(), dtype=torch.float32, device=rgb.device)
    for sample_depth, sample_rgb, weight in zip(depths, pixels, bilinear, strict=True):
        sdf = sample_depth - z
        sample_valid = (
            projected
            & (sample_depth >= cfg.depth_minimum_distance)
            & (sample_depth <= cfg.depth_maximum_distance)
            & (sdf >= -cfg.truncation_distance)
            & (sdf <= cfg.truncation_distance)
        )
        sample_w = torch.where(sample_valid, weight * coverage, torch.zeros_like(weight))
        total_rgb += sample_rgb * sample_w.unsqueeze(-1)
        total_w += sample_w

    direct = total_w > 0
    expected = torch.zeros_like(total_rgb)
    expected[direct] = total_rgb[direct] / total_w[direct].unsqueeze(-1)
    support, support_active = _support_rgb_reference(mapper, rgb, pool_idx)
    fallback = ~direct & support_active
    expected[fallback] = support[fallback]
    return expected, direct | fallback


def _expected_features_from_grid_node(
    mapper: Mapper,
    depth: torch.Tensor,
    feature_grid: torch.Tensor,
    pool_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected feature for the block feature-grid node, with support fallback."""
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    data = mapper.tsdf.data
    cfg = mapper.config
    world = _grid_node_world(mapper, pool_idx).to(device=feature_grid.device)
    intrinsics = _intrinsics(str(feature_grid.device), cfg.image_height, cfg.image_width)
    z = world[:, 2]
    u = intrinsics[0, 0] * world[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * world[:, 1] / z + intrinsics[1, 2]
    px = torch.floor(u + 0.5).long()
    py = torch.floor(v + 0.5).long()
    projected = (
        (z > cfg.depth_minimum_distance)
        & (z <= cfg.depth_maximum_distance)
        & (px >= 0)
        & (px < cfg.image_width)
        & (py >= 0)
        & (py < cfg.image_height)
    )
    px_clamped = px.clamp(0, cfg.image_width - 1)
    py_clamped = py.clamp(0, cfg.image_height - 1)
    depth0 = depth[0].float()
    sample_depth = depth0[py_clamped, px_clamped]
    sdf = sample_depth - z
    direct = (
        projected
        & (sample_depth >= cfg.depth_minimum_distance)
        & (sample_depth <= cfg.depth_maximum_distance)
        & (sdf >= -cfg.truncation_distance)
        & (sdf <= cfg.truncation_distance)
    )
    feature_h = feature_grid.shape[1]
    feature_w = feature_grid.shape[2]
    gy = ((py_clamped * feature_h) // cfg.image_height).clamp(0, feature_h - 1)
    gx = ((px_clamped * feature_w) // cfg.image_width).clamp(0, feature_w - 1)
    expected = torch.zeros(
        (pool_idx.numel(), feature_grid.shape[-1]),
        dtype=torch.float32,
        device=feature_grid.device,
    )
    expected[direct] = feature_grid[0, gy[direct], gx[direct]].float()
    support, support_active = _support_feature_reference(mapper, feature_grid, pool_idx)
    fallback = ~direct & support_active
    expected[fallback] = support[fallback]
    return expected, direct | fallback


def _expected_features_from_support(
    mapper: Mapper,
    feature_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera_integrator = mapper.integrator._tsdf_integrator._camera_integrator
    n_visible = int(camera_integrator.visible_count.item())
    max_pool = int(mapper.tsdf.data.num_allocated.item())
    feature_dim = feature_grid.shape[-1]
    expected_sum = torch.zeros(
        (max_pool, feature_dim),
        dtype=torch.float32,
        device=feature_grid.device,
    )
    expected_weight = torch.zeros(max_pool, dtype=torch.float32, device=feature_grid.device)
    feature_h = feature_grid.shape[1]
    feature_w = feature_grid.shape[2]
    for vis_idx in range(n_visible):
        pool_idx = int(camera_integrator.pool_indices[vis_idx].item())
        if pool_idx < 0:
            continue
        for cam_i in range(feature_grid.shape[0]):
            count = int(camera_integrator.support_counts[vis_idx, cam_i].item())
            count = min(count, mapper.config.max_support_pixels_per_block_camera)
            if count <= 0:
                continue
            pixels = camera_integrator.support_pixels[vis_idx, cam_i, :count].long()
            py = pixels // mapper.config.image_width
            px = pixels - py * mapper.config.image_width
            valid = (
                (py >= 0)
                & (py < mapper.config.image_height)
                & (px >= 0)
                & (px < mapper.config.image_width)
            )
            if valid.any():
                gy = ((py[valid] * feature_h) // mapper.config.image_height).clamp(
                    min=0, max=feature_h - 1
                )
                gx = ((px[valid] * feature_w) // mapper.config.image_width).clamp(
                    min=0, max=feature_w - 1
                )
                values = feature_grid[cam_i, gy, gx].float()
                expected_sum[pool_idx] += values.sum(dim=0)
                expected_weight[pool_idx] += float(values.shape[0])
    return expected_sum, expected_weight


def _sample_esdf_nearest(mapper: Mapper, points: torch.Tensor) -> torch.Tensor:
    grid = mapper.compute_esdf()
    field = grid.feature_tensor
    origin = torch.tensor(grid.pose[:3], dtype=torch.float32, device=points.device)
    dims = torch.tensor(grid.dims, dtype=torch.float32, device=points.device)
    idx = torch.round((points - origin + 0.5 * dims) / grid.voxel_size).long()
    idx[:, 0].clamp_(0, field.shape[0] - 1)
    idx[:, 1].clamp_(0, field.shape[1] - 1)
    idx[:, 2].clamp_(0, field.shape[2] - 1)
    return field[idx[:, 0], idx[:, 1], idx[:, 2]].float()


def test_plane_surface_and_esdf_distance_are_voxel_accurate(warp_init, device):
    mapper = _make_mapper(device)
    obs = _constant_plane_observation(device, rgb_value=(128, 128, 128))
    mapper.integrate(obs)

    stats = mapper.get_stats(scan_pool=False)
    assert stats["last_integration"]["num_visible_blocks"] > 0
    assert stats["last_integration_kernel_timings_ms"] == {}

    surface = mapper.extract_occupied_voxels(surface_only=True, sdf_threshold=0.04)
    assert len(surface) > 0
    assert torch.isfinite(surface.centers).all()
    median_z = surface.centers[:, 2].median()
    assert torch.abs(median_z - PLANE_Z) <= 2.0 * VOXEL_SIZE

    points = torch.tensor(
        [
            [0.0, 0.0, PLANE_Z],
            [0.0, 0.0, PLANE_Z - 0.12],
            [0.0, 0.0, PLANE_Z + 0.12],
        ],
        dtype=torch.float32,
        device=device,
    )
    distances = _sample_esdf_nearest(mapper, points)
    assert distances[0] <= 2.0 * VOXEL_SIZE
    torch.testing.assert_close(
        distances[1:],
        torch.full((2,), 0.12, dtype=torch.float32, device=device),
        atol=0.05,
        rtol=0.0,
    )


def _map_snapshot(mapper: Mapper) -> dict[str, torch.Tensor]:
    """TSDF blocks sorted by key, ESDF, and sorted occupied voxels of a mapper."""
    blocks = mapper.tsdf.export_blocks()
    keys = blocks["active_block_coords"].long()
    order = torch.argsort((keys[:, 0] * 4096 + keys[:, 1]) * 4096 + keys[:, 2])
    voxel_grid = mapper.compute_esdf()
    occupied = mapper.extract_occupied_voxels(surface_only=False).centers
    occupied_order = torch.argsort(
        occupied[:, 0] * 1.0e6 + occupied[:, 1] * 1.0e3 + occupied[:, 2]
    )
    return {
        "block_keys": keys[order],
        "block_data": blocks["block_data"][order],
        "esdf": voxel_grid.feature_tensor.clone(),
        "esdf_pose": torch.as_tensor(voxel_grid.pose, dtype=torch.float32),
        "occupied": occupied[occupied_order],
    }


def test_set_origin_matches_mapper_built_at_new_center(warp_init, device):
    """set_origin() on a warm mapper gives the same map as a fresh mapper at that center."""
    new_center = (0.013, -0.027, PLANE_Z + 0.031)
    depth = torch.full((IMAGE_H, IMAGE_W), PLANE_Z, dtype=torch.float32, device=device)
    depth[8:16, 10:20] = PLANE_Z - 0.2
    rgb = torch.full((IMAGE_H, IMAGE_W, 3), 90, dtype=torch.uint8, device=device)
    obs = _observation(device=device, depth=depth, rgb=rgb)

    expected_mapper = _make_mapper(device, grid_center=new_center)
    expected_mapper.integrate(obs)
    expected = _map_snapshot(expected_mapper)

    mapper = _make_mapper(device)
    mapper.integrate(obs)
    _map_snapshot(mapper)  # warm kernels and the ESDF CUDA graph on the old center
    mapper.set_origin(torch.tensor(new_center))
    assert mapper.tsdf.data.num_allocated.item() == 0
    mapper.integrate(obs)
    actual = _map_snapshot(mapper)

    assert expected["block_keys"].shape[0] > 0
    assert expected["occupied"].shape[0] > 0
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name
    assert torch.equal(mapper.config.grid_center, torch.tensor(new_center))


def test_tilted_plane_surface_matches_analytic_plane(warp_init, device):
    normal = torch.tensor([0.25, -0.10, 1.0], dtype=torch.float32, device=device)
    normal = normal / normal.norm().clamp(min=1e-6)
    point = torch.tensor([0.0, 0.0, PLANE_Z], dtype=torch.float32, device=device)
    depth = _render_identity_plane_depth(device, normal=normal, point=point)
    rgb = torch.full((IMAGE_H, IMAGE_W, 3), 127, dtype=torch.uint8, device=device)

    mapper = _make_mapper(device)
    mapper.integrate(_observation(device=device, depth=depth, rgb=rgb))
    surface = mapper.extract_occupied_voxels(surface_only=True, sdf_threshold=0.04)

    assert len(surface) > 0
    plane_distance = torch.abs((surface.centers - point).matmul(normal))
    assert plane_distance.median() <= 2.0 * VOXEL_SIZE
    assert torch.quantile(plane_distance, 0.90) <= 4.0 * VOXEL_SIZE


def test_constant_rgb_accumulates_across_frames(warp_init, device):
    mapper = _make_mapper(device)
    mapper.integrate(_constant_plane_observation(device, rgb_value=(255, 0, 0)))
    pool_idx, rgb_first, weights_first = _active_rgb(mapper)
    assert pool_idx.numel() > 0
    expected_red = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    torch.testing.assert_close(
        rgb_first,
        expected_red.view(1, 3).expand_as(rgb_first),
        atol=0.03,
        rtol=0.0,
    )

    mapper.integrate(_constant_plane_observation(device, rgb_value=(0, 0, 255)))
    _, rgb_second, weights_second = _active_rgb(mapper)
    expected_purple = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float32, device=device)
    torch.testing.assert_close(
        rgb_second,
        expected_purple.view(1, 3).expand_as(rgb_second),
        atol=0.03,
        rtol=0.0,
    )
    assert weights_second.sum() > weights_first.sum()


def test_gradient_rgb_matches_color_grid_projection(warp_init, device):
    mapper = _make_mapper(device, support_capacity=8)
    x = torch.arange(IMAGE_W, dtype=torch.float32, device=device).view(1, IMAGE_W)
    y = torch.arange(IMAGE_H, dtype=torch.float32, device=device).view(IMAGE_H, 1)
    rgb = torch.empty((IMAGE_H, IMAGE_W, 3), dtype=torch.uint8, device=device)
    rgb[..., 0] = torch.round(x.expand(IMAGE_H, IMAGE_W) * 255.0 / (IMAGE_W - 1)).to(
        torch.uint8
    )
    rgb[..., 1] = torch.round(y.expand(IMAGE_H, IMAGE_W) * 255.0 / (IMAGE_H - 1)).to(
        torch.uint8
    )
    rgb[..., 2] = 64
    depth = torch.full((IMAGE_H, IMAGE_W), PLANE_Z, dtype=torch.float32, device=device)

    mapper.integrate(_observation(device=device, depth=depth, rgb=rgb))
    pool_idx, normalized, _ = _active_rgb(mapper)
    expected_active, valid = _expected_rgb_from_grid_node(mapper, depth, rgb, pool_idx)

    assert valid.all()
    torch.testing.assert_close(normalized, expected_active, atol=0.03, rtol=0.0)


@pytest.mark.parametrize("feature_kernel", ["grouped", "tiled"])
def test_constant_features_include_trailing_channels(warp_init, device, feature_kernel):
    feature_vector = torch.tensor(
        [-0.75, -0.25, 0.0, 0.25, 0.50, 0.75, 1.0],
        dtype=torch.float32,
        device=device,
    )
    feature_grid = _constant_feature_grid(device, feature_vector)
    mapper = _make_mapper(
        device,
        feature_dim=feature_vector.numel(),
        feature_integration_kernel=feature_kernel,
        feature_channels_per_thread=5,
    )

    mapper.integrate(
        _constant_plane_observation(device, rgb_value=(0, 0, 0), feature_grid=feature_grid)
    )
    _, normalized, weights = _active_features(mapper)

    assert weights.numel() > 0
    torch.testing.assert_close(
        normalized,
        feature_vector.view(1, -1).expand_as(normalized),
        atol=0.03,
        rtol=0.0,
    )


@pytest.mark.parametrize("feature_kernel", ["grouped", "tiled"])
def test_spatial_features_match_grid_node_reference(
    warp_init,
    device,
    feature_kernel,
):
    feature_dim = 9
    feature_grid = _spatial_feature_grid(device, feature_dim=feature_dim)
    mapper = _make_mapper(
        device,
        feature_dim=feature_dim,
        feature_integration_kernel=feature_kernel,
        support_capacity=8,
    )

    obs = _constant_plane_observation(
        device,
        rgb_value=(0, 0, 0),
        feature_grid=feature_grid,
    )
    mapper.integrate(obs)
    pool_idx, normalized, _ = _active_features(mapper)
    expected_active, valid = _expected_features_from_grid_node(
        mapper,
        obs.depth_image,
        feature_grid,
        pool_idx,
    )

    assert valid.all()
    torch.testing.assert_close(normalized, expected_active, atol=0.04, rtol=0.0)


def test_grouped_tiled_feature_outputs_match_for_same_scene(warp_init, device):
    feature_dim = 9
    feature_vector = torch.linspace(
        -1.0,
        1.0,
        feature_dim,
        dtype=torch.float32,
        device=device,
    )
    feature_grid = _constant_feature_grid(device, feature_vector)
    obs = _constant_plane_observation(
        device,
        rgb_value=(0, 0, 0),
        feature_grid=feature_grid,
    )
    grouped = _make_mapper(device, feature_dim=feature_dim, feature_integration_kernel="grouped")
    tiled = _make_mapper(device, feature_dim=feature_dim, feature_integration_kernel="tiled")

    grouped.integrate(obs)
    tiled.integrate(obs)
    grouped_coords, grouped_features = _sort_active_features_by_coord(grouped)
    tiled_coords, tiled_features = _sort_active_features_by_coord(tiled)

    assert grouped_coords.shape == tiled_coords.shape
    torch.testing.assert_close(grouped_coords, tiled_coords)
    torch.testing.assert_close(grouped_features, tiled_features, atol=0.04, rtol=0.0)


def test_time_decay_reduces_tsdf_rgb_and_feature_weights(warp_init, device):
    feature_vector = torch.tensor([0.2, -0.1, 0.8], dtype=torch.float32, device=device)
    feature_grid = _constant_feature_grid(device, feature_vector)
    mapper = _make_mapper(
        device,
        feature_dim=feature_vector.numel(),
        decay_factor=0.5,
        frustum_decay_factor=1.0,
    )
    mapper.integrate(
        _constant_plane_observation(device, rgb_value=(64, 128, 192), feature_grid=feature_grid)
    )
    n = int(mapper.tsdf.data.num_allocated.item())
    before_tsdf = mapper.tsdf.data.block_data[:n, :, 1].float().sum()
    before_rgb = mapper.tsdf.data.block_grid_rgb[:n, :, 3].float().sum()
    before_feature = mapper.tsdf.data.block_feature_weight[:n].float().sum()

    empty_depth = torch.zeros((IMAGE_H, IMAGE_W), dtype=torch.float32, device=device)
    empty_rgb = torch.zeros((IMAGE_H, IMAGE_W, 3), dtype=torch.uint8, device=device)
    mapper.integrate(
        _observation(
            device=device,
            depth=empty_depth,
            rgb=empty_rgb,
            feature_grid=feature_grid,
        )
    )
    after_tsdf = mapper.tsdf.data.block_data[:n, :, 1].float().sum()
    after_rgb = mapper.tsdf.data.block_grid_rgb[:n, :, 3].float().sum()
    after_feature = mapper.tsdf.data.block_feature_weight[:n].float().sum()

    assert before_tsdf > 0
    expected_decay = torch.tensor(0.5, dtype=torch.float32, device=device)
    torch.testing.assert_close(
        after_tsdf / before_tsdf,
        expected_decay,
        atol=0.08,
        rtol=0.0,
    )
    torch.testing.assert_close(
        after_rgb / before_rgb,
        expected_decay,
        atol=0.08,
        rtol=0.0,
    )
    torch.testing.assert_close(
        after_feature / before_feature,
        expected_decay,
        atol=0.08,
        rtol=0.0,
    )


def test_clear_region_removes_stale_rgb_and_features(warp_init, device):
    feature_a = torch.tensor([1.0, 0.0, -0.5], dtype=torch.float32, device=device)
    feature_b = torch.tensor([-0.5, 0.5, 1.0], dtype=torch.float32, device=device)
    mapper = _make_mapper(device, feature_dim=feature_a.numel())
    mapper.integrate(
        _constant_plane_observation(
            device,
            rgb_value=(255, 0, 0),
            feature_grid=_constant_feature_grid(device, feature_a),
        )
    )

    n_cleared = mapper.clear_regions(
        torch.tensor([-1.0, -1.0, 0.5], dtype=torch.float32, device=device),
        torch.tensor([1.0, 1.0, 1.5], dtype=torch.float32, device=device),
    )
    assert n_cleared > 0
    n = int(mapper.tsdf.data.num_allocated.item())
    assert mapper.tsdf.data.block_grid_rgb[:n, :, 3].float().sum() == 0
    assert mapper.tsdf.data.block_feature_weight[:n].float().sum() == 0

    mapper.integrate(
        _constant_plane_observation(
            device,
            rgb_value=(0, 0, 255),
            feature_grid=_constant_feature_grid(device, feature_b),
        )
    )
    _, rgb_normalized, _ = _active_rgb(mapper)
    _, feature_normalized, _ = _active_features(mapper)
    expected_blue = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    torch.testing.assert_close(
        rgb_normalized,
        expected_blue.view(1, 3).expand_as(rgb_normalized),
        atol=0.03,
        rtol=0.0,
    )
    torch.testing.assert_close(
        feature_normalized,
        feature_b.view(1, -1).expand_as(feature_normalized),
        atol=0.03,
        rtol=0.0,
    )


def test_stats_report_last_integration_and_kernel_timings(warp_init, device):
    mapper = _make_mapper(device, profile_kernel_timings=True)
    mapper.integrate(_constant_plane_observation(device))
    stats = mapper.get_stats(scan_pool=False)

    assert stats["frame_count"] == 1
    assert stats["last_integration"]["num_visible_blocks"] > 0
    assert stats["last_integration"]["support_overflow_count"] >= 0
    assert stats["last_integration"]["profile_kernel_timings"] is True
    timings = stats["last_integration_kernel_timings_ms"]
    assert timings
    assert all(isinstance(value, float) and value >= 0.0 for value in timings.values())

    empty_depth = torch.zeros((IMAGE_H, IMAGE_W), dtype=torch.float32, device=device)
    empty_rgb = torch.zeros((IMAGE_H, IMAGE_W, 3), dtype=torch.uint8, device=device)
    mapper.integrate(_observation(device=device, depth=empty_depth, rgb=empty_rgb))
    stats = mapper.get_stats(scan_pool=False)
    assert stats["frame_count"] == 2
    assert stats["last_integration"]["num_visible_blocks"] == 0


def test_support_capacity_overflow_keeps_rgb_grid_and_support_features(warp_init, device):
    feature_vector = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32, device=device)
    feature_grid = _constant_feature_grid(device, feature_vector)
    mapper = _make_mapper(
        device,
        feature_dim=feature_vector.numel(),
        support_capacity=1,
        feature_integration_kernel="grouped",
    )
    obs = _constant_plane_observation(
        device,
        rgb_value=(32, 160, 224),
        feature_grid=feature_grid,
    )
    mapper.integrate(obs)

    stats = mapper.get_stats(scan_pool=False)
    assert stats["last_integration"]["support_overflow_count"] > 0

    rgb_pool_idx, rgb_normalized, _ = _active_rgb(mapper)
    expected_rgb = torch.tensor(
        [32.0 / 255.0, 160.0 / 255.0, 224.0 / 255.0],
        dtype=torch.float32,
        device=device,
    )
    torch.testing.assert_close(
        rgb_normalized,
        expected_rgb.view(1, 3).expand_as(rgb_normalized),
        atol=0.03,
        rtol=0.0,
    )

    expected_feature_sum, expected_feature_weight = _expected_features_from_support(
        mapper,
        feature_grid,
    )
    feature_pool_idx, feature_normalized, _ = _active_features(mapper)
    torch.testing.assert_close(
        feature_normalized,
        expected_feature_sum[feature_pool_idx]
        / expected_feature_weight[feature_pool_idx].view(-1, 1),
        atol=0.03,
        rtol=0.0,
    )


def test_two_camera_mapper_averages_rgb_and_features(warp_init, device):
    feature_dim = 3
    feature_grid = torch.empty((2, 7, 11, feature_dim), dtype=torch.float16, device=device)
    feature_a = torch.tensor([1.0, 0.0, -0.5], dtype=torch.float16, device=device)
    feature_b = torch.tensor([-1.0, 0.5, 0.5], dtype=torch.float16, device=device)
    feature_grid[0] = feature_a.view(1, 1, -1)
    feature_grid[1] = feature_b.view(1, 1, -1)

    depth = torch.full((2, IMAGE_H, IMAGE_W), PLANE_Z, dtype=torch.float32, device=device)
    rgb = torch.zeros((2, IMAGE_H, IMAGE_W, 3), dtype=torch.uint8, device=device)
    rgb[0, ..., 0] = 255
    rgb[1, ..., 2] = 255
    obs = _observation(
        device=device,
        depth=depth,
        rgb=rgb,
        feature_grid=feature_grid,
        pose=_identity_pose(device, 2),
        intrinsics=_intrinsics(device).view(1, 3, 3).expand(2, 3, 3).contiguous(),
    )
    mapper = _make_mapper(
        device,
        num_cameras=2,
        feature_dim=feature_dim,
        feature_integration_kernel="tiled",
    )

    mapper.integrate(obs)
    _, rgb_normalized, _ = _active_rgb(mapper)
    _, feature_normalized, _ = _active_features(mapper)

    expected_rgb = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float32, device=device)
    expected_feature = ((feature_a.float() + feature_b.float()) * 0.5).view(1, -1)
    torch.testing.assert_close(
        rgb_normalized,
        expected_rgb.view(1, 3).expand_as(rgb_normalized),
        atol=0.03,
        rtol=0.0,
    )
    torch.testing.assert_close(
        feature_normalized,
        expected_feature.expand_as(feature_normalized),
        atol=0.03,
        rtol=0.0,
    )


def _bumpy_scene_mapper(device: str) -> Mapper:
    """Mapper holding a plane with raised patches, so blocks are allocated at several depths."""
    depth = torch.full((IMAGE_H, IMAGE_W), PLANE_Z, dtype=torch.float32, device=device)
    depth[4:12, 5:15] = PLANE_Z - 0.15
    depth[18:28, 22:36] = PLANE_Z - 0.3
    depth[2:6, 30:38] = PLANE_Z + 0.1
    rgb = torch.full((IMAGE_H, IMAGE_W, 3), 120, dtype=torch.uint8, device=device)
    mapper = _make_mapper(device)
    mapper.integrate(_observation(device=device, depth=depth, rgb=rgb))
    return mapper


def _tsdf_tensors(mapper: Mapper) -> dict[str, torch.Tensor]:
    """Every tensor of the TSDF storage, cloned."""
    return {
        name: value.clone()
        for name, value in vars(mapper.tsdf.data).items()
        if isinstance(value, torch.Tensor)
    }


def _restore_tsdf_tensors(mapper: Mapper, saved: dict[str, torch.Tensor]) -> None:
    for name, value in saved.items():
        getattr(mapper.tsdf.data, name).copy_(value)


def _reference_aabb_to_block_bounds(tsdf, bounds_min, bounds_max) -> tuple:
    """Frozen copy of the former single-box ``_world_aabb_to_block_bounds`` (host, float64).

    Kept as the reference the batched on-device conversion must reproduce exactly.
    """
    lo_in = (
        torch.as_tensor(
            bounds_min,
            dtype=torch.float32,
        )
        .flatten()
        .detach()
        .cpu()
    )
    hi_in = (
        torch.as_tensor(
            bounds_max,
            dtype=torch.float32,
        )
        .flatten()
        .detach()
        .cpu()
    )
    if lo_in.numel() != 3 or hi_in.numel() != 3:
        log_and_raise(
            "clear_region bounds must each contain 3 values, got "
            f"bounds_min={tuple(lo_in.shape)}, bounds_max={tuple(hi_in.shape)}."
        )

    lo = torch.minimum(lo_in, hi_in)
    hi = torch.maximum(lo_in, hi_in)
    if not torch.isfinite(lo).all() or not torch.isfinite(hi).all():
        log_and_raise("clear_region bounds must be finite.")

    origin = tsdf.data.origin.detach().to(device="cpu", dtype=torch.float32).flatten()
    voxel_size = float(tsdf.config.voxel_size)
    block_size = int(tsdf.block_size)

    grid_D, grid_H, grid_W = (int(v) for v in tsdf.config.grid_shape)

    center_offset = (
        torch.tensor(
            [grid_W, grid_H, grid_D],
            dtype=torch.float32,
        )
        * 0.5
    )
    v_lo = (lo - origin) / voxel_size + center_offset
    v_hi = (hi - origin) / voxel_size + center_offset

    # Include blocks touching exact AABB boundaries. This can over-clear
    # one adjacent block on boundary-aligned regions, but avoids misses.
    eps_voxels = 1.0e-6
    min_bx = math.floor((float(v_lo[0]) - eps_voxels) / block_size)
    min_by = math.floor((float(v_lo[1]) - eps_voxels) / block_size)
    min_bz = math.floor((float(v_lo[2]) - eps_voxels) / block_size)
    max_bx = math.floor((float(v_hi[0]) + eps_voxels) / block_size)
    max_by = math.floor((float(v_hi[1]) + eps_voxels) / block_size)
    max_bz = math.floor((float(v_hi[2]) + eps_voxels) / block_size)

    max_grid_bx = math.ceil(grid_W / block_size) - 1
    max_grid_by = math.ceil(grid_H / block_size) - 1
    max_grid_bz = math.ceil(grid_D / block_size) - 1
    if max_grid_bx < 0 or max_grid_by < 0 or max_grid_bz < 0:
        return 0, 0, 0, 0, 0, 0, grid_W, grid_H, grid_D
    if (
        max_bx < 0
        or max_by < 0
        or max_bz < 0
        or min_bx > max_grid_bx
        or min_by > max_grid_by
        or min_bz > max_grid_bz
    ):
        return 0, 0, 0, 0, 0, 0, grid_W, grid_H, grid_D
    min_bx = max(min_bx, 0)
    min_by = max(min_by, 0)
    min_bz = max(min_bz, 0)
    max_bx = min(max_bx, max_grid_bx)
    max_by = min(max_by, max_grid_by)
    max_bz = min(max_bz, max_grid_bz)

    offset_x = (max_grid_bx + 1) // 2
    offset_y = (max_grid_by + 1) // 2
    offset_z = (max_grid_bz + 1) // 2
    min_bx -= offset_x
    max_bx -= offset_x
    min_by -= offset_y
    max_by -= offset_y
    min_bz -= offset_z
    max_bz -= offset_z

    count_x = max_bx - min_bx + 1
    count_y = max_by - min_by + 1
    count_z = max_bz - min_bz + 1
    if count_x <= 0 or count_y <= 0 or count_z <= 0:
        return 0, 0, 0, 0, 0, 0, grid_W, grid_H, grid_D

    return (
        min_bx,
        min_by,
        min_bz,
        count_x,
        count_y,
        count_z,
        grid_W,
        grid_H,
        grid_D,
    )


def _reference_cleared_pools(tsdf, bounds_min: torch.Tensor, bounds_max: torch.Tensor) -> set[int]:
    """Pools of the allocated blocks each box intersects, from the stored block keys.

    Independent of the hash table and of the collect kernel: a block is cleared when its
    key lies in the block range the frozen single-box conversion gives the box.
    """
    data = tsdf.data
    n_alloc = int(data.num_allocated.item())
    keys = data.block_coords[: n_alloc * 3].view(n_alloc, 3).long().cpu()
    allocated = data.block_to_hash_slot[:n_alloc].cpu() >= 0
    pools = set()
    for i in range(bounds_min.shape[0]):
        bx, by, bz, cx, cy, cz, *_ = _reference_aabb_to_block_bounds(
            tsdf, bounds_min[i], bounds_max[i]
        )
        if cx <= 0 or cy <= 0 or cz <= 0:
            continue
        lo = torch.tensor([bx, by, bz])
        hi = lo + torch.tensor([cx, cy, cz])
        inside = ((keys >= lo) & (keys < hi)).all(dim=1) & allocated
        pools.update(torch.nonzero(inside).flatten().tolist())
    return pools


def _clear_test_boxes(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sphere-like boxes: inside the map, overlapping, straddling and outside the grid,
    corner-swapped, and boxes whose faces sit exactly on block boundaries."""
    gen = torch.Generator(device="cpu").manual_seed(7)
    n = 300
    centers = torch.rand((n, 3), generator=gen) * torch.tensor([1.4, 1.2, 1.2]) - torch.tensor(
        [0.7, 0.6, 0.6 - PLANE_Z]
    )
    radii = torch.rand((n, 1), generator=gen) * 0.08 + 0.005
    bounds_min = centers - radii
    bounds_max = centers + radii
    # corner-swapped boxes must be handled like their sorted version
    bounds_min[::7], bounds_max[::7] = bounds_max[::7].clone(), bounds_min[::7].clone()
    # faces exactly on block boundaries (block edge = block_size * voxel_size = 0.04 m)
    edge = 2 * VOXEL_SIZE
    k = torch.arange(10, dtype=torch.float32)
    aligned_min = torch.stack([k * edge - 0.2, k * 0.0 - 0.1, k * 0.0 + PLANE_Z - 0.1], dim=1)
    aligned_max = aligned_min + edge
    bounds_min = torch.cat([bounds_min, aligned_min])
    bounds_max = torch.cat([bounds_max, aligned_max])
    return bounds_min.to(device), bounds_max.to(device)


def test_clear_regions_block_bounds_match_reference(warp_init, device):
    """The on-device box -> block conversion matches the frozen host conversion exactly."""
    mapper = _bumpy_scene_mapper(device)
    tsdf = mapper.tsdf
    camera = mapper._integrator._tsdf_integrator._camera_integrator
    bounds_min, bounds_max = _clear_test_boxes(device)

    min_block, block_count = camera._world_aabbs_to_block_bounds(tsdf, bounds_min, bounds_max)
    n_non_empty = 0
    for i in range(bounds_min.shape[0]):
        ref = _reference_aabb_to_block_bounds(tsdf, bounds_min[i], bounds_max[i])
        batched_count = block_count[i].tolist()
        if ref[3] <= 0 or ref[4] <= 0 or ref[5] <= 0:
            assert min(batched_count) <= 0, i
            continue
        n_non_empty += 1
        assert min_block[i].tolist() == list(ref[0:3]), i
        assert batched_count == list(ref[3:6]), i
    assert n_non_empty > 100


def test_clear_regions_matches_reference_bitwise(warp_init, device):
    """clear_regions leaves the map bit-identical to clearing the reference block set."""
    mapper = _bumpy_scene_mapper(device)
    camera = mapper._integrator._tsdf_integrator._camera_integrator
    bounds_min, bounds_max = _clear_test_boxes(device)
    before = _tsdf_tensors(mapper)

    expected_pools = _reference_cleared_pools(mapper.tsdf, bounds_min, bounds_max)
    camera.clear_blocks(
        mapper.tsdf, torch.tensor(sorted(expected_pools), dtype=torch.int32, device=device)
    )
    mapper._integrator._site_index.fill_(-1)
    mapper._integrator._dist_field.zero_()
    expected_tsdf = _tsdf_tensors(mapper)
    expected_esdf = mapper.compute_esdf().feature_tensor.clone()

    _restore_tsdf_tensors(mapper, before)
    n_cleared = mapper.clear_regions(bounds_min, bounds_max)
    actual_tsdf = _tsdf_tensors(mapper)
    actual_esdf = mapper.compute_esdf().feature_tensor.clone()

    assert len(expected_pools) > 0
    assert n_cleared == len(expected_pools)
    assert set(camera.clear_pool_indices[:n_cleared].tolist()) == expected_pools
    for name, value in expected_tsdf.items():
        assert torch.equal(actual_tsdf[name], value), name
    assert torch.equal(actual_esdf, expected_esdf)
    # the reference really cleared something, so the comparison is not vacuous
    assert not torch.equal(before["block_data"], expected_tsdf["block_data"])


def test_clear_regions_repeated_calls_stay_exact(warp_init, device):
    """The dedup marks of one call never hide blocks from the next call."""
    mapper = _bumpy_scene_mapper(device)
    bounds_min, bounds_max = _clear_test_boxes(device)
    before = _tsdf_tensors(mapper)
    half = bounds_min.shape[0] // 2

    mapper.clear_regions(bounds_min, bounds_max)
    expected = _tsdf_tensors(mapper)

    _restore_tsdf_tensors(mapper, before)
    mapper.clear_regions(bounds_min[:half], bounds_max[:half])
    mapper.clear_regions(bounds_min[half:], bounds_max[half:])
    mapper.clear_regions(bounds_min, bounds_max)
    actual = _tsdf_tensors(mapper)
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def test_clear_region_alias_matches_clear_regions(warp_init, device):
    """The deprecated single-box clear_region clears exactly what clear_regions does."""
    mapper = _bumpy_scene_mapper(device)
    bounds_min, bounds_max = _clear_test_boxes(device)
    before = _tsdf_tensors(mapper)
    for i in range(bounds_min.shape[0]):
        mapper.clear_regions(bounds_min[i], bounds_max[i])
    expected = _tsdf_tensors(mapper)

    _restore_tsdf_tensors(mapper, before)
    for i in range(bounds_min.shape[0]):
        mapper.clear_region(bounds_min[i], bounds_max[i])
    actual = _tsdf_tensors(mapper)
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def test_clear_regions_empty_and_invalid_input(warp_init, device):
    mapper = _bumpy_scene_mapper(device)
    empty = torch.zeros((0, 3), dtype=torch.float32, device=device)
    assert mapper.clear_regions(empty, empty) == 0

    far = torch.tensor([[50.0, 50.0, 50.0]], dtype=torch.float32, device=device)
    assert mapper.clear_regions(far, far + 0.1) == 0

    bad = torch.tensor([[float("nan"), 0.0, 1.0]], dtype=torch.float32, device=device)
    with pytest.raises(Exception):
        mapper.clear_regions(bad, bad + 0.1)
