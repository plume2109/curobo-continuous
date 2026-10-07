# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""LiDAR range-image TSDF/RGB/feature integration kernels."""

from __future__ import annotations

import warp as wp

from curobo._src.perception.mapper.kernel.wp_integrate_common import (
    compute_tsdf_weight,
    floor_div,
)
from curobo._src.util.warp import warp_constant_suffix, warp_kernel


def make_lidar_integrate_kernels(
    block_size: int,
    *,
    feature_dim: int,
    lidar_num_sensors: int,
    lidar_image_height: int,
    lidar_image_width: int,
    num_samples: int,
    grid_shape: tuple[int, int, int],
    voxel_size: float,
    truncation_distance: float,
    lidar_feature_grid_shape: tuple[int, int] | None,
    feature_channels_per_thread: int,
    max_feature_tile_channels: int,
    max_support_pixels_per_block_lidar: int,
    color_grid_size: int,
    feature_block_grid_size: int,
    pack_key_only,
    unpack_block_key,
    hash_lookup,
    world_to_continuous_voxel,
    block_local_to_world,
    block_grid_to_key_coords,
    block_key_to_voxel_base,
) -> dict[str, object]:
    """Build LiDAR range-image integration kernels."""
    BLOCK_SIZE = wp.constant(block_size)
    NUM_LIDARS = wp.constant(wp.int32(lidar_num_sensors))
    LIDAR_IMAGE_HEIGHT = wp.constant(wp.int32(lidar_image_height))
    LIDAR_IMAGE_WIDTH = wp.constant(wp.int32(lidar_image_width))
    NUM_SAMPLES = wp.constant(wp.int32(num_samples))
    GRID_D = wp.constant(wp.int32(grid_shape[0]))
    GRID_H = wp.constant(wp.int32(grid_shape[1]))
    GRID_W = wp.constant(wp.int32(grid_shape[2]))
    VOXEL_SIZE = wp.constant(wp.float32(voxel_size))
    TRUNCATION_DIST = wp.constant(wp.float32(truncation_distance))
    PI = wp.constant(wp.float32(3.141592653589793))
    TWO_PI = wp.constant(wp.float32(6.283185307179586))
    safe_step = (float(block_size) * float(voxel_size)) / 1.42
    STEP_SIZE = wp.constant(wp.float32(safe_step))
    FEATURE_DIM = wp.constant(wp.int32(feature_dim))
    color_grid_voxels = int(color_grid_size) ** 3
    COLOR_GRID_SIZE = wp.constant(wp.int32(color_grid_size))
    COLOR_GRID_VOXELS = wp.constant(wp.int32(color_grid_voxels))
    feature_grid_voxels = int(feature_block_grid_size) ** 3
    FEATURE_BLOCK_GRID_SIZE = wp.constant(wp.int32(feature_block_grid_size))
    FEATURE_GRID_VOXELS = wp.constant(wp.int32(feature_grid_voxels))
    if lidar_feature_grid_shape is None:
        lidar_feature_grid_height = 1
        lidar_feature_grid_width = 1
    else:
        lidar_feature_grid_height = int(lidar_feature_grid_shape[0])
        lidar_feature_grid_width = int(lidar_feature_grid_shape[1])
    LIDAR_FEATURE_GRID_HEIGHT = wp.constant(wp.int32(lidar_feature_grid_height))
    LIDAR_FEATURE_GRID_WIDTH = wp.constant(wp.int32(lidar_feature_grid_width))
    FEATURE_CHANNELS_PER_THREAD = wp.constant(feature_channels_per_thread)
    feature_tile_channels = max(1, min(int(feature_dim), int(max_feature_tile_channels)))
    FEATURE_TILE_CHANNELS = wp.constant(feature_tile_channels)
    support_capacity = int(max_support_pixels_per_block_lidar)
    SUPPORT_CAPACITY = wp.constant(support_capacity)
    suffix_hash = warp_constant_suffix(
        block_size,
        feature_dim,
        lidar_num_sensors,
        lidar_image_height,
        lidar_image_width,
        num_samples,
        grid_shape,
        voxel_size,
        truncation_distance,
        lidar_feature_grid_shape,
        feature_channels_per_thread,
        max_feature_tile_channels,
        max_support_pixels_per_block_lidar,
        color_grid_size,
        feature_block_grid_size,
    )
    suffix = f"bs{block_size}_cfg{suffix_hash}"

    @wp.func
    def _lidar_pixel_ray(
        pixel_idx: wp.int32,
        min_elev: wp.float32,
        max_elev: wp.float32,
    ) -> wp.vec3:
        px = pixel_idx % LIDAR_IMAGE_WIDTH
        py = pixel_idx // LIDAR_IMAGE_WIDTH
        azimuth = wp.float32(px) * TWO_PI / wp.float32(LIDAR_IMAGE_WIDTH) - PI
        elevation = min_elev
        if LIDAR_IMAGE_HEIGHT > wp.int32(1):
            elevation = max_elev - (
                wp.float32(py) * (max_elev - min_elev) / wp.float32(LIDAR_IMAGE_HEIGHT - wp.int32(1))
            )
        cos_elev = wp.cos(elevation)
        return wp.vec3(
            wp.cos(azimuth) * cos_elev,
            wp.sin(azimuth) * cos_elev,
            wp.sin(elevation),
        )

    @wp.func
    def _lidar_uv_to_ray(
        u_px: wp.int32,
        v_px: wp.int32,
        min_elev: wp.float32,
        max_elev: wp.float32,
    ) -> wp.vec3:
        pixel_idx = v_px * LIDAR_IMAGE_WIDTH + u_px
        return _lidar_pixel_ray(pixel_idx, min_elev, max_elev)

    @wp.func
    def _range_valid(
        value: wp.float32,
        min_range: wp.float32,
        max_range: wp.float32,
    ) -> bool:
        return value >= min_range and value <= max_range

    @wp.func
    def _lidar_feature_grid_coords(px: wp.int32, py: wp.int32) -> wp.vec2i:
        gx = (px * LIDAR_FEATURE_GRID_WIDTH) // LIDAR_IMAGE_WIDTH
        gy = (py * LIDAR_FEATURE_GRID_HEIGHT) // LIDAR_IMAGE_HEIGHT
        if gx < wp.int32(0):
            gx = wp.int32(0)
        if gy < wp.int32(0):
            gy = wp.int32(0)
        if gx >= LIDAR_FEATURE_GRID_WIDTH:
            gx = LIDAR_FEATURE_GRID_WIDTH - wp.int32(1)
        if gy >= LIDAR_FEATURE_GRID_HEIGHT:
            gy = LIDAR_FEATURE_GRID_HEIGHT - wp.int32(1)
        return wp.vec2i(gx, gy)

    @warp_kernel(f"lidar_compute_block_keys_only_kernel_{suffix}")
    def lidar_compute_block_keys_only_kernel(
        lidar_positions: wp.array2d(dtype=wp.float32),
        lidar_quaternions: wp.array2d(dtype=wp.float32),
        range_images: wp.array3d(dtype=wp.float32),
        valid_range_m: wp.array2d(dtype=wp.float32),
        elevation_range_rad: wp.array2d(dtype=wp.float32),
        block_keys: wp.array(dtype=wp.int64),
        origin: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        n_pixels = LIDAR_IMAGE_HEIGHT * LIDAR_IMAGE_WIDTH
        samples_per_lidar = n_pixels * NUM_SAMPLES
        lidar_idx = tid // samples_per_lidar
        remainder = tid % samples_per_lidar
        pixel_idx = remainder // NUM_SAMPLES
        sample_idx = remainder % NUM_SAMPLES

        if lidar_idx >= NUM_LIDARS or pixel_idx >= n_pixels:
            block_keys[tid] = wp.int64(-1)
            return

        px = pixel_idx % LIDAR_IMAGE_WIDTH
        py = pixel_idx // LIDAR_IMAGE_WIDTH
        min_range = valid_range_m[lidar_idx, 0]
        max_range = valid_range_m[lidar_idx, 1]
        range_m = range_images[lidar_idx, py, px]
        if not _range_valid(range_m, min_range, max_range):
            block_keys[tid] = wp.int64(-1)
            return

        min_elev = elevation_range_rad[lidar_idx, 0]
        max_elev = elevation_range_rad[lidar_idx, 1]
        ray_dir = _lidar_pixel_ray(pixel_idx, min_elev, max_elev)

        r_start = wp.max(range_m - TRUNCATION_DIST, min_range)
        r_sample = r_start + wp.float32(sample_idx) * STEP_SIZE
        if r_sample > range_m + TRUNCATION_DIST + STEP_SIZE:
            block_keys[tid] = wp.int64(-1)
            return

        lidar_pos = wp.vec3(
            lidar_positions[lidar_idx, 0],
            lidar_positions[lidar_idx, 1],
            lidar_positions[lidar_idx, 2],
        )
        lidar_quat = wp.quaternion(
            lidar_quaternions[lidar_idx, 1],
            lidar_quaternions[lidar_idx, 2],
            lidar_quaternions[lidar_idx, 3],
            lidar_quaternions[lidar_idx, 0],
        )
        point_world = lidar_pos + wp.quat_rotate(lidar_quat, ray_dir * r_sample)
        voxel_f = world_to_continuous_voxel(point_world, origin[0])

        vx = wp.int32(wp.floor(voxel_f[0]))
        vy = wp.int32(wp.floor(voxel_f[1]))
        vz = wp.int32(wp.floor(voxel_f[2]))

        if vx < 0 or vx >= GRID_W or vy < 0 or vy >= GRID_H or vz < 0 or vz >= GRID_D:
            block_keys[tid] = wp.int64(-1)
            return

        bx_grid = floor_div(vx, BLOCK_SIZE)
        by_grid = floor_div(vy, BLOCK_SIZE)
        bz_grid = floor_div(vz, BLOCK_SIZE)
        key = block_grid_to_key_coords(bx_grid, by_grid, bz_grid)
        block_keys[tid] = pack_key_only(key[0], key[1], key[2])

    @warp_kernel(
        f"lidar_build_support_pixels_from_keys_kernel_{suffix}_sc{support_capacity}"
    )
    def lidar_build_support_pixels_from_keys_kernel(
        block_keys: wp.array(dtype=wp.int64),
        n_keys: wp.int32,
        hash_table: wp.array(dtype=wp.int64),
        hash_capacity: wp.int32,
        max_blocks: wp.int32,
        visible_epoch: wp.array(dtype=wp.int32),
        frame_epoch: wp.int32,
        pool_to_visible_slot: wp.array(dtype=wp.int32),
        visible_capacity: wp.int32,
        support_counts: wp.array2d(dtype=wp.int32),
        support_pixels: wp.array3d(dtype=wp.int32),
        support_overflow_count: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        if tid >= n_keys:
            return

        key = block_keys[tid]
        if key == wp.int64(-1):
            return

        n_pixels = LIDAR_IMAGE_HEIGHT * LIDAR_IMAGE_WIDTH
        samples_per_lidar = n_pixels * NUM_SAMPLES
        lidar_idx = tid // samples_per_lidar
        remainder = tid % samples_per_lidar
        pixel_idx = remainder // NUM_SAMPLES
        sample_idx = remainder % NUM_SAMPLES

        if lidar_idx >= NUM_LIDARS or pixel_idx >= n_pixels:
            return
        if sample_idx > wp.int32(0):
            prev_key = block_keys[tid - wp.int32(1)]
            if prev_key == key:
                return

        coords = unpack_block_key(key)
        pool_idx = hash_lookup(
            hash_table,
            coords[0],
            coords[1],
            coords[2],
            hash_capacity,
        )
        if pool_idx < wp.int32(0) or pool_idx >= max_blocks:
            return
        if visible_epoch[pool_idx] != frame_epoch:
            return

        vis_idx = pool_to_visible_slot[pool_idx]
        if vis_idx < wp.int32(0) or vis_idx >= visible_capacity:
            return

        slot = wp.atomic_add(support_counts, vis_idx, lidar_idx, wp.int32(1))
        if slot < SUPPORT_CAPACITY:
            support_pixels[vis_idx, lidar_idx, slot] = pixel_idx
        else:
            wp.atomic_add(support_overflow_count, 0, wp.int32(1))

    @wp.func
    def _nearest_lidar_range(
        lidar_idx: wp.int32,
        u_float: wp.float32,
        v_float: wp.float32,
        voxel_lidar: wp.vec3,
        min_range: wp.float32,
        max_range: wp.float32,
        min_elev: wp.float32,
        max_elev: wp.float32,
        nearest_max_dist_to_ray_m: wp.float32,
        range_images: wp.array3d(dtype=wp.float32),
    ) -> wp.vec2:
        u_near = wp.int32(wp.floor(u_float + wp.float32(0.5)))
        if u_near >= LIDAR_IMAGE_WIDTH:
            u_near = u_near - LIDAR_IMAGE_WIDTH
        if u_near < wp.int32(0):
            u_near = u_near + LIDAR_IMAGE_WIDTH
        v_near = wp.int32(0)
        if LIDAR_IMAGE_HEIGHT > wp.int32(1):
            v_near = wp.int32(wp.floor(v_float + wp.float32(0.5)))
            if v_near < wp.int32(0) or v_near >= LIDAR_IMAGE_HEIGHT:
                return wp.vec2(0.0, 0.0)

        surface_range = range_images[lidar_idx, v_near, u_near]
        if not _range_valid(surface_range, min_range, max_range):
            return wp.vec2(0.0, 0.0)

        ray = _lidar_uv_to_ray(u_near, v_near, min_elev, max_elev)
        proj = wp.dot(voxel_lidar, ray)
        closest = ray * proj
        diff = voxel_lidar - closest
        dist_to_ray = wp.sqrt(wp.dot(diff, diff))
        if dist_to_ray > nearest_max_dist_to_ray_m:
            return wp.vec2(0.0, 0.0)
        return wp.vec2(surface_range, 1.0)

    @wp.func
    def _interpolate_lidar_range(
        lidar_idx: wp.int32,
        u_float: wp.float32,
        v_float: wp.float32,
        voxel_lidar: wp.vec3,
        min_range: wp.float32,
        max_range: wp.float32,
        min_elev: wp.float32,
        max_elev: wp.float32,
        linear_max_diff_m: wp.float32,
        nearest_max_dist_to_ray_m: wp.float32,
        range_images: wp.array3d(dtype=wp.float32),
    ) -> wp.vec2:
        if LIDAR_IMAGE_HEIGHT == wp.int32(1):
            return _nearest_lidar_range(
                lidar_idx,
                u_float,
                wp.float32(0.0),
                voxel_lidar,
                min_range,
                max_range,
                min_elev,
                max_elev,
                nearest_max_dist_to_ray_m,
                range_images,
            )

        u0 = wp.int32(wp.floor(u_float))
        v0 = wp.int32(wp.floor(v_float))
        if v0 < wp.int32(0) or v0 >= LIDAR_IMAGE_HEIGHT - wp.int32(1):
            return _nearest_lidar_range(
                lidar_idx,
                u_float,
                v_float,
                voxel_lidar,
                min_range,
                max_range,
                min_elev,
                max_elev,
                nearest_max_dist_to_ray_m,
                range_images,
            )
        if u0 < wp.int32(0):
            u0 = u0 + LIDAR_IMAGE_WIDTH
        if u0 >= LIDAR_IMAGE_WIDTH:
            u0 = u0 - LIDAR_IMAGE_WIDTH
        u1 = u0 + wp.int32(1)
        if u1 >= LIDAR_IMAGE_WIDTH:
            u1 = wp.int32(0)
        v1 = v0 + wp.int32(1)

        d00 = range_images[lidar_idx, v0, u0]
        d01 = range_images[lidar_idx, v0, u1]
        d10 = range_images[lidar_idx, v1, u0]
        d11 = range_images[lidar_idx, v1, u1]
        valid = (
            _range_valid(d00, min_range, max_range)
            and _range_valid(d01, min_range, max_range)
            and _range_valid(d10, min_range, max_range)
            and _range_valid(d11, min_range, max_range)
        )
        if valid:
            fu = u_float - wp.floor(u_float)
            fv = v_float - wp.floor(v_float)
            top = d00 * (wp.float32(1.0) - fu) + d01 * fu
            bottom = d10 * (wp.float32(1.0) - fu) + d11 * fu
            interp = top * (wp.float32(1.0) - fv) + bottom * fv
            max_diff = wp.max(
                wp.max(wp.abs(d00 - interp), wp.abs(d01 - interp)),
                wp.max(wp.abs(d10 - interp), wp.abs(d11 - interp)),
            )
            if max_diff <= linear_max_diff_m:
                return wp.vec2(interp, 1.0)

        return _nearest_lidar_range(
            lidar_idx,
            u_float,
            v_float,
            voxel_lidar,
            min_range,
            max_range,
            min_elev,
            max_elev,
            nearest_max_dist_to_ray_m,
            range_images,
        )

    @warp_kernel(f"lidar_integrate_voxels_kernel_{suffix}")
    def lidar_integrate_voxels_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        lidar_positions: wp.array2d(dtype=wp.float32),
        lidar_quaternions: wp.array2d(dtype=wp.float32),
        range_images: wp.array3d(dtype=wp.float32),
        valid_range_m: wp.array2d(dtype=wp.float32),
        elevation_range_rad: wp.array2d(dtype=wp.float32),
        linear_interpolation_max_allowable_difference_m: float,
        nearest_interpolation_max_allowable_dist_to_ray_m: float,
        block_coords: wp.array(dtype=wp.int32),
        block_data: wp.array3d(dtype=wp.float16),
        origin: wp.array(dtype=wp.vec3),
    ):
        vis_idx, local_idx = wp.tid()
        if vis_idx >= n_visible:
            return

        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < 0:
            return

        bx = block_coords[pool_idx * 3 + 0]
        by = block_coords[pool_idx * 3 + 1]
        bz = block_coords[pool_idx * 3 + 2]
        voxel_center = block_local_to_world(bx, by, bz, local_idx, origin[0])

        total_sw = wp.float32(0.0)
        total_w = wp.float32(0.0)

        for lidar_i in range(lidar_num_sensors):
            lidar_pos = wp.vec3(
                lidar_positions[lidar_i, 0],
                lidar_positions[lidar_i, 1],
                lidar_positions[lidar_i, 2],
            )
            lidar_quat = wp.quaternion(
                lidar_quaternions[lidar_i, 1],
                lidar_quaternions[lidar_i, 2],
                lidar_quaternions[lidar_i, 3],
                lidar_quaternions[lidar_i, 0],
            )
            voxel_lidar = wp.quat_rotate(wp.quat_inverse(lidar_quat), voxel_center - lidar_pos)
            voxel_range = wp.sqrt(wp.dot(voxel_lidar, voxel_lidar))

            min_range = valid_range_m[lidar_i, 0]
            max_range = valid_range_m[lidar_i, 1]
            if not _range_valid(voxel_range, min_range, max_range):
                continue

            min_elev = elevation_range_rad[lidar_i, 0]
            max_elev = elevation_range_rad[lidar_i, 1]
            xy_norm = wp.sqrt(voxel_lidar[0] * voxel_lidar[0] + voxel_lidar[1] * voxel_lidar[1])
            elevation = wp.atan2(voxel_lidar[2], xy_norm)
            if LIDAR_IMAGE_HEIGHT == wp.int32(1):
                v_float = wp.float32(0.0)
            else:
                if elevation < min_elev or elevation > max_elev:
                    continue
                v_float = (max_elev - elevation) * (
                    wp.float32(LIDAR_IMAGE_HEIGHT - wp.int32(1)) / (max_elev - min_elev)
                )

            azimuth = wp.atan2(voxel_lidar[1], voxel_lidar[0])
            u_float = (azimuth + PI) * (wp.float32(LIDAR_IMAGE_WIDTH) / TWO_PI)
            if u_float >= wp.float32(LIDAR_IMAGE_WIDTH):
                u_float = u_float - wp.float32(LIDAR_IMAGE_WIDTH)
            if u_float < wp.float32(0.0):
                u_float = u_float + wp.float32(LIDAR_IMAGE_WIDTH)

            interp = _interpolate_lidar_range(
                lidar_i,
                u_float,
                v_float,
                voxel_lidar,
                min_range,
                max_range,
                min_elev,
                max_elev,
                linear_interpolation_max_allowable_difference_m,
                nearest_interpolation_max_allowable_dist_to_ray_m,
                range_images,
            )
            if interp[1] <= wp.float32(0.0):
                continue

            surface_range = interp[0]
            sdf = surface_range - voxel_range
            if sdf >= -TRUNCATION_DIST:
                sdf_clamped = wp.min(sdf, TRUNCATION_DIST)
                weight = compute_tsdf_weight(surface_range, VOXEL_SIZE)
                total_sw = total_sw + sdf_clamped * weight
                total_w = total_w + weight

        if total_w > wp.float32(0.0):
            old_sw = wp.float32(block_data[pool_idx, local_idx, 0])
            old_w = wp.float32(block_data[pool_idx, local_idx, 1])
            block_data[pool_idx, local_idx, 0] = wp.float16(old_sw + total_sw)
            block_data[pool_idx, local_idx, 1] = wp.float16(old_w + total_w)

    @warp_kernel(f"lidar_integrate_block_grid_rgb_kernel_{suffix}")
    def lidar_integrate_block_grid_rgb_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        lidar_positions: wp.array2d(dtype=wp.float32),
        lidar_quaternions: wp.array2d(dtype=wp.float32),
        range_images: wp.array3d(dtype=wp.float32),
        rgb_images_flat: wp.array2d(dtype=wp.vec3ub),
        support_counts: wp.array2d(dtype=wp.int32),
        support_pixels: wp.array3d(dtype=wp.int32),
        valid_range_m: wp.array2d(dtype=wp.float32),
        elevation_range_rad: wp.array2d(dtype=wp.float32),
        block_coords: wp.array(dtype=wp.int32),
        block_grid_rgb: wp.array3d(dtype=wp.float16),
        origin: wp.array(dtype=wp.vec3),
    ):
        vis_idx, lidar_i, node_idx = wp.tid()
        if vis_idx >= n_visible or node_idx >= COLOR_GRID_VOXELS:
            return
        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < wp.int32(0):
            return

        bx = block_coords[pool_idx * 3 + 0]
        by = block_coords[pool_idx * 3 + 1]
        bz = block_coords[pool_idx * 3 + 2]

        gx_node = node_idx % COLOR_GRID_SIZE
        gy_node = (node_idx // COLOR_GRID_SIZE) % COLOR_GRID_SIZE
        gz_node = node_idx // (COLOR_GRID_SIZE * COLOR_GRID_SIZE)

        local_x = wp.float32(0.0)
        local_y = wp.float32(0.0)
        local_z = wp.float32(0.0)
        if COLOR_GRID_SIZE > wp.int32(1):
            span = wp.float32(BLOCK_SIZE - wp.int32(1))
            denom = wp.float32(COLOR_GRID_SIZE - wp.int32(1))
            local_x = wp.float32(0.5) + wp.float32(gx_node) * span / denom
            local_y = wp.float32(0.5) + wp.float32(gy_node) * span / denom
            local_z = wp.float32(0.5) + wp.float32(gz_node) * span / denom
        else:
            center = wp.float32(BLOCK_SIZE) * wp.float32(0.5)
            local_x = center
            local_y = center
            local_z = center

        base = block_key_to_voxel_base(bx, by, bz)
        center_offset_x = wp.float32(GRID_W) * wp.float32(0.5)
        center_offset_y = wp.float32(GRID_H) * wp.float32(0.5)
        center_offset_z = wp.float32(GRID_D) * wp.float32(0.5)
        node_world = (
            origin[0]
            + wp.vec3(
                wp.float32(base[0]) + local_x - center_offset_x,
                wp.float32(base[1]) + local_y - center_offset_y,
                wp.float32(base[2]) + local_z - center_offset_z,
            )
            * VOXEL_SIZE
        )

        lidar_pos = wp.vec3(
            lidar_positions[lidar_i, 0],
            lidar_positions[lidar_i, 1],
            lidar_positions[lidar_i, 2],
        )
        lidar_quat = wp.quaternion(
            lidar_quaternions[lidar_i, 1],
            lidar_quaternions[lidar_i, 2],
            lidar_quaternions[lidar_i, 3],
            lidar_quaternions[lidar_i, 0],
        )
        node_lidar = wp.quat_rotate(wp.quat_inverse(lidar_quat), node_world - lidar_pos)
        node_range = wp.sqrt(wp.dot(node_lidar, node_lidar))
        inv_255 = wp.float32(1.0 / 255.0)

        min_range = valid_range_m[lidar_i, 0]
        max_range = valid_range_m[lidar_i, 1]
        if not _range_valid(node_range, min_range, max_range):
            count = support_counts[vis_idx, lidar_i]
            if count > wp.int32(0):
                pixel_idx = support_pixels[vis_idx, lidar_i, 0]
                px = pixel_idx % LIDAR_IMAGE_WIDTH
                py = pixel_idx // LIDAR_IMAGE_WIDTH
                if py >= wp.int32(0) and py < LIDAR_IMAGE_HEIGHT and px >= wp.int32(0) and px < LIDAR_IMAGE_WIDTH:
                    row = lidar_i * LIDAR_IMAGE_HEIGHT + py
                    rgb = rgb_images_flat[row, px]
                    wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 0, wp.float16(wp.float32(rgb[0]) * inv_255))
                    wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 1, wp.float16(wp.float32(rgb[1]) * inv_255))
                    wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 2, wp.float16(wp.float32(rgb[2]) * inv_255))
                    wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 3, wp.float16(1.0))
            return

        min_elev = elevation_range_rad[lidar_i, 0]
        max_elev = elevation_range_rad[lidar_i, 1]
        xy_norm = wp.sqrt(node_lidar[0] * node_lidar[0] + node_lidar[1] * node_lidar[1])
        elevation = wp.atan2(node_lidar[2], xy_norm)
        v_float = wp.float32(0.0)
        if LIDAR_IMAGE_HEIGHT > wp.int32(1):
            if elevation < min_elev or elevation > max_elev:
                count = support_counts[vis_idx, lidar_i]
                if count > wp.int32(0):
                    pixel_idx = support_pixels[vis_idx, lidar_i, 0]
                    px = pixel_idx % LIDAR_IMAGE_WIDTH
                    py = pixel_idx // LIDAR_IMAGE_WIDTH
                    if py >= wp.int32(0) and py < LIDAR_IMAGE_HEIGHT and px >= wp.int32(0) and px < LIDAR_IMAGE_WIDTH:
                        row = lidar_i * LIDAR_IMAGE_HEIGHT + py
                        rgb = rgb_images_flat[row, px]
                        wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 0, wp.float16(wp.float32(rgb[0]) * inv_255))
                        wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 1, wp.float16(wp.float32(rgb[1]) * inv_255))
                        wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 2, wp.float16(wp.float32(rgb[2]) * inv_255))
                        wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 3, wp.float16(1.0))
                return
            v_float = (max_elev - elevation) * (
                wp.float32(LIDAR_IMAGE_HEIGHT - wp.int32(1)) / (max_elev - min_elev)
            )

        azimuth = wp.atan2(node_lidar[1], node_lidar[0])
        u_float = (azimuth + PI) * (wp.float32(LIDAR_IMAGE_WIDTH) / TWO_PI)
        if u_float >= wp.float32(LIDAR_IMAGE_WIDTH):
            u_float = u_float - wp.float32(LIDAR_IMAGE_WIDTH)
        if u_float < wp.float32(0.0):
            u_float = u_float + wp.float32(LIDAR_IMAGE_WIDTH)

        u0 = wp.int32(wp.floor(u_float))
        if u0 < wp.int32(0):
            u0 = u0 + LIDAR_IMAGE_WIDTH
        if u0 >= LIDAR_IMAGE_WIDTH:
            u0 = u0 - LIDAR_IMAGE_WIDTH
        u1 = u0 + wp.int32(1)
        if u1 >= LIDAR_IMAGE_WIDTH:
            u1 = wp.int32(0)
        fu = u_float - wp.floor(u_float)

        v0 = wp.int32(0)
        v1 = wp.int32(0)
        fv = wp.float32(0.0)
        if LIDAR_IMAGE_HEIGHT > wp.int32(1):
            v0 = wp.int32(wp.floor(v_float))
            if v0 < wp.int32(0) or v0 >= LIDAR_IMAGE_HEIGHT:
                count = support_counts[vis_idx, lidar_i]
                if count > wp.int32(0):
                    pixel_idx = support_pixels[vis_idx, lidar_i, 0]
                    px = pixel_idx % LIDAR_IMAGE_WIDTH
                    py = pixel_idx // LIDAR_IMAGE_WIDTH
                    if py >= wp.int32(0) and py < LIDAR_IMAGE_HEIGHT and px >= wp.int32(0) and px < LIDAR_IMAGE_WIDTH:
                        row = lidar_i * LIDAR_IMAGE_HEIGHT + py
                        rgb = rgb_images_flat[row, px]
                        wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 0, wp.float16(wp.float32(rgb[0]) * inv_255))
                        wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 1, wp.float16(wp.float32(rgb[1]) * inv_255))
                        wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 2, wp.float16(wp.float32(rgb[2]) * inv_255))
                        wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 3, wp.float16(1.0))
                return
            v1 = wp.min(v0 + wp.int32(1), LIDAR_IMAGE_HEIGHT - wp.int32(1))
            fv = v_float - wp.float32(v0)

        wx0 = wp.float32(1.0) - fu
        wy0 = wp.float32(1.0) - fv
        w00 = wx0 * wy0
        w10 = fu * wy0
        w01 = wx0 * fv
        w11 = fu * fv
        total_r = wp.float32(0.0)
        total_g = wp.float32(0.0)
        total_b = wp.float32(0.0)
        total_w = wp.float32(0.0)

        range00 = range_images[lidar_i, v0, u0]
        if _range_valid(range00, min_range, max_range):
            sdf00 = range00 - node_range
            if sdf00 >= -TRUNCATION_DIST and sdf00 <= TRUNCATION_DIST:
                weight00 = w00 * compute_tsdf_weight(range00, VOXEL_SIZE)
                row00 = lidar_i * LIDAR_IMAGE_HEIGHT + v0
                rgb00 = rgb_images_flat[row00, u0]
                total_r = total_r + wp.float32(rgb00[0]) * inv_255 * weight00
                total_g = total_g + wp.float32(rgb00[1]) * inv_255 * weight00
                total_b = total_b + wp.float32(rgb00[2]) * inv_255 * weight00
                total_w = total_w + weight00

        range10 = range_images[lidar_i, v0, u1]
        if _range_valid(range10, min_range, max_range):
            sdf10 = range10 - node_range
            if sdf10 >= -TRUNCATION_DIST and sdf10 <= TRUNCATION_DIST:
                weight10 = w10 * compute_tsdf_weight(range10, VOXEL_SIZE)
                row10 = lidar_i * LIDAR_IMAGE_HEIGHT + v0
                rgb10 = rgb_images_flat[row10, u1]
                total_r = total_r + wp.float32(rgb10[0]) * inv_255 * weight10
                total_g = total_g + wp.float32(rgb10[1]) * inv_255 * weight10
                total_b = total_b + wp.float32(rgb10[2]) * inv_255 * weight10
                total_w = total_w + weight10

        range01 = range_images[lidar_i, v1, u0]
        if _range_valid(range01, min_range, max_range):
            sdf01 = range01 - node_range
            if sdf01 >= -TRUNCATION_DIST and sdf01 <= TRUNCATION_DIST:
                weight01 = w01 * compute_tsdf_weight(range01, VOXEL_SIZE)
                row01 = lidar_i * LIDAR_IMAGE_HEIGHT + v1
                rgb01 = rgb_images_flat[row01, u0]
                total_r = total_r + wp.float32(rgb01[0]) * inv_255 * weight01
                total_g = total_g + wp.float32(rgb01[1]) * inv_255 * weight01
                total_b = total_b + wp.float32(rgb01[2]) * inv_255 * weight01
                total_w = total_w + weight01

        range11 = range_images[lidar_i, v1, u1]
        if _range_valid(range11, min_range, max_range):
            sdf11 = range11 - node_range
            if sdf11 >= -TRUNCATION_DIST and sdf11 <= TRUNCATION_DIST:
                weight11 = w11 * compute_tsdf_weight(range11, VOXEL_SIZE)
                row11 = lidar_i * LIDAR_IMAGE_HEIGHT + v1
                rgb11 = rgb_images_flat[row11, u1]
                total_r = total_r + wp.float32(rgb11[0]) * inv_255 * weight11
                total_g = total_g + wp.float32(rgb11[1]) * inv_255 * weight11
                total_b = total_b + wp.float32(rgb11[2]) * inv_255 * weight11
                total_w = total_w + weight11

        if total_w > wp.float32(0.0):
            wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 0, wp.float16(total_r))
            wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 1, wp.float16(total_g))
            wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 2, wp.float16(total_b))
            wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 3, wp.float16(total_w))
        else:
            count = support_counts[vis_idx, lidar_i]
            if count > wp.int32(0):
                pixel_idx = support_pixels[vis_idx, lidar_i, 0]
                px = pixel_idx % LIDAR_IMAGE_WIDTH
                py = pixel_idx // LIDAR_IMAGE_WIDTH
                if py >= wp.int32(0) and py < LIDAR_IMAGE_HEIGHT and px >= wp.int32(0) and px < LIDAR_IMAGE_WIDTH:
                    row = lidar_i * LIDAR_IMAGE_HEIGHT + py
                    rgb = rgb_images_flat[row, px]
                    wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 0, wp.float16(wp.float32(rgb[0]) * inv_255))
                    wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 1, wp.float16(wp.float32(rgb[1]) * inv_255))
                    wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 2, wp.float16(wp.float32(rgb[2]) * inv_255))
                    wp.atomic_add(block_grid_rgb, pool_idx, node_idx, 3, wp.float16(1.0))

    @warp_kernel(
        f"lidar_integrate_features_from_support_grouped_kernel_{suffix}_fcpt"
        f"{feature_channels_per_thread}_sc{support_capacity}"
    )
    def lidar_integrate_features_from_support_grouped_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        lidar_positions: wp.array2d(dtype=wp.float32),
        lidar_quaternions: wp.array2d(dtype=wp.float32),
        range_images: wp.array3d(dtype=wp.float32),
        valid_range_m: wp.array2d(dtype=wp.float32),
        elevation_range_rad: wp.array2d(dtype=wp.float32),
        block_coords: wp.array(dtype=wp.int32),
        support_counts: wp.array2d(dtype=wp.int32),
        support_pixels: wp.array3d(dtype=wp.int32),
        feature_grid: wp.array4d(dtype=wp.float16),
        block_features: wp.array3d(dtype=wp.float16),
        block_feature_weight: wp.array2d(dtype=wp.float16),
        origin: wp.array(dtype=wp.vec3),
    ):
        n_channel_groups = (FEATURE_DIM + FEATURE_CHANNELS_PER_THREAD - wp.int32(1)) // FEATURE_CHANNELS_PER_THREAD
        vis_idx, lidar_i, node_group_idx = wp.tid()
        node_idx = node_group_idx // n_channel_groups
        feature_channel_group_idx = node_group_idx - node_idx * n_channel_groups
        if (
            vis_idx >= n_visible
            or node_idx >= FEATURE_GRID_VOXELS
            or feature_channel_group_idx >= n_channel_groups
        ):
            return

        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < wp.int32(0):
            return

        bx = block_coords[pool_idx * 3 + 0]
        by = block_coords[pool_idx * 3 + 1]
        bz = block_coords[pool_idx * 3 + 2]

        gx_node = node_idx % FEATURE_BLOCK_GRID_SIZE
        gy_node = (node_idx // FEATURE_BLOCK_GRID_SIZE) % FEATURE_BLOCK_GRID_SIZE
        gz_node = node_idx // (FEATURE_BLOCK_GRID_SIZE * FEATURE_BLOCK_GRID_SIZE)

        local_x = wp.float32(0.0)
        local_y = wp.float32(0.0)
        local_z = wp.float32(0.0)
        if FEATURE_BLOCK_GRID_SIZE > wp.int32(1):
            span = wp.float32(BLOCK_SIZE - wp.int32(1))
            denom = wp.float32(FEATURE_BLOCK_GRID_SIZE - wp.int32(1))
            local_x = wp.float32(0.5) + wp.float32(gx_node) * span / denom
            local_y = wp.float32(0.5) + wp.float32(gy_node) * span / denom
            local_z = wp.float32(0.5) + wp.float32(gz_node) * span / denom
        else:
            center = wp.float32(BLOCK_SIZE) * wp.float32(0.5)
            local_x = center
            local_y = center
            local_z = center

        base = block_key_to_voxel_base(bx, by, bz)
        center_offset_x = wp.float32(GRID_W) * wp.float32(0.5)
        center_offset_y = wp.float32(GRID_H) * wp.float32(0.5)
        center_offset_z = wp.float32(GRID_D) * wp.float32(0.5)
        node_world = (
            origin[0]
            + wp.vec3(
                wp.float32(base[0]) + local_x - center_offset_x,
                wp.float32(base[1]) + local_y - center_offset_y,
                wp.float32(base[2]) + local_z - center_offset_z,
            )
            * VOXEL_SIZE
        )

        base_feature_channel = feature_channel_group_idx * FEATURE_CHANNELS_PER_THREAD
        feature_acc = wp.zeros(FEATURE_CHANNELS_PER_THREAD, dtype=wp.float32)
        total_w = wp.float32(0.0)

        lidar_pos = wp.vec3(
            lidar_positions[lidar_i, 0],
            lidar_positions[lidar_i, 1],
            lidar_positions[lidar_i, 2],
        )
        lidar_quat = wp.quaternion(
            lidar_quaternions[lidar_i, 1],
            lidar_quaternions[lidar_i, 2],
            lidar_quaternions[lidar_i, 3],
            lidar_quaternions[lidar_i, 0],
        )
        node_lidar = wp.quat_rotate(wp.quat_inverse(lidar_quat), node_world - lidar_pos)
        node_range = wp.sqrt(wp.dot(node_lidar, node_lidar))
        min_range = valid_range_m[lidar_i, 0]
        max_range = valid_range_m[lidar_i, 1]

        if _range_valid(node_range, min_range, max_range):
            min_elev = elevation_range_rad[lidar_i, 0]
            max_elev = elevation_range_rad[lidar_i, 1]
            xy_norm = wp.sqrt(node_lidar[0] * node_lidar[0] + node_lidar[1] * node_lidar[1])
            elevation = wp.atan2(node_lidar[2], xy_norm)
            can_project = bool(True)
            v_float = wp.float32(0.0)
            if LIDAR_IMAGE_HEIGHT > wp.int32(1):
                if elevation < min_elev or elevation > max_elev:
                    can_project = False
                else:
                    v_float = (max_elev - elevation) * (
                        wp.float32(LIDAR_IMAGE_HEIGHT - wp.int32(1))
                        / (max_elev - min_elev)
                    )

            if can_project:
                azimuth = wp.atan2(node_lidar[1], node_lidar[0])
                u_float = (azimuth + PI) * (wp.float32(LIDAR_IMAGE_WIDTH) / TWO_PI)
                if u_float >= wp.float32(LIDAR_IMAGE_WIDTH):
                    u_float = u_float - wp.float32(LIDAR_IMAGE_WIDTH)
                if u_float < wp.float32(0.0):
                    u_float = u_float + wp.float32(LIDAR_IMAGE_WIDTH)

                u0 = wp.int32(wp.floor(u_float))
                if u0 < wp.int32(0):
                    u0 = u0 + LIDAR_IMAGE_WIDTH
                if u0 >= LIDAR_IMAGE_WIDTH:
                    u0 = u0 - LIDAR_IMAGE_WIDTH
                u1 = u0 + wp.int32(1)
                if u1 >= LIDAR_IMAGE_WIDTH:
                    u1 = wp.int32(0)
                fu = u_float - wp.floor(u_float)

                v0 = wp.int32(0)
                v1 = wp.int32(0)
                fv = wp.float32(0.0)
                can_sample = bool(True)
                if LIDAR_IMAGE_HEIGHT > wp.int32(1):
                    v0 = wp.int32(wp.floor(v_float))
                    if v0 < wp.int32(0) or v0 >= LIDAR_IMAGE_HEIGHT:
                        can_sample = False
                    else:
                        v1 = wp.min(v0 + wp.int32(1), LIDAR_IMAGE_HEIGHT - wp.int32(1))
                        fv = v_float - wp.float32(v0)

                if can_sample:
                    wx0 = wp.float32(1.0) - fu
                    wy0 = wp.float32(1.0) - fv
                    w00 = wx0 * wy0
                    w10 = fu * wy0
                    w01 = wx0 * fv
                    w11 = fu * fv

                    range00 = range_images[lidar_i, v0, u0]
                    if _range_valid(range00, min_range, max_range):
                        sdf00 = range00 - node_range
                        if sdf00 >= -TRUNCATION_DIST and sdf00 <= TRUNCATION_DIST:
                            sample_w = w00 * compute_tsdf_weight(range00, VOXEL_SIZE)
                            fg = _lidar_feature_grid_coords(u0, v0)
                            for feature_channel_offset in range(feature_channels_per_thread):
                                feature_channel = base_feature_channel + feature_channel_offset
                                if feature_channel < FEATURE_DIM:
                                    feature_acc[feature_channel_offset] = (
                                        feature_acc[feature_channel_offset]
                                        + wp.float32(
                                            feature_grid[
                                                lidar_i,
                                                fg[1],
                                                fg[0],
                                                feature_channel,
                                            ]
                                        )
                                        * sample_w
                                    )
                            total_w = total_w + sample_w

                    range10 = range_images[lidar_i, v0, u1]
                    if _range_valid(range10, min_range, max_range):
                        sdf10 = range10 - node_range
                        if sdf10 >= -TRUNCATION_DIST and sdf10 <= TRUNCATION_DIST:
                            sample_w = w10 * compute_tsdf_weight(range10, VOXEL_SIZE)
                            fg = _lidar_feature_grid_coords(u1, v0)
                            for feature_channel_offset in range(feature_channels_per_thread):
                                feature_channel = base_feature_channel + feature_channel_offset
                                if feature_channel < FEATURE_DIM:
                                    feature_acc[feature_channel_offset] = (
                                        feature_acc[feature_channel_offset]
                                        + wp.float32(
                                            feature_grid[
                                                lidar_i,
                                                fg[1],
                                                fg[0],
                                                feature_channel,
                                            ]
                                        )
                                        * sample_w
                                    )
                            total_w = total_w + sample_w

                    range01 = range_images[lidar_i, v1, u0]
                    if _range_valid(range01, min_range, max_range):
                        sdf01 = range01 - node_range
                        if sdf01 >= -TRUNCATION_DIST and sdf01 <= TRUNCATION_DIST:
                            sample_w = w01 * compute_tsdf_weight(range01, VOXEL_SIZE)
                            fg = _lidar_feature_grid_coords(u0, v1)
                            for feature_channel_offset in range(feature_channels_per_thread):
                                feature_channel = base_feature_channel + feature_channel_offset
                                if feature_channel < FEATURE_DIM:
                                    feature_acc[feature_channel_offset] = (
                                        feature_acc[feature_channel_offset]
                                        + wp.float32(
                                            feature_grid[
                                                lidar_i,
                                                fg[1],
                                                fg[0],
                                                feature_channel,
                                            ]
                                        )
                                        * sample_w
                                    )
                            total_w = total_w + sample_w

                    range11 = range_images[lidar_i, v1, u1]
                    if _range_valid(range11, min_range, max_range):
                        sdf11 = range11 - node_range
                        if sdf11 >= -TRUNCATION_DIST and sdf11 <= TRUNCATION_DIST:
                            sample_w = w11 * compute_tsdf_weight(range11, VOXEL_SIZE)
                            fg = _lidar_feature_grid_coords(u1, v1)
                            for feature_channel_offset in range(feature_channels_per_thread):
                                feature_channel = base_feature_channel + feature_channel_offset
                                if feature_channel < FEATURE_DIM:
                                    feature_acc[feature_channel_offset] = (
                                        feature_acc[feature_channel_offset]
                                        + wp.float32(
                                            feature_grid[
                                                lidar_i,
                                                fg[1],
                                                fg[0],
                                                feature_channel,
                                            ]
                                        )
                                        * sample_w
                                    )
                            total_w = total_w + sample_w

        if total_w <= wp.float32(0.0):
            count = support_counts[vis_idx, lidar_i]
            if count > wp.int32(0):
                pixel_idx = support_pixels[vis_idx, lidar_i, 0]
                px = pixel_idx % LIDAR_IMAGE_WIDTH
                py = pixel_idx // LIDAR_IMAGE_WIDTH
                if py >= wp.int32(0) and py < LIDAR_IMAGE_HEIGHT and px >= wp.int32(0) and px < LIDAR_IMAGE_WIDTH:
                    fg = _lidar_feature_grid_coords(px, py)
                    for feature_channel_offset in range(feature_channels_per_thread):
                        feature_channel = base_feature_channel + feature_channel_offset
                        if feature_channel < FEATURE_DIM:
                            feature_acc[feature_channel_offset] = (
                                feature_acc[feature_channel_offset]
                                + wp.float32(
                                    feature_grid[
                                        lidar_i,
                                        fg[1],
                                        fg[0],
                                        feature_channel,
                                    ]
                                )
                            )
                    total_w = total_w + wp.float32(1.0)

        if total_w <= wp.float32(0.0):
            return

        for feature_channel_offset in range(feature_channels_per_thread):
            feature_channel = base_feature_channel + feature_channel_offset
            if feature_channel < FEATURE_DIM:
                wp.atomic_add(
                    block_features,
                    pool_idx,
                    node_idx,
                    feature_channel,
                    wp.float16(feature_acc[feature_channel_offset]),
                )
        if base_feature_channel == wp.int32(0):
            wp.atomic_add(block_feature_weight, pool_idx, node_idx, wp.float16(total_w))

    @warp_kernel(
        f"lidar_integrate_features_from_support_tiled_kernel_{suffix}_tile"
        f"{feature_tile_channels}_sc{support_capacity}"
    )
    def lidar_integrate_features_from_support_tiled_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        lidar_positions: wp.array2d(dtype=wp.float32),
        lidar_quaternions: wp.array2d(dtype=wp.float32),
        range_images: wp.array3d(dtype=wp.float32),
        valid_range_m: wp.array2d(dtype=wp.float32),
        elevation_range_rad: wp.array2d(dtype=wp.float32),
        block_coords: wp.array(dtype=wp.int32),
        support_counts: wp.array2d(dtype=wp.int32),
        support_pixels: wp.array3d(dtype=wp.int32),
        feature_grid: wp.array4d(dtype=wp.float16),
        block_features: wp.array3d(dtype=wp.float16),
        block_feature_weight: wp.array2d(dtype=wp.float16),
        origin: wp.array(dtype=wp.vec3),
    ):
        n_channel_tiles = (FEATURE_DIM + FEATURE_TILE_CHANNELS - wp.int32(1)) // FEATURE_TILE_CHANNELS
        vis_idx, lidar_i, node_tile_idx, lane = wp.tid()
        node_idx = node_tile_idx // n_channel_tiles
        feature_tile_idx = node_tile_idx - node_idx * n_channel_tiles
        if (
            vis_idx >= n_visible
            or node_idx >= FEATURE_GRID_VOXELS
            or feature_tile_idx >= n_channel_tiles
        ):
            return
        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < wp.int32(0):
            return

        bx = block_coords[pool_idx * 3 + 0]
        by = block_coords[pool_idx * 3 + 1]
        bz = block_coords[pool_idx * 3 + 2]

        gx_node = node_idx % FEATURE_BLOCK_GRID_SIZE
        gy_node = (node_idx // FEATURE_BLOCK_GRID_SIZE) % FEATURE_BLOCK_GRID_SIZE
        gz_node = node_idx // (FEATURE_BLOCK_GRID_SIZE * FEATURE_BLOCK_GRID_SIZE)

        local_x = wp.float32(0.0)
        local_y = wp.float32(0.0)
        local_z = wp.float32(0.0)
        if FEATURE_BLOCK_GRID_SIZE > wp.int32(1):
            span = wp.float32(BLOCK_SIZE - wp.int32(1))
            denom = wp.float32(FEATURE_BLOCK_GRID_SIZE - wp.int32(1))
            local_x = wp.float32(0.5) + wp.float32(gx_node) * span / denom
            local_y = wp.float32(0.5) + wp.float32(gy_node) * span / denom
            local_z = wp.float32(0.5) + wp.float32(gz_node) * span / denom
        else:
            center = wp.float32(BLOCK_SIZE) * wp.float32(0.5)
            local_x = center
            local_y = center
            local_z = center

        base = block_key_to_voxel_base(bx, by, bz)
        center_offset_x = wp.float32(GRID_W) * wp.float32(0.5)
        center_offset_y = wp.float32(GRID_H) * wp.float32(0.5)
        center_offset_z = wp.float32(GRID_D) * wp.float32(0.5)
        node_world = (
            origin[0]
            + wp.vec3(
                wp.float32(base[0]) + local_x - center_offset_x,
                wp.float32(base[1]) + local_y - center_offset_y,
                wp.float32(base[2]) + local_z - center_offset_z,
            )
            * VOXEL_SIZE
        )

        base_feature_channel = feature_tile_idx * FEATURE_TILE_CHANNELS
        feature_acc = wp.tile_zeros(shape=feature_tile_channels, dtype=wp.float32)
        total_w = wp.float32(0.0)

        lidar_pos = wp.vec3(
            lidar_positions[lidar_i, 0],
            lidar_positions[lidar_i, 1],
            lidar_positions[lidar_i, 2],
        )
        lidar_quat = wp.quaternion(
            lidar_quaternions[lidar_i, 1],
            lidar_quaternions[lidar_i, 2],
            lidar_quaternions[lidar_i, 3],
            lidar_quaternions[lidar_i, 0],
        )
        node_lidar = wp.quat_rotate(wp.quat_inverse(lidar_quat), node_world - lidar_pos)
        node_range = wp.sqrt(wp.dot(node_lidar, node_lidar))
        min_range = valid_range_m[lidar_i, 0]
        max_range = valid_range_m[lidar_i, 1]

        if _range_valid(node_range, min_range, max_range):
            min_elev = elevation_range_rad[lidar_i, 0]
            max_elev = elevation_range_rad[lidar_i, 1]
            xy_norm = wp.sqrt(node_lidar[0] * node_lidar[0] + node_lidar[1] * node_lidar[1])
            elevation = wp.atan2(node_lidar[2], xy_norm)
            can_project = bool(True)
            v_float = wp.float32(0.0)
            if LIDAR_IMAGE_HEIGHT > wp.int32(1):
                if elevation < min_elev or elevation > max_elev:
                    can_project = False
                else:
                    v_float = (max_elev - elevation) * (
                        wp.float32(LIDAR_IMAGE_HEIGHT - wp.int32(1))
                        / (max_elev - min_elev)
                    )

            if can_project:
                azimuth = wp.atan2(node_lidar[1], node_lidar[0])
                u_float = (azimuth + PI) * (wp.float32(LIDAR_IMAGE_WIDTH) / TWO_PI)
                if u_float >= wp.float32(LIDAR_IMAGE_WIDTH):
                    u_float = u_float - wp.float32(LIDAR_IMAGE_WIDTH)
                if u_float < wp.float32(0.0):
                    u_float = u_float + wp.float32(LIDAR_IMAGE_WIDTH)

                u0 = wp.int32(wp.floor(u_float))
                if u0 < wp.int32(0):
                    u0 = u0 + LIDAR_IMAGE_WIDTH
                if u0 >= LIDAR_IMAGE_WIDTH:
                    u0 = u0 - LIDAR_IMAGE_WIDTH
                u1 = u0 + wp.int32(1)
                if u1 >= LIDAR_IMAGE_WIDTH:
                    u1 = wp.int32(0)
                fu = u_float - wp.floor(u_float)

                v0 = wp.int32(0)
                v1 = wp.int32(0)
                fv = wp.float32(0.0)
                can_sample = bool(True)
                if LIDAR_IMAGE_HEIGHT > wp.int32(1):
                    v0 = wp.int32(wp.floor(v_float))
                    if v0 < wp.int32(0) or v0 >= LIDAR_IMAGE_HEIGHT:
                        can_sample = False
                    else:
                        v1 = wp.min(v0 + wp.int32(1), LIDAR_IMAGE_HEIGHT - wp.int32(1))
                        fv = v_float - wp.float32(v0)

                if can_sample:
                    wx0 = wp.float32(1.0) - fu
                    wy0 = wp.float32(1.0) - fv
                    w00 = wx0 * wy0
                    w10 = fu * wy0
                    w01 = wx0 * fv
                    w11 = fu * fv

                    range00 = range_images[lidar_i, v0, u0]
                    if _range_valid(range00, min_range, max_range):
                        sdf00 = range00 - node_range
                        if sdf00 >= -TRUNCATION_DIST and sdf00 <= TRUNCATION_DIST:
                            sample_w = w00 * compute_tsdf_weight(range00, VOXEL_SIZE)
                            fg = _lidar_feature_grid_coords(u0, v0)
                            feature_vals_h = wp.tile_load(
                                feature_grid[lidar_i, fg[1], fg[0]],
                                shape=feature_tile_channels,
                                offset=base_feature_channel,
                                bounds_check=True,
                            )
                            feature_acc = feature_acc + wp.tile_astype(
                                feature_vals_h,
                                dtype=wp.float32,
                            ) * sample_w
                            total_w = total_w + sample_w

                    range10 = range_images[lidar_i, v0, u1]
                    if _range_valid(range10, min_range, max_range):
                        sdf10 = range10 - node_range
                        if sdf10 >= -TRUNCATION_DIST and sdf10 <= TRUNCATION_DIST:
                            sample_w = w10 * compute_tsdf_weight(range10, VOXEL_SIZE)
                            fg = _lidar_feature_grid_coords(u1, v0)
                            feature_vals_h = wp.tile_load(
                                feature_grid[lidar_i, fg[1], fg[0]],
                                shape=feature_tile_channels,
                                offset=base_feature_channel,
                                bounds_check=True,
                            )
                            feature_acc = feature_acc + wp.tile_astype(
                                feature_vals_h,
                                dtype=wp.float32,
                            ) * sample_w
                            total_w = total_w + sample_w

                    range01 = range_images[lidar_i, v1, u0]
                    if _range_valid(range01, min_range, max_range):
                        sdf01 = range01 - node_range
                        if sdf01 >= -TRUNCATION_DIST and sdf01 <= TRUNCATION_DIST:
                            sample_w = w01 * compute_tsdf_weight(range01, VOXEL_SIZE)
                            fg = _lidar_feature_grid_coords(u0, v1)
                            feature_vals_h = wp.tile_load(
                                feature_grid[lidar_i, fg[1], fg[0]],
                                shape=feature_tile_channels,
                                offset=base_feature_channel,
                                bounds_check=True,
                            )
                            feature_acc = feature_acc + wp.tile_astype(
                                feature_vals_h,
                                dtype=wp.float32,
                            ) * sample_w
                            total_w = total_w + sample_w

                    range11 = range_images[lidar_i, v1, u1]
                    if _range_valid(range11, min_range, max_range):
                        sdf11 = range11 - node_range
                        if sdf11 >= -TRUNCATION_DIST and sdf11 <= TRUNCATION_DIST:
                            sample_w = w11 * compute_tsdf_weight(range11, VOXEL_SIZE)
                            fg = _lidar_feature_grid_coords(u1, v1)
                            feature_vals_h = wp.tile_load(
                                feature_grid[lidar_i, fg[1], fg[0]],
                                shape=feature_tile_channels,
                                offset=base_feature_channel,
                                bounds_check=True,
                            )
                            feature_acc = feature_acc + wp.tile_astype(
                                feature_vals_h,
                                dtype=wp.float32,
                            ) * sample_w
                            total_w = total_w + sample_w

        if total_w <= wp.float32(0.0):
            count = support_counts[vis_idx, lidar_i]
            if count > wp.int32(0):
                pixel_idx = support_pixels[vis_idx, lidar_i, 0]
                px = pixel_idx % LIDAR_IMAGE_WIDTH
                py = pixel_idx // LIDAR_IMAGE_WIDTH
                if py >= wp.int32(0) and py < LIDAR_IMAGE_HEIGHT and px >= wp.int32(0) and px < LIDAR_IMAGE_WIDTH:
                    fg = _lidar_feature_grid_coords(px, py)
                    feature_vals_h = wp.tile_load(
                        feature_grid[lidar_i, fg[1], fg[0]],
                        shape=feature_tile_channels,
                        offset=base_feature_channel,
                        bounds_check=True,
                    )
                    feature_acc = feature_acc + wp.tile_astype(
                        feature_vals_h,
                        dtype=wp.float32,
                    )
                    total_w = total_w + wp.float32(1.0)

        if total_w <= wp.float32(0.0):
            return

        feature_acc_h = wp.tile_astype(feature_acc, dtype=wp.float16)
        wp.tile_atomic_add(
            block_features[pool_idx, node_idx],
            feature_acc_h,
            offset=base_feature_channel,
            bounds_check=True,
        )
        if feature_tile_idx == wp.int32(0) and lane == wp.int32(0):
            wp.atomic_add(block_feature_weight, pool_idx, node_idx, wp.float16(total_w))

    return {
        "lidar_compute_block_keys_only_kernel": lidar_compute_block_keys_only_kernel,
        "lidar_build_support_pixels_from_keys_kernel": (
            lidar_build_support_pixels_from_keys_kernel
        ),
        "lidar_integrate_voxels_kernel": lidar_integrate_voxels_kernel,
        "lidar_integrate_block_grid_rgb_kernel": lidar_integrate_block_grid_rgb_kernel,
        "lidar_integrate_features_from_support_grouped_kernel": (
            lidar_integrate_features_from_support_grouped_kernel
        ),
        "lidar_integrate_features_from_support_tiled_kernel": (
            lidar_integrate_features_from_support_tiled_kernel
        ),
    }
