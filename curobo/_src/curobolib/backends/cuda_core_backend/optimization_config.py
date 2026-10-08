# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration for optimization kernel compilation."""

# Standard Library
from pathlib import Path
from typing import List, Tuple

from cuda.core import LaunchConfig

# CuRobo
from curobo._src.curobolib.backends.cuda_core_backend.kernel_config import CudaCoreKernelCfg


class OptimizationKernelCfg(CudaCoreKernelCfg):
    """Configuration for optimization kernel compilation"""

    def __init__(self):
        super().__init__("optimization")

    def get_kernel_files(self, kernel_type: str) -> List[str]:
        """Get kernel source files for a given kernel type.

        Args:
            kernel_type: Type of kernel ("line_search", "lbfgs")

        Returns:
            List of kernel filenames
        """
        kernel_files = {
            "line_search": ["line_search/line_search_kernel.cuh"],
            "lbfgs": ["lbfgs/lbfgs_step_kernel.cuh"],
        }
        return kernel_files.get(kernel_type, [])

    def get_include_dirs(self) -> List[Path]:
        """Get include directories for kernel compilation"""
        # Get base include dirs and add optimization-specific ones
        base_dirs = self.get_base_include_dirs()
        optimization_dirs = [
            self.kernel_dir,  # kernels/optimization/
            self.kernel_dir / "line_search",
            self.kernel_dir / "lbfgs",
        ]
        return base_dirs + optimization_dirs


class LineSearchLaunchCfg:
    """Helper class for calculating launch configurations for line search kernels"""

    @staticmethod
    def calculate_config(opt_dim: int, batchsize: int) -> LaunchConfig:
        """Calculate launch configuration for line search kernel.

        Ported from launch_line_search in line_search_kernel_launch.cu (lines 94-98)

        Args:
            opt_dim: Optimization dimension (number of parameters)
            batchsize: Number of parallel searches

        Returns:
            LaunchConfig for kernel launch
        """
        threads_per_block = opt_dim
        blocks_per_grid = batchsize

        return LaunchConfig(grid=blocks_per_grid, block=threads_per_block, shmem_size=0)


class LBFGSLaunchCfg:
    """Helper class for calculating launch configurations for LBFGS kernels"""

    #: Dynamic shared memory available without opting in.
    MAX_SHARED_BASE = 48000
    #: Dynamic shared memory available after opting in (Volta+).
    MAX_SHARED_ALLOWED = 65536

    @staticmethod
    def compact_shared_memory_bytes(v_dim: int, history_m: int) -> int:
        """Shared memory used by ``kernel_lbfgs_step_compact``.

        Matches ``compact_shared_memory_floats`` in ``lbfgs_step_helpers.cuh``.
        """
        padded_dim = (v_dim + 3) & ~3
        vector_stride = padded_dim + ((4 - padded_dim) & 31)
        gram_stride = history_m | 1
        floats = (
            (2 * history_m + 1) * vector_stride + 2 * history_m * gram_stride + 5 * history_m + 34
        )
        return floats * 4

    @staticmethod
    def calculate_config(
        batch_size: int, v_dim: int, history_m: int, use_shared_buffers: bool
    ) -> Tuple[LaunchConfig, bool, int]:
        """Calculate launch configuration for LBFGS step kernel.

        The compact kernel (shared buffers) runs ``v_dim`` rounded up to a multiple of 32
        threads per problem. The global-memory kernel runs ``v_dim`` threads.

        Args:
            batch_size: Number of batches
            v_dim: Variable dimension
            history_m: History size
            use_shared_buffers: Whether to use the compact shared memory kernel

        Returns:
            Tuple of (LaunchConfig, use_shared_buffers_actual, max_shared_memory_needed)
        """
        compact_smem_size = LBFGSLaunchCfg.compact_shared_memory_bytes(v_dim, history_m)
        if use_shared_buffers and compact_smem_size <= LBFGSLaunchCfg.MAX_SHARED_ALLOWED:
            threads_per_block = ((v_dim + 31) // 32) * 32
            config = LaunchConfig(
                grid=batch_size, block=threads_per_block, shmem_size=compact_smem_size
            )
            return config, True, max(compact_smem_size, LBFGSLaunchCfg.MAX_SHARED_BASE)

        config = LaunchConfig(grid=batch_size, block=v_dim, shmem_size=history_m * 4)
        return config, False, LBFGSLaunchCfg.MAX_SHARED_BASE
