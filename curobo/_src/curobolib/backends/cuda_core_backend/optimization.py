# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""cuda.core backend for optimization kernels.
Provides runtime compilation of CUDA kernels with the same interface as PyBind11 extensions.
"""

# Standard Library
from typing import List

# Third Party
import torch
from cuda.core import LaunchConfig

# CuRobo
from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
from curobo._src.curobolib.backends.cuda_core_backend.optimization_config import (
    LBFGSLaunchCfg,
    LineSearchLaunchCfg,
    LineSearchLBFGSLaunchCfg,
    OptimizationKernelCfg,
)
from curobo._src.util.logging import log_and_raise


def launch_line_search(
    best_cost: torch.Tensor,
    best_action: torch.Tensor,
    best_iteration: torch.Tensor,
    current_iteration: torch.Tensor,
    converged_global: torch.Tensor,
    convergence_iteration: int,
    cost_delta_threshold: float,
    cost_relative_threshold: float,
    exploration_cost: torch.Tensor,
    exploration_action: torch.Tensor,
    exploration_gradient: torch.Tensor,
    exploration_idx: torch.Tensor,
    selected_cost: torch.Tensor,
    selected_action: torch.Tensor,
    selected_gradient: torch.Tensor,
    selected_idx: torch.Tensor,
    search_cost: torch.Tensor,
    search_action: torch.Tensor,
    search_gradient: torch.Tensor,
    step_direction: torch.Tensor,
    search_magnitudes: torch.Tensor,
    armijo_threshold_c_1: float,
    curvature_threshold_c_2: float,
    strong_wolfe: bool,
    approx_wolfe: bool,
    n_linesearch: int,
    opt_dim: int,
    batchsize: int,
):
    """Launch line search kernel using cuda.core runtime compilation.

    This function has the EXACT same signature as the PyBind11 version
    to ensure seamless backend swapping.

    Args:
        All tensor arguments and parameters as expected by line search kernel

    Note:
        Modifies output tensors in-place
    """
    runtime = get_runtime()

    # Get cuda.core kernel cache
    cache = runtime.get_cuda_core_cache()

    # Get kernel configuration
    kernel_config = OptimizationKernelCfg()

    # Select kernel based on n_linesearch
    # kernel_line_search<float, N_LINESEARCH> where N_LINESEARCH is 4 or -1 (runtime)
    if n_linesearch == 4:
        kernel_name = "curobo::optimization::line_search::kernel_line_search<float, 4>"
    else:
        kernel_name = "curobo::optimization::line_search::kernel_line_search<float, -1>"

    # Get or compile kernel
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("line_search")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    # Calculate launch configuration
    config = LineSearchLaunchCfg.calculate_config(opt_dim, batchsize)

    # Get stream wrapper
    pt_stream = torch.cuda.current_stream()
    stream = cache.get_stream_wrapper(pt_stream)

    # Prepare kernel arguments as data pointers
    # Match the C++ kernel signature from lines 104-132 in line_search_kernel_launch.cu
    kernel_args = (
        best_cost.data_ptr(),
        best_action.data_ptr(),
        best_iteration.data_ptr(),
        current_iteration.data_ptr(),
        converged_global.data_ptr(),
        convergence_iteration,
        cost_delta_threshold,
        cost_relative_threshold,
        exploration_cost.data_ptr(),
        exploration_action.data_ptr(),
        exploration_gradient.data_ptr(),
        exploration_idx.data_ptr(),
        selected_cost.data_ptr(),
        selected_action.data_ptr(),
        selected_gradient.data_ptr(),
        selected_idx.data_ptr(),
        search_cost.data_ptr(),
        search_action.data_ptr(),
        search_gradient.data_ptr(),
        step_direction.data_ptr(),
        search_magnitudes.data_ptr(),
        armijo_threshold_c_1,
        curvature_threshold_c_2,
        strong_wolfe,
        approx_wolfe,
        n_linesearch,
        opt_dim,
        batchsize,
    )

    # Launch kernel
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)


def launch_lbfgs_step(
    step_vec: torch.Tensor,
    rho_buffer: torch.Tensor,
    y_buffer: torch.Tensor,
    s_buffer: torch.Tensor,
    gram_buffer: torch.Tensor,
    q: torch.Tensor,
    grad_q: torch.Tensor,
    x_0: torch.Tensor,
    grad_0: torch.Tensor,
    epsilon: float,
    batch_size: int,
    history_m: int,
    v_dim: int,
    stable_mode: bool,
    use_shared_buffers: bool,
    action_step_max: torch.Tensor,
    scale_step: bool,
) -> List[torch.Tensor]:
    """Launch LBFGS step kernel using cuda.core runtime compilation.

    This function has the EXACT same signature as the PyBind11 version
    to ensure seamless backend swapping.

    Args:
        All tensor arguments and parameters as expected by LBFGS step kernel

    Returns:
        List of tensors: [step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0]

    Note:
        Modifies output tensors in-place
    """
    # print("LBFGS: ")
    # print(batch_size, v_dim, history_m)
    runtime = get_runtime()

    # Get cuda.core kernel cache
    cache = runtime.get_cuda_core_cache()

    # Get kernel configuration
    kernel_config = OptimizationKernelCfg()

    # Calculate launch configuration
    config, use_shared_buffers_actual, max_shared_memory_needed = (
        LBFGSLaunchCfg.calculate_config(batch_size, v_dim, history_m, use_shared_buffers)
    )

    # Kernels are compiled per history_m.
    if history_m > 31:
        log_and_raise("History_m greater than 31 is not supported")
    if history_m < 0:
        log_and_raise("History_m less than 0 is not supported")

    if use_shared_buffers_actual:
        kernel_name = f"curobo::optimization::kernel_lbfgs_step_compact<float, {history_m}>"
    else:
        kernel_name = f"curobo::optimization::kernel_lbfgs_step<float, false, {history_m}>"

    # Get or compile kernel
    kernel_files = [kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("lbfgs")]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    # Opt in to more than 48 KB of dynamic shared memory (Volta+), the cuda.core equivalent
    # of cudaFuncSetAttribute.
    if use_shared_buffers_actual and max_shared_memory_needed > LBFGSLaunchCfg.MAX_SHARED_BASE:
        import cuda.bindings.runtime as cudart

        kernel_ptr = int(kernel._handle)
        err = cudart.cudaFuncSetAttribute(
            kernel_ptr,
            cudart.cudaFuncAttribute.cudaFuncAttributeMaxDynamicSharedMemorySize,
            max_shared_memory_needed,
        )
        if err[0] != cudart.cudaError_t.cudaSuccess:
            log_and_raise(
                f"Failed to configure max dynamic shared memory size for kernel {kernel_name}. Error: {err}"
            )

    # Get stream wrapper
    pt_stream = torch.cuda.current_stream()
    stream = cache.get_stream_wrapper(pt_stream)

    # Kernel arguments as data pointers; only the compact kernel takes the Gram buffer.
    gram_args = (gram_buffer.data_ptr(),) if use_shared_buffers_actual else ()
    tensor_args = (
        step_vec.data_ptr(),
        rho_buffer.data_ptr(),
        y_buffer.data_ptr(),
        s_buffer.data_ptr(),
        *gram_args,
        q.data_ptr(),
        x_0.data_ptr(),
        grad_0.data_ptr(),
        grad_q.data_ptr(),
    )
    kernel_args = (
        *tensor_args,
        epsilon,
        batch_size,
        history_m,
        v_dim,
        stable_mode,
        # Step scaling is skipped when the kernel receives a null action_step_max.
        action_step_max.data_ptr() if scale_step else 0,
        action_step_max.numel() if scale_step else 1,
    )

    # Launch kernel
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)

    # Return the same tensors as PyBind version
    return [step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0]


def launch_line_search_lbfgs_step(
    best_cost: torch.Tensor,
    best_action: torch.Tensor,
    best_iteration: torch.Tensor,
    current_iteration: torch.Tensor,
    converged_global: torch.Tensor,
    convergence_iteration: int,
    cost_delta_threshold: float,
    cost_relative_threshold: float,
    exploration_cost: torch.Tensor,
    exploration_action: torch.Tensor,
    exploration_gradient: torch.Tensor,
    exploration_idx: torch.Tensor,
    selected_cost: torch.Tensor,
    selected_action: torch.Tensor,
    selected_gradient: torch.Tensor,
    selected_idx: torch.Tensor,
    search_cost: torch.Tensor,
    search_action: torch.Tensor,
    search_gradient: torch.Tensor,
    search_direction: torch.Tensor,
    search_magnitudes: torch.Tensor,
    armijo_threshold_c_1: float,
    curvature_threshold_c_2: float,
    strong_wolfe: bool,
    approx_wolfe: bool,
    n_linesearch: int,
    step_vec: torch.Tensor,
    rho_buffer: torch.Tensor,
    y_buffer: torch.Tensor,
    s_buffer: torch.Tensor,
    gram_buffer: torch.Tensor,
    x_0: torch.Tensor,
    grad_0: torch.Tensor,
    epsilon: float,
    batch_size: int,
    history_m: int,
    v_dim: int,
    stable_mode: bool,
    action_step_max: torch.Tensor,
    scale_step: bool,
):
    """Launch the fused Wolfe line search and compact L-BFGS step kernel.

    Same signature as the PyBind11 version. Modifies the iteration-state tensors, the
    L-BFGS buffers, ``step_vec`` and ``search_action`` (next search points) in place.
    """
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = OptimizationKernelCfg()

    if history_m > 31 or history_m < 1:
        log_and_raise(f"history_m must be in [1, 31], got {history_m}")

    kernel_name = f"curobo::optimization::kernel_line_search_lbfgs_step<{history_m}>"
    kernel_files = [
        kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("line_search_lbfgs")
    ]
    kernel = cache.get_or_compile_kernel(
        source_files=kernel_files,
        kernel_name=kernel_name,
        include_dirs=kernel_config.get_include_dirs(),
        compile_flags=kernel_config.get_compile_flags(),
    )

    config, dynamic_shared_memory = LineSearchLBFGSLaunchCfg.calculate_config(
        batch_size, v_dim, history_m
    )
    if dynamic_shared_memory > LBFGSLaunchCfg.MAX_SHARED_BASE:
        import cuda.bindings.runtime as cudart

        err = cudart.cudaFuncSetAttribute(
            int(kernel._handle),
            cudart.cudaFuncAttribute.cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory,
        )
        if err[0] != cudart.cudaError_t.cudaSuccess:
            log_and_raise(
                f"Failed to configure max dynamic shared memory size for kernel {kernel_name}. Error: {err}"
            )

    stream = cache.get_stream_wrapper(torch.cuda.current_stream())

    kernel_args = (
        best_cost.data_ptr(),
        best_action.data_ptr(),
        best_iteration.data_ptr(),
        current_iteration.data_ptr(),
        converged_global.data_ptr(),
        convergence_iteration,
        cost_delta_threshold,
        cost_relative_threshold,
        exploration_cost.data_ptr(),
        exploration_action.data_ptr(),
        exploration_gradient.data_ptr(),
        exploration_idx.data_ptr(),
        selected_cost.data_ptr(),
        selected_action.data_ptr(),
        selected_gradient.data_ptr(),
        selected_idx.data_ptr(),
        search_cost.data_ptr(),
        search_action.data_ptr(),
        search_gradient.data_ptr(),
        search_direction.data_ptr(),
        search_magnitudes.data_ptr(),
        armijo_threshold_c_1,
        curvature_threshold_c_2,
        strong_wolfe,
        approx_wolfe,
        n_linesearch,
        step_vec.data_ptr(),
        rho_buffer.data_ptr(),
        y_buffer.data_ptr(),
        s_buffer.data_ptr(),
        gram_buffer.data_ptr(),
        x_0.data_ptr(),
        grad_0.data_ptr(),
        epsilon,
        batch_size,
        history_m,
        v_dim,
        stable_mode,
        # Step scaling is skipped when the kernel receives a null action_step_max.
        action_step_max.data_ptr() if scale_step else 0,
        action_step_max.numel() if scale_step else 1,
    )
    launch_kernel(kernel_name, stream, config, kernel, *kernel_args)
