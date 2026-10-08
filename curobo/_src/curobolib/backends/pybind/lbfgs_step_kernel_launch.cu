/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda.h>
#include <torch/extension.h>
#include <vector>
#include <array>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <common/torch_cuda_utils.h>
#include "lbfgs_step_kernel.cuh"
#include "lbfgs_line_search_step_kernel.cuh"

namespace curobo{
  namespace optimization{





// ============================================================================
// LAUNCH HELPER FUNCTIONS AND STRUCTS
// ============================================================================

using LBFGSStepKernel = void(*)(float*, float*, float*, float*, float*, float*, float*,
                                const float*, float, int, int, int, bool, const float*, int);
using LBFGSCompactKernel = void(*)(float*, float*, float*, float*, float*, const float*, float*,
                                   float*, const float*, float, int, int, int, bool,
                                   const float*, int);

// Kernels specialized for common history sizes; others use the runtime-history kernels.
template<int M>
inline void select_specialized_kernels(LBFGSCompactKernel& compact, LBFGSStepKernel& global)
{
    compact = kernel_lbfgs_step_compact<float, M>;
    global = kernel_lbfgs_step<float, false, M>;
}

inline void select_lbfgs_kernels(int history_m, LBFGSCompactKernel& compact,
                                 LBFGSStepKernel& global)
{
    switch (history_m) {
        case 5:  select_specialized_kernels<5>(compact, global); return;
        case 6:  select_specialized_kernels<6>(compact, global); return;
        case 7:  select_specialized_kernels<7>(compact, global); return;
        case 15: select_specialized_kernels<15>(compact, global); return;
        case 24: select_specialized_kernels<24>(compact, global); return;
        case 27: select_specialized_kernels<27>(compact, global); return;
        case 28: select_specialized_kernels<28>(compact, global); return;
        case 31: select_specialized_kernels<31>(compact, global); return;
        default:
            compact = kernel_lbfgs_step_compact<float>;
            global = kernel_lbfgs_step<float, false>;
    }
}

std::vector<torch::Tensor>
launch_lbfgs_step(torch::Tensor step_vec, torch::Tensor rho_buffer,
                torch::Tensor y_buffer, torch::Tensor s_buffer, torch::Tensor gram_buffer,
                torch::Tensor q, torch::Tensor grad_q, torch::Tensor x_0, torch::Tensor grad_0,
                const float epsilon, const int batch_size, const int history_m,
                const int v_dim, const bool stable_mode, const bool use_shared_buffers,
                torch::Tensor action_step_max, const bool scale_step)
{
    // Validate all inputs
    curobo::common::validate_cuda_input(step_vec, "step_vec");
    curobo::common::validate_cuda_input(rho_buffer, "rho_buffer");
    curobo::common::validate_cuda_input(y_buffer, "y_buffer");
    curobo::common::validate_cuda_input(s_buffer, "s_buffer");
    curobo::common::validate_cuda_input(gram_buffer, "gram_buffer");
    curobo::common::validate_cuda_input(q, "q");
    curobo::common::validate_cuda_input(x_0, "x_0");
    curobo::common::validate_cuda_input(grad_0, "grad_0");
    curobo::common::validate_cuda_input(grad_q, "grad_q");
    if (scale_step) {
        curobo::common::validate_cuda_input(action_step_max, "action_step_max");
    }
    // Step scaling is skipped when the kernel receives a null action_step_max.
    const float* action_step_max_ptr = scale_step ? action_step_max.data_ptr<float>() : nullptr;
    const int action_dim = scale_step ? static_cast<int>(action_step_max.numel()) : 1;

    assert(v_dim < 1024 && history_m < 32);

    LBFGSCompactKernel compact_kernel;
    LBFGSStepKernel global_kernel;
    select_lbfgs_kernels(history_m, compact_kernel, global_kernel);

    const int max_shared_base = 48000;
    const int max_shared_allowed = 65536;
    const int compact_smem_size =
        lbfgs::compact_shared_memory_floats(v_dim, history_m) * sizeof(float);

    bool use_compact = use_shared_buffers && compact_smem_size <= max_shared_allowed;
    if (use_compact && compact_smem_size > max_shared_base) {
        use_compact = cudaFuncSetAttribute(
            compact_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            compact_smem_size) == cudaSuccess;
    }

    cudaStream_t stream = curobo::common::get_cuda_stream();

    if (use_compact) {
        const int threads = ((v_dim + 31) / 32) * 32;
        compact_kernel<<<batch_size, threads, compact_smem_size, stream>>>(
            step_vec.data_ptr<float>(),
            rho_buffer.data_ptr<float>(),
            y_buffer.data_ptr<float>(),
            s_buffer.data_ptr<float>(),
            gram_buffer.data_ptr<float>(),
            q.data_ptr<float>(),
            x_0.data_ptr<float>(),
            grad_0.data_ptr<float>(),
            grad_q.data_ptr<float>(),
            epsilon, batch_size, history_m, v_dim, stable_mode,
            action_step_max_ptr, action_dim);
    } else {
        const int basic_smem_size = history_m * v_dim * sizeof(float);
        global_kernel<<<batch_size, v_dim, basic_smem_size, stream>>>(
            step_vec.data_ptr<float>(),
            rho_buffer.data_ptr<float>(),
            y_buffer.data_ptr<float>(),
            s_buffer.data_ptr<float>(),
            q.data_ptr<float>(),
            x_0.data_ptr<float>(),
            grad_0.data_ptr<float>(),
            grad_q.data_ptr<float>(),
            epsilon, batch_size, history_m, v_dim, stable_mode,
            action_step_max_ptr, action_dim);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return { step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0 };
}

using LineSearchLBFGSKernel = void(*)(
    float*, float*, int32_t*, int32_t*, uint8_t*, int, float, float,
    float*, float*, float*, int32_t*, float*, float*, float*, int32_t*,
    const float*, float*, const float*, const float*, const float*, float, float, bool, bool, int,
    float*, float*, float*, float*, float*, float*, float*,
    float, int, int, int, bool, const float*, int);

inline LineSearchLBFGSKernel select_line_search_lbfgs_kernel(int history_m)
{
    switch (history_m) {
        case 5:  return kernel_line_search_lbfgs_step<5>;
        case 6:  return kernel_line_search_lbfgs_step<6>;
        case 7:  return kernel_line_search_lbfgs_step<7>;
        case 15: return kernel_line_search_lbfgs_step<15>;
        case 24: return kernel_line_search_lbfgs_step<24>;
        case 27: return kernel_line_search_lbfgs_step<27>;
        case 28: return kernel_line_search_lbfgs_step<28>;
        case 31: return kernel_line_search_lbfgs_step<31>;
        default: return kernel_line_search_lbfgs_step<>;
    }
}

void launch_line_search_lbfgs_step(
    torch::Tensor best_cost, torch::Tensor best_action, torch::Tensor best_iteration,
    torch::Tensor current_iteration, torch::Tensor converged_global,
    const int convergence_iteration, const float cost_delta_threshold,
    const float cost_relative_threshold,
    torch::Tensor exploration_cost, torch::Tensor exploration_action,
    torch::Tensor exploration_gradient, torch::Tensor exploration_idx,
    torch::Tensor selected_cost, torch::Tensor selected_action,
    torch::Tensor selected_gradient, torch::Tensor selected_idx,
    const torch::Tensor search_cost, torch::Tensor search_action,
    const torch::Tensor search_gradient, const torch::Tensor search_direction,
    const torch::Tensor search_magnitudes, const float armijo_threshold_c_1,
    const float curvature_threshold_c_2, const bool strong_wolfe, const bool approx_wolfe,
    const int n_linesearch,
    torch::Tensor step_vec, torch::Tensor rho_buffer, torch::Tensor y_buffer,
    torch::Tensor s_buffer, torch::Tensor gram_buffer, torch::Tensor x_0, torch::Tensor grad_0,
    const float epsilon, const int batch_size, const int history_m, const int v_dim,
    const bool stable_mode, torch::Tensor action_step_max, const bool scale_step)
{
    curobo::common::validate_cuda_input(search_action, "search_action");
    curobo::common::validate_cuda_input(search_gradient, "search_gradient");
    curobo::common::validate_cuda_input(step_vec, "step_vec");
    curobo::common::validate_cuda_input(gram_buffer, "gram_buffer");
    if (scale_step) {
        curobo::common::validate_cuda_input(action_step_max, "action_step_max");
    }
    assert(v_dim < 1024 && history_m < 32 && n_linesearch <= MAX_FUSED_LINE_SEARCH);

    const float* action_step_max_ptr = scale_step ? action_step_max.data_ptr<float>() : nullptr;
    const int action_dim = scale_step ? static_cast<int>(action_step_max.numel()) : 1;

    LineSearchLBFGSKernel kernel = select_line_search_lbfgs_kernel(history_m);
    const int smem_size = lbfgs::compact_shared_memory_floats(v_dim, history_m) * sizeof(float);
    if (smem_size > 48000) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    const int threads = ((v_dim + 31) / 32) * 32;
    cudaStream_t stream = curobo::common::get_cuda_stream();

    kernel<<<batch_size, threads, smem_size, stream>>>(
        best_cost.data_ptr<float>(), best_action.data_ptr<float>(),
        best_iteration.data_ptr<int32_t>(), current_iteration.data_ptr<int32_t>(),
        converged_global.data_ptr<uint8_t>(), convergence_iteration, cost_delta_threshold,
        cost_relative_threshold,
        exploration_cost.data_ptr<float>(), exploration_action.data_ptr<float>(),
        exploration_gradient.data_ptr<float>(), exploration_idx.data_ptr<int32_t>(),
        selected_cost.data_ptr<float>(), selected_action.data_ptr<float>(),
        selected_gradient.data_ptr<float>(), selected_idx.data_ptr<int32_t>(),
        search_cost.data_ptr<float>(), search_action.data_ptr<float>(),
        search_gradient.data_ptr<float>(), search_direction.data_ptr<float>(),
        search_magnitudes.data_ptr<float>(), armijo_threshold_c_1, curvature_threshold_c_2,
        strong_wolfe, approx_wolfe, n_linesearch,
        step_vec.data_ptr<float>(), rho_buffer.data_ptr<float>(), y_buffer.data_ptr<float>(),
        s_buffer.data_ptr<float>(), gram_buffer.data_ptr<float>(), x_0.data_ptr<float>(),
        grad_0.data_ptr<float>(),
        epsilon, batch_size, history_m, v_dim, stable_mode, action_step_max_ptr, action_dim);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
}
