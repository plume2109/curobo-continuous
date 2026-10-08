/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "third_party/helper_math.h"
#include "common/block_warp_reductions.cuh"
#include "common/curobo_constants.h"
#include "common/math.cuh"
#include "lbfgs_step_helpers.cuh"
#include "line_search_helpers.cuh"

namespace curobo{
namespace optimization{

/** Most line-search points the fused kernel supports (one reduction slot per warp each). */
constexpr int MAX_FUSED_LINE_SEARCH = 8;

/**
 * @brief Sum over the 32 lanes of a full warp, returned to every lane.
 *
 * Lane 0 adds in the same order as warp_reduce (shuffle down by 16, 8, 4, 2, 1), so the
 * result matches block_reduce_sum bit for bit.
 */
__device__ __forceinline__ float full_warp_sum(float value)
{
  #pragma unroll
  for (int offset = curobo::common::warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(curobo::common::fullMask, value, offset);
  }
  return value;
}

/**
 * @brief Wolfe line search, compact L-BFGS step and next line-search points in one kernel.
 *
 * Runs, per problem, what kernel_line_search, kernel_lbfgs_step_compact and the PyTorch
 * x + scale * step that builds the next search points run one after another:
 *
 * 1. Line search over the evaluated points search_action (costs search_cost, gradients
 *    search_gradient) along search_direction: picks the exploration and selected points,
 *    writes them, and updates the best cost, best action and convergence state.
 * 2. Compact L-BFGS step at the exploration point (see kernel_lbfgs_step_compact), with
 *    the optional step scaling to action_step_max.
 * 3. Overwrites search_action with the next search points,
 *    exploration_action + search_magnitudes[i] * step.
 *
 * All loads, including every search point and the history of step 2, are issued before the
 * line-search reduction, so their latency overlaps it. Each thread reads and writes only its own element of every vector, so
 * search_direction may alias step_vec, and search_action is read before it is overwritten.
 *
 * Launch with blockDim.x = v_dim rounded up to a multiple of 32, one block per problem,
 * and compact_shared_memory_floats(v_dim, m) floats of dynamic shared memory.
 * n_linesearch <= MAX_FUSED_LINE_SEARCH.
 *
 * @tparam FIXED_M Compile-time history size (-1 for runtime)
 */
template<int FIXED_M = -1>
__global__ void kernel_line_search_lbfgs_step(
  // Best-solution and convergence tracking (see kernel_line_search)
  float *best_cost,                    // b
  float *best_action,                  // b x v_dim
  int32_t *best_iteration,             // b
  int32_t *current_iteration,          // b
  uint8_t *converged_global,           // b
  const int convergence_iteration,
  const float cost_delta_threshold,
  const float cost_relative_threshold,
  // Line search
  float *exploration_cost,             // b
  float *exploration_action,           // b x v_dim
  float *exploration_gradient,         // b x v_dim
  int32_t *exploration_idx,            // b x n_linesearch
  float *selected_cost,                // b
  float *selected_action,              // b x v_dim
  float *selected_gradient,            // b x v_dim
  int32_t *selected_idx,               // b x n_linesearch
  const float *search_cost,            // b x n_linesearch
  float *search_action,                // b x n_linesearch x v_dim, overwritten with next points
  const float *search_gradient,        // b x n_linesearch x v_dim
  const float *search_direction,       // b x v_dim
  const float *search_magnitudes,      // n_linesearch
  const float armijo_threshold_c_1,
  const float curvature_threshold_c_2,
  const bool strong_wolfe,
  const bool approx_wolfe,
  const int n_linesearch,
  // L-BFGS (see kernel_lbfgs_step_compact)
  float *step_vec,                     // b x v_dim
  float *rho_buffer,                   // m x b x 1
  float *y_buffer,                     // m x b x v_dim
  float *s_buffer,                     // m x b x v_dim
  float *gram_buffer,                  // b x 2 x m x m: S^T Y, then Y^T Y
  float *x_0,                          // b x v_dim
  float *grad_0,                       // b x v_dim
  const float epsilon, const int batchsize, const int lbfgs_history, const int v_dim,
  const bool stable_mode,
  const float *action_step_max,        // action_dim; nullptr skips step scaling
  const int action_dim)
{
  extern __shared__ __align__(16) float compact_smem[];
  __shared__ float line_search_partial[MAX_FUSED_LINE_SEARCH * curobo::common::warpSize];
  __shared__ int exploration_id_sh;
  __shared__ int selected_id_sh;
  __shared__ bool update_best_sh;

  constexpr bool is_compile_time = (FIXED_M > 0);
  const int m = is_compile_time ? FIXED_M : lbfgs_history;
  const int vector_stride = curobo::optimization::lbfgs::compact_vector_stride(v_dim);
  const int gram_stride = curobo::optimization::lbfgs::compact_gram_stride(m);
  const int padded_dim = (v_dim + 3) & ~3;
  const int newest = m - 1;

  float* s_sh = compact_smem;
  float* y_sh = &s_sh[m * vector_stride];
  float* g_sh = &y_sh[m * vector_stride];
  float* sy_sh = &g_sh[vector_stride];
  float* yy_sh = &sy_sh[m * gram_stride];
  float* sg_sh = &yy_sh[m * gram_stride];
  float* yg_sh = &sg_sh[m];
  float* rho_sh = &yg_sh[m];
  float* coef_s = &rho_sh[m];
  float* coef_y = &coef_s[m];
  float* gamma_sh = &coef_y[m];
  float* data = &gamma_sh[1];
  float* result = &data[32];

  const int batch = blockIdx.x; // one block per problem
  const int tid = threadIdx.x;
  const int lane = tid % curobo::common::warpSize;
  const int warp = tid / curobo::common::warpSize;
  const int num_warps = blockDim.x / curobo::common::warpSize;
  const bool owns_element = tid < v_dim;
  const int64_t history_stride = (int64_t)batchsize * v_dim;
  const int64_t element = (int64_t)batch * v_dim + tid;
  const int64_t search_offset = (int64_t)batch * n_linesearch * v_dim + tid;
  float* gram_sy = &gram_buffer[(int64_t)batch * 2 * m * m];
  float* gram_yy = &gram_sy[m * m];

  // Phase 0: loads. Every search point, the reference point and the scalars the line search
  // needs are loaded before the first synchronization, so the selected point comes from
  // registers instead of a load that waits for the Wolfe decision.
  const float direction = owns_element ? search_direction[element] : 0.0f;
  float search_x[MAX_FUSED_LINE_SEARCH];
  float search_g[MAX_FUSED_LINE_SEARCH];
  #pragma unroll
  for (int i = 0; i < MAX_FUSED_LINE_SEARCH; i++) {
    search_x[i] = 0.0f;
    search_g[i] = 0.0f;
    if (owns_element && i < n_linesearch) {
      search_x[i] = search_action[search_offset + (int64_t)i * v_dim];
      search_g[i] = search_gradient[search_offset + (int64_t)i * v_dim];
    }
  }
  const float x_ref = owns_element ? x_0[element] : 0.0f;
  const float g_ref = owns_element ? grad_0[element] : 0.0f;

  const bool search_lane = warp == 0 && lane < n_linesearch;
  const float lane_cost = search_lane ? search_cost[batch * n_linesearch + lane] : 0.0f;
  const float lane_magnitude = search_lane ? search_magnitudes[lane] : 0.0f;
  float best_cost_value = 0.0f;
  int best_iteration_value = 0;
  int current_iteration_value = 0;
  if (tid == 0) {
    best_cost_value = best_cost[batch];
    best_iteration_value = best_iteration[batch];
    current_iteration_value = current_iteration[batch];
  }

  // Directional derivatives g_i^T d of the search points, one partial sum per warp.
  #pragma unroll
  for (int i = 0; i < MAX_FUSED_LINE_SEARCH; i++) {
    if (i < n_linesearch) {
      const float partial = full_warp_sum(search_g[i] * direction);
      if (lane == 0) {
        line_search_partial[i * curobo::common::warpSize + warp] = partial;
      }
    }
  }

  // History moves up by one slot; the new pair goes last once the line search has picked q.
  if (owns_element) {
    #pragma unroll
    for (int i = 1; i < m; i++) {
      s_sh[(i - 1) * vector_stride + tid] = s_buffer[i * history_stride + element];
      y_sh[(i - 1) * vector_stride + tid] = y_buffer[i * history_stride + element];
    }
  } else if (tid < padded_dim) {
    // Zero padding read by the float4 dot products.
    for (int i = 0; i < m; i++) {
      s_sh[i * vector_stride + tid] = 0.0f;
      y_sh[i * vector_stride + tid] = 0.0f;
    }
    g_sh[tid] = 0.0f;
  }
  const int kept = newest > 0 ? newest : 1; // divisor guard for m = 1, where nothing is kept
  for (int k = tid; k < newest * newest; k += blockDim.x) {
    const int i = k / kept;
    const int j = k % kept;
    sy_sh[i * gram_stride + j] = gram_sy[(i + 1) * m + j + 1];
    yy_sh[i * gram_stride + j] = gram_yy[(i + 1) * m + j + 1];
  }
  if (tid < newest) {
    rho_sh[tid] = rho_buffer[(tid + 1) * batchsize + batch];
  }
  __syncthreads();

  // Phase 1: warp 0 finishes the reductions, checks the Wolfe conditions and updates the
  // costs and convergence state (as kernel_line_search does).
  if (warp == 0) {
    float g_step = 0.0f;
    for (int i = 0; i < n_linesearch; i++) {
      float partial = lane < num_warps ?
          line_search_partial[i * curobo::common::warpSize + lane] : 0.0f;
      partial = full_warp_sum(partial);
      if (lane == i) {
        g_step = partial;
      }
    }
    const float g_step_0 = __shfl_sync(curobo::common::fullMask, g_step, 0);
    const float cost_0 = __shfl_sync(curobo::common::fullMask, lane_cost, 0);

    bool wolfe_1 = false;
    bool wolfe = false;
    bool wolfe_2 = false;
    line_search::evaluate_wolfe_conditions(search_lane, lane_magnitude, cost_0, lane_cost,
        g_step, g_step_0, armijo_threshold_c_1, curvature_threshold_c_2, strong_wolfe,
        approx_wolfe, wolfe, wolfe_1, wolfe_2);
    int exploration_id = 0;
    int selected_id = 0;
    line_search::compute_wolfe_indices(wolfe_1, wolfe, search_lane, approx_wolfe, strong_wolfe,
                                       exploration_id, selected_id);
    const float exploration_cost_value =
        __shfl_sync(curobo::common::fullMask, lane_cost, exploration_id);
    const float selected_cost_value =
        __shfl_sync(curobo::common::fullMask, lane_cost, selected_id);

    if (lane == 0) {
      exploration_id_sh = exploration_id;
      selected_id_sh = selected_id;
      exploration_cost[batch] = exploration_cost_value;
      selected_cost[batch] = selected_cost_value;

      current_iteration_value++;
      bool update_best = false;
      bool converged = false;
      line_search::check_best_convergence(best_cost_value, selected_cost_value,
          cost_delta_threshold, cost_relative_threshold, current_iteration_value,
          convergence_iteration, best_iteration_value, update_best, converged);
      update_best_sh = update_best;
      converged_global[batch] = converged;
      best_iteration[batch] = best_iteration_value;
      current_iteration[batch] = current_iteration_value;
      if (update_best) {
        best_cost[batch] = selected_cost_value;
      }
    }
  }
  __syncthreads();

  // Phase 2: write the line-search results; the exploration point is the L-BFGS point q.
  const int exploration_id = exploration_id_sh;
  const int selected_id = selected_id_sh;
  float q_t = 0.0f, gq = 0.0f, selected_x = 0.0f, selected_g = 0.0f;
  #pragma unroll
  for (int i = 0; i < MAX_FUSED_LINE_SEARCH; i++) {
    if (i == exploration_id) {
      q_t = search_x[i];
      gq = search_g[i];
    }
    if (i == selected_id) {
      selected_x = search_x[i];
      selected_g = search_g[i];
    }
  }
  if (owns_element) {
    exploration_action[element] = q_t;
    exploration_gradient[element] = gq;
    selected_action[element] = selected_x;
    selected_gradient[element] = selected_g;
    if (update_best_sh) {
      best_action[element] = selected_x;
    }
    s_sh[newest * vector_stride + tid] = q_t - x_ref;
    y_sh[newest * vector_stride + tid] = gq - g_ref;
    g_sh[tid] = gq;
  }
  if (tid < n_linesearch) {
    exploration_idx[batch * n_linesearch + tid] = exploration_id;
    selected_idx[batch * n_linesearch + tid] = selected_id;
  }
  __syncthreads();

  // Phase 3: dot products with the new pair and the new gradient.
  curobo::optimization::lbfgs::compute_new_gram_entries(
      m, vector_stride, gram_stride, padded_dim, s_sh, y_sh, g_sh, sy_sh, yy_sh, sg_sh, yg_sh,
      rho_sh, stable_mode);
  __syncthreads();

  // Phase 4: one warp runs the two-loop recursion on scalars.
  if (warp == 0) {
    curobo::optimization::lbfgs::compact_two_loop_coefficients<FIXED_M>(
        m, gram_stride, sy_sh, yy_sh, sg_sh, yg_sh, rho_sh, epsilon, stable_mode,
        coef_s, coef_y, gamma_sh);
  }
  __syncthreads();

  // Phase 5: step = -(gamma * g + sum_i coef_s_i * s_i + coef_y_i * y_i).
  float step = 0.0f;
  if (owns_element) {
    float d = gamma_sh[0] * g_sh[tid];
    #pragma unroll
    for (int i = 0; i < m; i++) {
      d = fmaf(coef_s[i], s_sh[i * vector_stride + tid], d);
      d = fmaf(coef_y[i], y_sh[i * vector_stride + tid], d);
    }
    step = -d;
  }

  if (action_step_max != nullptr)
  {
    step = curobo::optimization::lbfgs::scale_step_to_action_step_max(
        step, action_step_max, action_dim, v_dim, &data[0], &result[0]);
  }

  // Global writes: step, next search points, reference point, history, Gram matrices, rho.
  if (owns_element) {
    step_vec[element] = step;
    for (int i = 0; i < n_linesearch; i++) {
      search_action[search_offset + (int64_t)i * v_dim] = fmaf(search_magnitudes[i], step, q_t);
    }
    x_0[element] = q_t;
    grad_0[element] = gq;
    #pragma unroll
    for (int i = 0; i < m; i++) {
      s_buffer[i * history_stride + element] = s_sh[i * vector_stride + tid];
      y_buffer[i * history_stride + element] = y_sh[i * vector_stride + tid];
    }
  }
  for (int k = tid; k < m * m; k += blockDim.x) {
    const int i = k / m;
    const int j = k % m;
    gram_sy[k] = sy_sh[i * gram_stride + j];
    gram_yy[k] = yy_sh[i * gram_stride + j];
  }
  if (tid < m) {
    rho_buffer[tid * batchsize + batch] = rho_sh[tid];
  }
}

} // namespace optimization
} // namespace curobo
