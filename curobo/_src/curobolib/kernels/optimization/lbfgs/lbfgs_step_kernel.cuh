/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */



#include "third_party/helper_math.h"
#include "common/block_warp_reductions.cuh"
#include "common/curobo_constants.h"
#include "common/math.cuh"
#include "lbfgs_step_helpers.cuh"

namespace curobo{
namespace optimization{


template<typename ScalarType, bool rolled_ys, int FIXED_M = -1>
__global__ void kernel_lbfgs_step(
  ScalarType *step_vec,     // b x 175
  ScalarType *rho_buffer,   // m x b x 1
  ScalarType *y_buffer,     // m x b x 175
  ScalarType *s_buffer,     // m x b x 175
  ScalarType *q,            // b x 175
  ScalarType *x_0,          // b x 175
  ScalarType *grad_0,       // b x 175
  const ScalarType *grad_q, // b x 175
  const float epsilon, const int batchsize, const int m, const int v_dim,
  const bool stable_mode = false,                // s_buffer and y_buffer are not rolled by default
  const ScalarType *action_step_max = nullptr, // action_dim; nullptr skips step scaling
  const int action_dim = 1)
{
  extern __shared__ float alpha_buffer_sh[];

  // Constexpr logic for compile-time vs runtime
  constexpr bool is_compile_time = (FIXED_M > 0);
  const int effective_m = is_compile_time ? FIXED_M : m;

  __shared__ ScalarType data[32];
  // temporary buffer needed for block-wide reduction
  __shared__ ScalarType result;
  // result of the reduction or vector-vector dot product
  int batch = blockIdx.x; // one block per batch

  if (threadIdx.x >= v_dim)
    return;

  // Load state and compute y, s differences; also returns gq to avoid redundant global load
  ScalarType y, s, gq;
  curobo::optimization::lbfgs::load_lbfgs_state_and_compute_differences(
      batch, threadIdx.x, v_dim, grad_q, q, grad_0, x_0, y, s, gq);

  // Compute y^T * s for rho calculation
  curobo::common::block_reduce_sum(y * s, v_dim, &data[0], &result);
  ScalarType numerator = result;

  // Update history buffers
  curobo::optimization::lbfgs::update_lbfgs_history_buffers<ScalarType, rolled_ys, FIXED_M>(
      batch, threadIdx.x, batchsize, v_dim, effective_m, y, s, y_buffer, s_buffer);

  // Update rho buffer
  curobo::optimization::lbfgs::update_rho_buffer(
      batch, threadIdx.x, batchsize, effective_m, numerator, stable_mode, rho_buffer);
  // The two-loop recursion reads rho values written by other threads.
  __syncthreads();

  ////////////////////
  // L-BFGS two-loop algorithm
  ////////////////////

  // gq already loaded from load_lbfgs_state_and_compute_differences (no redundant global read)

  // Backward pass (first loop)
  curobo::optimization::lbfgs::lbfgs_backward_pass<ScalarType, FIXED_M>(
      threadIdx.x, batch, batchsize, v_dim, effective_m, gq, s_buffer, y_buffer,
      rho_buffer, alpha_buffer_sh, &data[0], &result);

  // Reload y from history buffer to shorten its live range (cheap L1 hit, saves a register
  // across the entire backward pass)
  ScalarType y_latest = y_buffer[(effective_m - 1) * batchsize * v_dim + batch * v_dim + threadIdx.x];

  // Compute L-BFGS scaling factor and apply it
  curobo::optimization::lbfgs::compute_lbfgs_scaling(
      y_latest, numerator, epsilon, stable_mode, gq, v_dim, &data[0], &result);

  // Forward pass (second loop)
  curobo::optimization::lbfgs::lbfgs_forward_pass<ScalarType, FIXED_M>(
      threadIdx.x, batch, batchsize, v_dim, effective_m, gq, s_buffer, y_buffer,
      rho_buffer, alpha_buffer_sh, &data[0], &result);

  ScalarType step = -gq;
  if (action_step_max != nullptr)
  {
    step = curobo::optimization::lbfgs::scale_step_to_action_step_max(
        step, action_step_max, action_dim, v_dim, &data[0], &result);
  }

  // Store final step direction
  step_vec[batch * v_dim + threadIdx.x] = step;
}



/**
 * @brief Compact L-BFGS step: one block per problem, history and Gram matrices in shared memory.
 *
 * Computes the same step as kernel_lbfgs_step, but from dot products: only the 5 * m - 1 dot
 * products that involve the new (s, y) pair or the new gradient are computed (in one parallel
 * pass), S^T Y and Y^T Y persist across iterations in gram_buffer, and the two-loop recursion
 * runs on scalars in one warp. Three block synchronizations, plus two for step scaling.
 *
 * Launch with blockDim.x = v_dim rounded up to a multiple of 32 and
 * compact_shared_memory_floats(v_dim, m) floats of dynamic shared memory.
 *
 * gram_buffer must hold S^T Y and Y^T Y of the s_buffer and y_buffer passed in
 * (zero for zeroed history); the kernel updates it with the history.
 */
template<typename ScalarType, int FIXED_M = -1>
__global__ void kernel_lbfgs_step_compact(
  ScalarType *step_vec,     // b x v_dim
  ScalarType *rho_buffer,   // m x b x 1
  ScalarType *y_buffer,     // m x b x v_dim
  ScalarType *s_buffer,     // m x b x v_dim
  ScalarType *gram_buffer,  // b x 2 x m x m: S^T Y, then Y^T Y
  const ScalarType *q,      // b x v_dim
  ScalarType *x_0,          // b x v_dim
  ScalarType *grad_0,       // b x v_dim
  const ScalarType *grad_q, // b x v_dim
  const float epsilon, const int batchsize, const int lbfgs_history, const int v_dim,
  const bool stable_mode = false,
  const ScalarType *action_step_max = nullptr, // action_dim; nullptr skips step scaling
  const int action_dim = 1)
{
  extern __shared__ __align__(16) float compact_smem[];

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
  const bool owns_element = tid < v_dim;
  const int64_t history_stride = (int64_t)batchsize * v_dim;
  const int64_t element = (int64_t)batch * v_dim + tid;
  ScalarType* gram_sy = &gram_buffer[(int64_t)batch * 2 * m * m];
  ScalarType* gram_yy = &gram_sy[m * m];

  // Phase 1: loads only. The history moves up by one slot in shared memory and the new
  // (s, y) pair goes last. Global writes wait until the end of the kernel: a global store
  // between loads of the same buffer would serialize the loads.
  ScalarType gq = 0.0f, q_t = 0.0f;
  if (owns_element) {
    gq = grad_q[element];
    q_t = q[element];
    const ScalarType gq_0 = grad_0[element];
    const ScalarType q_0 = x_0[element];

    #pragma unroll
    for (int i = 1; i < m; i++) {
      s_sh[(i - 1) * vector_stride + tid] = s_buffer[i * history_stride + element];
      y_sh[(i - 1) * vector_stride + tid] = y_buffer[i * history_stride + element];
    }
    s_sh[newest * vector_stride + tid] = q_t - q_0;
    y_sh[newest * vector_stride + tid] = gq - gq_0;
    g_sh[tid] = gq;
  } else if (tid < padded_dim) {
    // Zero padding read by the float4 dot products.
    for (int i = 0; i < m; i++) {
      s_sh[i * vector_stride + tid] = 0.0f;
      y_sh[i * vector_stride + tid] = 0.0f;
    }
    g_sh[tid] = 0.0f;
  }

  // Gram entries between kept pairs move up by one row and one column.
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

  // Phase 2: dot products with the new pair and the new gradient.
  curobo::optimization::lbfgs::compute_new_gram_entries(
      m, vector_stride, gram_stride, padded_dim, s_sh, y_sh, g_sh, sy_sh, yy_sh, sg_sh, yg_sh,
      rho_sh, stable_mode);
  __syncthreads();

  // Phase 3: one warp runs the two-loop recursion on scalars.
  if (tid < curobo::common::warpSize) {
    curobo::optimization::lbfgs::compact_two_loop_coefficients<FIXED_M>(
        m, gram_stride, sy_sh, yy_sh, sg_sh, yg_sh, rho_sh, epsilon, stable_mode,
        coef_s, coef_y, gamma_sh);
  }
  __syncthreads();

  // Phase 4: step = -(gamma * g + sum_i coef_s_i * s_i + coef_y_i * y_i).
  ScalarType step = 0.0f;
  if (owns_element) {
    ScalarType d = gamma_sh[0] * g_sh[tid];
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

  // Global writes: step, reference point, history, Gram matrices and rho.
  if (owns_element) {
    step_vec[element] = step;
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
