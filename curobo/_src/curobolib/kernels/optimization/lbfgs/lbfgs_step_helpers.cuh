/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "common/curobo_constants.h"
#include "common/math.cuh"
#include "common/block_warp_reductions.cuh"
#include "third_party/helper_math.h"

namespace curobo{
namespace optimization{
namespace lbfgs{

    /**
     * @brief Loads current L-BFGS state and computes y and s vectors
     *
     * This function handles the common pattern of loading current gradients and positions,
     * computing the difference vectors y (gradient difference) and s (position difference),
     * and updating the state arrays. Also returns the current gradient value to avoid
     * a redundant global memory load in the caller.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @param batch Current batch index
     * @param thread_idx Current thread index within batch
     * @param v_dim Optimization dimension size
     * @param grad_q Current gradient array
     * @param q Current position array
     * @param grad_0 Previous gradient array (input/output)
     * @param x_0 Previous position array (input/output)
     * @param y Output y vector (gradient difference)
     * @param s Output s vector (position difference)
     * @param gq_out Output current gradient value for reuse by caller
     */
    template<typename ScalarType>
    __device__ __forceinline__ void load_lbfgs_state_and_compute_differences(
        const int batch,
        const int thread_idx,
        const int v_dim,
        const ScalarType* grad_q,
        const ScalarType* q,
        ScalarType* grad_0,
        ScalarType* x_0,
        ScalarType& y,
        ScalarType& s,
        ScalarType& gq_out)
    {
        const uint32_t batch_vdim_tidx = batch * v_dim + thread_idx;

        // Load current state
        ScalarType gq = grad_q[batch_vdim_tidx];
        ScalarType q_t = q[batch_vdim_tidx];

        // Load previous state
        ScalarType gq_0 = grad_0[batch_vdim_tidx];
        ScalarType q_0 = x_0[batch_vdim_tidx];

        // Compute differences
        y = gq - gq_0;
        s = q_t - q_0;

        // Update previous state for next iteration
        grad_0[batch_vdim_tidx] = gq;
        x_0[batch_vdim_tidx] = q_t;

        // Return gradient value for reuse (avoids redundant global load)
        gq_out = gq;
    }

    /**
     * @brief Updates L-BFGS history buffers with new y and s vectors
     *
     * This function handles the rolling/shifting of history buffers and updates
     * them with new y and s vectors. Supports both rolled and non-rolled buffer modes.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @tparam rolled_ys Whether buffers are pre-rolled
     * @param batch Current batch index
     * @param thread_idx Current thread index within batch
     * @param batchsize Total batch size
     * @param v_dim Optimization dimension size
     * @param history_m History buffer size
     * @param y New y vector value
     * @param s New s vector value
     * @param y_buffer History buffer for y vectors (input/output)
     * @param s_buffer History buffer for s vectors (input/output)
     */
    template<typename ScalarType, bool rolled_ys, int FIXED_M = -1>
    __device__ __forceinline__ void update_lbfgs_history_buffers(
        const int batch,
        const int thread_idx,
        const int batchsize,
        const int v_dim,
        const int history_m,
        const ScalarType y,
        const ScalarType s,
        ScalarType* y_buffer,
        ScalarType* s_buffer)
    {
        const uint32_t batch_vdim_tidx = batch * v_dim + thread_idx;

        if constexpr (!rolled_ys) {
            constexpr bool is_compile_time = (FIXED_M > 0);
            if constexpr (is_compile_time) {
                for (int i = 1; i < FIXED_M; i++) {
                    const uint32_t src_idx = i * batchsize * v_dim + batch_vdim_tidx;
                    const uint32_t dst_idx = (i - 1) * batchsize * v_dim + batch_vdim_tidx;

                    s_buffer[dst_idx] = s_buffer[src_idx];
                    y_buffer[dst_idx] = y_buffer[src_idx];
                }
            }
            else {


            // Shift old values: move [1, 2, ..., m-1] to [0, 1, ..., m-2]
            for (int i = 1; i < history_m; i++) {
                const uint32_t src_idx = i * batchsize * v_dim + batch_vdim_tidx;
                const uint32_t dst_idx = (i - 1) * batchsize * v_dim + batch_vdim_tidx;

                s_buffer[dst_idx] = s_buffer[src_idx];
                y_buffer[dst_idx] = y_buffer[src_idx];
            }
            }
        }

        // Store new values at the end
        const uint32_t new_idx = (history_m - 1) * batchsize * v_dim + batch_vdim_tidx;
        s_buffer[new_idx] = s;
        y_buffer[new_idx] = y;
    }

    /**
     * @brief Updates rho buffer with new curvature information
     *
     * This function computes the new rho value (1 / (y^T * s)) and updates
     * the rho buffer, handling both stability checks and buffer rolling.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @param batch Current batch index
     * @param thread_idx Current thread index within batch
     * @param batchsize Total batch size
     * @param history_m History buffer size
     * @param denominator Computed y^T * s value
     * @param stable_mode Whether to apply stability checks
     * @param rho_buffer Rho history buffer (input/output)
     */
    template<typename ScalarType>
    __device__ __forceinline__ void update_rho_buffer(
        const int batch,
        const int thread_idx,
        const int batchsize,
        const int history_m,
        const ScalarType denominator,
        const bool stable_mode,
        ScalarType* rho_buffer)
    {
        // Shift old rho values (threads < history_m - 1)
        if (thread_idx < history_m - 1) {
            ScalarType rho = rho_buffer[(thread_idx + 1) * batchsize + batch];
            rho_buffer[thread_idx * batchsize + batch] = rho;
        }

        // Compute and store new rho value (thread == history_m - 1)
        if (thread_idx == history_m - 1) {
            ScalarType rho = 1.0 / denominator;

            // Stability check: avoid division by zero
            if (stable_mode && (denominator <= 0.0)) {
                rho = 0.0;
            }

            rho_buffer[thread_idx * batchsize + batch] = rho;
        }
    }

    /**
     * @brief Performs the backward pass of the L-BFGS two-loop algorithm
     *
     * This function implements the first loop of the L-BFGS two-loop recursion,
     * computing alpha values and updating the search direction.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @param thread_idx Current thread index within batch
     * @param batch Current batch index
     * @param batchsize Total batch size
     * @param v_dim Optimization dimension size
     * @param history_m History buffer size
     * @param gq Current search direction (input/output)
     * @param s_buffer History buffer for s vectors
     * @param y_buffer History buffer for y vectors
     * @param rho_buffer History buffer for rho values
     * @param alpha_buffer Output buffer for alpha values
     * @param data Temporary reduction buffer
     * @param result Reduction result buffer
     */
    template<typename ScalarType, int HISTORY_M = -1>
    __device__ __forceinline__ void lbfgs_backward_pass(
        const int thread_idx,
        const int batch,
        const int batchsize,
        const int v_dim,
        const int history_m,
        ScalarType& gq,
        const ScalarType* s_buffer,
        const ScalarType* y_buffer,
        const ScalarType* rho_buffer,
        ScalarType* alpha_buffer,
        ScalarType* data,
        float* result)
    {
        constexpr bool is_compile_time = (HISTORY_M > 0);

        if constexpr (is_compile_time) {
            for (int i = HISTORY_M - 1; i > -1; i--) {
                float current_s = s_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];
                float current_y = y_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];
                float current_rho = rho_buffer[i * batchsize + batch];
                // Compute s^T * gq
                curobo::common::block_reduce_sum(
                    gq * current_s,
                    v_dim, &data[0], result);

                // alpha_i = rho_i * s_i^T * gq
                float current_alpha = result[0] * current_rho;

                // gq = gq - alpha_i * y_i
                gq = gq - current_alpha * current_y;

                if (thread_idx == 0)
                {
                    alpha_buffer[i] = current_alpha;
                }

            }
        }
        else {
            for (int i = history_m - 1; i > -1; i--) {
                float current_s = s_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];
                float current_y = y_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];
                float current_rho = rho_buffer[i * batchsize + batch];

                // Compute s^T * gq
                curobo::common::block_reduce_sum(
                    gq * current_s,
                    v_dim, &data[0], result);

                // alpha_i = rho_i * s_i^T * gq
                float current_alpha = result[0] * current_rho;


                // gq = gq - alpha_i * y_i
                gq = gq - current_alpha * current_y;

                if (thread_idx == 0)
                {
                  alpha_buffer[i] = current_alpha;
                }

            }

        }
    }

    /**
     * @brief Computes the L-BFGS scaling factor (gamma)
     *
     * This function computes the Hessian scaling factor gamma = (s^T * y) / (y^T * y)
     * and applies it to the search direction, with stability checks.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @param y Latest y vector value
     * @param numerator Precomputed y^T * s value
     * @param epsilon Stability epsilon value
     * @param stable_mode Whether to apply stability checks
     * @param gq Search direction (input/output)
     * @param v_dim Optimization dimension size
     * @param data Temporary reduction buffer
     * @param result Reduction result buffer
     */
    template<typename ScalarType>
    __device__ __forceinline__ void compute_lbfgs_scaling(
        const ScalarType y,
        const ScalarType numerator,
        const float epsilon,
        const bool stable_mode,
        ScalarType& gq,
        const int v_dim,
        ScalarType* data,
        float* result)
    {
        // Compute y^T * y
        curobo::common::block_reduce_sum(y * y, v_dim, &data[0], result);
        ScalarType denominator = result[0];

        // Compute gamma = (s^T * y) / (y^T * y)
        // Negative gamma (curvature condition violated) is clamped to 0 by relu below,
        // which lets the forward pass reconstruct the step from history alone.
        ScalarType var1 = numerator / denominator;

        // Apply stability checks
        if (stable_mode && (isinf(var1) || isnan(var1))) {
            var1 = epsilon;
        }

        // Apply scaling: gq = gamma * gq
        ScalarType gamma = curobo::common::relu(var1);
        gq = gamma * gq;
    }

    /**
     * @brief Performs the forward pass of the L-BFGS two-loop algorithm
     *
     * This function implements the second loop of the L-BFGS two-loop recursion,
     * computing the final search direction.
     *
     * @tparam ScalarType Floating point type (float/double)
     * @param thread_idx Current thread index within batch
     * @param batch Current batch index
     * @param batchsize Total batch size
     * @param v_dim Optimization dimension size
     * @param history_m History buffer size
     * @param gq Search direction (input/output)
     * @param s_buffer History buffer for s vectors
     * @param y_buffer History buffer for y vectors
     * @param rho_buffer History buffer for rho values
     * @param alpha_buffer Alpha values from backward pass
     * @param data Temporary reduction buffer
     * @param result Reduction result buffer
     */
    template<typename ScalarType, int HISTORY_M = -1>
    __device__ __forceinline__ void lbfgs_forward_pass(
        const int thread_idx,
        const int batch,
        const int batchsize,
        const int v_dim,
        const int history_m,
        ScalarType& gq,
        const ScalarType* s_buffer,
        const ScalarType* y_buffer,
        const ScalarType* rho_buffer,
        const ScalarType* alpha_buffer,
        ScalarType* data,
        float* result)
    {
        constexpr bool is_compile_time = (HISTORY_M > 0);

        if constexpr (is_compile_time) {
            for (int i = 0; i < HISTORY_M; i++) {
            float current_y = y_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];
            float current_rho = rho_buffer[i * batchsize + batch];
            //float current_alpha = alpha_buffer[ thread_idx * HISTORY_M + i];
            float current_alpha = alpha_buffer[i];

            float current_s = s_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];

            // Compute y^T * gq
            curobo::common::block_reduce_sum(
                gq * current_y,
                v_dim, &data[0], result);

            // beta = rho_i * y_i^T * gq
            ScalarType beta = result[0] * current_rho;

            // gq = gq + (alpha_i - beta) * s_i
                gq = gq + (current_alpha - beta) *
                         current_s;
            }
        }
        else {
            for (int i = 0; i < history_m; i++) {

                float current_y = y_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];
                float current_rho = rho_buffer[i * batchsize + batch];

                //float current_alpha = alpha_buffer[ thread_idx * history_m + i];
                float current_alpha = alpha_buffer[i];

                float current_s = s_buffer[i * batchsize * v_dim + batch * v_dim + thread_idx];

                // Compute y^T * gq
                curobo::common::block_reduce_sum(
                    gq * current_y,
                    v_dim, &data[0], result);

                // beta = rho_i * y_i^T * gq
                ScalarType beta = result[0] * current_rho;

                // gq = gq + (alpha_i - beta) * s_i
                gq = gq + (current_alpha - beta) * current_s;
            }
        }
    }

    /**
     * @brief Scales a problem's step so that no action dimension exceeds its step limit
     *
     * Computes scale = max(1, max_i |step_i| / action_step_max[i % action_dim]) over the
     * block and returns step / scale. Matches LineSearchStrategy.scale_action.
     *
     * @param step Step value held by this thread
     * @param action_step_max Maximum step per action dimension. Shape: (action_dim)
     * @param action_dim Action dimension; v_dim is action_horizon * action_dim
     * @param v_dim Optimization dimension size
     * @param data Shared memory buffer for reductions
     * @param result Shared memory for reduction results
     * @return Scaled step value for this thread
     */
    template<typename ScalarType>
    __device__ __forceinline__ ScalarType scale_step_to_action_step_max(
        const ScalarType step,
        const ScalarType* action_step_max,
        const int action_dim,
        const int v_dim,
        ScalarType* data,
        ScalarType* result)
    {
        const ScalarType ratio = fabsf(step) / action_step_max[threadIdx.x % action_dim];
        curobo::common::block_reduce_max(ratio, v_dim, &data[0], result);
        const ScalarType scale = fmaxf(result[0], 1.0f);
        return step / scale;
    }

    ////////////////////
    // Compact L-BFGS
    //
    // The two-loop recursion only needs dot products between g, s_i and y_i. The compact
    // kernel keeps S^T Y and Y^T Y in a persistent per-problem buffer (gram), so each
    // iteration computes only the dot products that involve the new pair (s, y) or the new
    // gradient g, runs the recursion on scalars in one warp, and builds the step as
    // d = gamma * g + sum_i a_i * s_i + sum_i b_i * y_i.
    //
    // Shared memory holds each s_i, y_i and g as a zero-padded row of vector_stride floats.
    // vector_stride is a multiple of 4 (float4 loads) and is 4 mod 32, so threads reading
    // float4s at the same offset of consecutive rows hit distinct banks.
    ////////////////////

    /** Row stride of the m x m Gram matrices in shared memory (odd, to avoid bank conflicts). */
    __host__ __device__ __forceinline__ int compact_gram_stride(const int history_m)
    {
        return history_m | 1;
    }

    /** Row stride of the s_i, y_i and g vectors in shared memory. */
    __host__ __device__ __forceinline__ int compact_vector_stride(const int v_dim)
    {
        const int padded_dim = (v_dim + 3) & ~3;
        return padded_dim + ((4 - padded_dim) & 31);
    }

    /**
     * @brief Shared memory floats needed by kernel_lbfgs_step_compact.
     *
     * Layout: s and y rows (2 * history_m * vector_stride), g (vector_stride), S^T Y and
     * Y^T Y (2 * history_m * gram_stride), s^T g, y^T g, rho, two coefficient arrays
     * (5 * history_m), gamma (1) and the block reduction scratch (32 + 1).
     */
    __host__ __device__ __forceinline__ int compact_shared_memory_floats(
        const int v_dim, const int history_m)
    {
        const int vector_stride = compact_vector_stride(v_dim);
        const int gram_stride = compact_gram_stride(history_m);
        return (2 * history_m + 1) * vector_stride + 2 * history_m * gram_stride +
               5 * history_m + 1 + 33;
    }

    /**
     * @brief Dot product of two zero-padded shared memory rows with float4 loads.
     *
     * @param a First row, 16-byte aligned
     * @param b Second row, 16-byte aligned
     * @param padded_dim Row length, a multiple of 4
     */
    __device__ __forceinline__ float row_dot(
        const float* a, const float* b, const int padded_dim)
    {
        const float4* a4 = reinterpret_cast<const float4*>(a);
        const float4* b4 = reinterpret_cast<const float4*>(b);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int t = 0; t < padded_dim / 4; t++) {
            const float4 x = a4[t];
            const float4 y = b4[t];
            acc0 = fmaf(x.x, y.x, acc0);
            acc1 = fmaf(x.y, y.y, acc1);
            acc2 = fmaf(x.z, y.z, acc2);
            acc3 = fmaf(x.w, y.w, acc3);
        }
        return (acc0 + acc1) + (acc2 + acc3);
    }

    /**
     * @brief Computes the 5 * history_m - 1 dot products the new iteration adds, one per thread.
     *
     * With newest = history_m - 1, writes S^T Y row and column newest, Y^T Y row and column
     * newest, s_i^T g and y_i^T g for every i, and rho_newest = 1 / (s^T y) (0 when
     * s^T y <= 0 in stable mode).
     *
     * @param history_m History size
     * @param vector_stride Row stride of s_sh, y_sh (compact_vector_stride)
     * @param gram_stride Row stride of sy_sh, yy_sh (compact_gram_stride)
     * @param padded_dim v_dim rounded up to a multiple of 4; rows are zero past v_dim
     * @param s_sh History s rows, oldest first. Row i at s_sh[i * vector_stride]
     * @param y_sh History y rows, same layout as s_sh
     * @param g_sh Current gradient row
     * @param sy_sh S^T Y, (i, j) = s_i^T y_j at sy_sh[i * gram_stride + j] (output row/column)
     * @param yy_sh Y^T Y, same layout as sy_sh (output row/column)
     * @param sg_sh s_i^T g (output). Shape: (history_m)
     * @param yg_sh y_i^T g (output). Shape: (history_m)
     * @param rho_sh rho history; entry newest is written (output). Shape: (history_m)
     * @param stable_mode Whether to zero rho for non-positive curvature
     */
    __device__ __forceinline__ void compute_new_gram_entries(
        const int history_m,
        const int vector_stride,
        const int gram_stride,
        const int padded_dim,
        const float* s_sh,
        const float* y_sh,
        const float* g_sh,
        float* sy_sh,
        float* yy_sh,
        float* sg_sh,
        float* yg_sh,
        float* rho_sh,
        const bool stable_mode)
    {
        const int newest = history_m - 1;
        const int num_dots = 5 * history_m - 1;

        for (int k = threadIdx.x; k < num_dots; k += blockDim.x) {
            // Dot product k: kind 0: s_newest^T y_col, 1: s_col^T y_newest (col < newest),
            // 2: y_newest^T y_col, 3: s_col^T g, 4: y_col^T g.
            int kind, col;
            if (k < history_m) {
                kind = 0; col = k;
            } else if (k < 2 * history_m - 1) {
                kind = 1; col = k - history_m;
            } else {
                const int r = k - (2 * history_m - 1);
                kind = 2 + r / history_m;
                col = r % history_m;
            }

            const float* a = (kind == 0) ? s_sh + newest * vector_stride :
                             (kind == 1 || kind == 3) ? s_sh + col * vector_stride :
                             (kind == 2) ? y_sh + newest * vector_stride :
                             y_sh + col * vector_stride;
            const float* b = (kind == 0 || kind == 2) ? y_sh + col * vector_stride :
                             (kind == 1) ? y_sh + newest * vector_stride : g_sh;

            const float value = row_dot(a, b, padded_dim);

            if (kind == 0) {
                sy_sh[newest * gram_stride + col] = value;
                if (col == newest) {
                    float rho = 1.0f / value;
                    if (stable_mode && (value <= 0.0f)) {
                        rho = 0.0f;
                    }
                    rho_sh[newest] = rho;
                }
            } else if (kind == 1) {
                sy_sh[col * gram_stride + newest] = value;
            } else if (kind == 2) {
                yy_sh[newest * gram_stride + col] = value;
                yy_sh[col * gram_stride + newest] = value;
            } else if (kind == 3) {
                sg_sh[col] = value;
            } else {
                yg_sh[col] = value;
            }
        }
    }

    /**
     * @brief Runs the L-BFGS two-loop recursion on scalars in one warp.
     *
     * Lane i owns history pair i (history_m <= 31). With alpha_j and c_j = alpha_j - beta_j
     * of the textbook two-loop recursion:
     *   s_i^T q_i = s_i^T g - sum_{j > i} alpha_j (S^T Y)_ij
     *   y_i^T r_i = gamma (y_i^T g - sum_j alpha_j (Y^T Y)_ij) + sum_{k < i} c_k (S^T Y)_ki
     * Each step broadcasts one coefficient with a shuffle and updates the remaining lanes,
     * so the recursion has no block synchronization. The step direction is
     * -(gamma g + sum_i c_i s_i - gamma sum_i alpha_i y_i).
     *
     * Must be called by all 32 lanes of one warp.
     *
     * @tparam FixedM Compile-time history size (-1 for runtime)
     * @param history_m Runtime history size (ignored if FixedM > 0)
     * @param gram_stride Row stride of sy_sh and yy_sh (compact_gram_stride)
     * @param sy_sh S^T Y
     * @param yy_sh Y^T Y
     * @param sg_sh s_i^T g
     * @param yg_sh y_i^T g
     * @param rho_sh rho history
     * @param epsilon Gamma used when s^T y / y^T y is not finite (stable mode)
     * @param stable_mode Whether to apply stability checks
     * @param coef_s Output coefficients of s_i, c_i
     * @param coef_y Output coefficients of y_i, -gamma * alpha_i
     * @param gamma_sh Output Hessian scaling gamma
     */
    template<int FixedM = -1>
    __device__ __forceinline__ void compact_two_loop_coefficients(
        const int history_m,
        const int gram_stride,
        const float* sy_sh,
        const float* yy_sh,
        const float* sg_sh,
        const float* yg_sh,
        const float* rho_sh,
        const float epsilon,
        const bool stable_mode,
        float* coef_s,
        float* coef_y,
        float* gamma_sh)
    {
        constexpr bool is_compile_time = (FixedM > 0);
        const int m = is_compile_time ? FixedM : history_m;
        const int lane = threadIdx.x % curobo::common::warpSize;
        const bool active = lane < m;
        const float rho = active ? rho_sh[lane] : 0.0f;

        // Backward pass: alpha_i = rho_i * s_i^T q_i, from newest to oldest. Each lane also
        // accumulates y_lane^T q = y_lane^T g - sum_j alpha_j (Y^T Y)_lane,j for the forward pass.
        float r = active ? sg_sh[lane] : 0.0f;
        float yq = active ? yg_sh[lane] : 0.0f;
        float alpha = 0.0f;
        #pragma unroll
        for (int i = m - 1; i >= 0; i--) {
            const float alpha_i = __shfl_sync(curobo::common::fullMask, rho * r, i);
            if (lane == i) {
                alpha = alpha_i;
            }
            if (lane < i) {
                r = fmaf(-alpha_i, sy_sh[lane * gram_stride + i], r);
            }
            if (active) {
                yq = fmaf(-alpha_i, yy_sh[lane * gram_stride + i], yq);
            }
        }

        // gamma = s^T y / y^T y of the newest pair, as in compute_lbfgs_scaling.
        const int newest = m - 1;
        float var1 = sy_sh[newest * gram_stride + newest] / yy_sh[newest * gram_stride + newest];
        if (stable_mode && (isinf(var1) || isnan(var1))) {
            var1 = epsilon;
        }
        const float gamma = curobo::common::relu(var1);

        // y_i^T (gamma * q)
        float u = gamma * yq;

        // Forward pass: c_i = alpha_i - rho_i * y_i^T r_i, from oldest to newest.
        float c = 0.0f;
        #pragma unroll
        for (int i = 0; i < m; i++) {
            const float c_i = __shfl_sync(curobo::common::fullMask, alpha - rho * u, i);
            if (lane == i) {
                c = c_i;
            }
            if (lane > i && active) {
                u = fmaf(c_i, sy_sh[i * gram_stride + lane], u);
            }
        }

        if (active) {
            coef_s[lane] = c;
            coef_y[lane] = -gamma * alpha;
        }
        if (lane == 0) {
            gamma_sh[0] = gamma;
        }
    }

} // namespace lbfgs
} // namespace optimization
} // namespace curobo