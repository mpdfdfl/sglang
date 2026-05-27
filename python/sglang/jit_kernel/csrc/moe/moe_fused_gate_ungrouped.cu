/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

namespace moe {

// Common constants
static constexpr int WARPS_PER_CTA = 6;
static constexpr int SMALL_TOKEN_THRESHOLD = 512;
static constexpr int VEC_SIZE = 4;

// Small token optimized kernel: Each block handles 1 token, NUM_EXPERTS threads collaborate
// to find top-k using iterative warp-level reduction.
// output_stride: row stride in output/indices tensors (>= topk, allows reserved slots for shared experts)
template <int NUM_EXPERTS>
__global__ void moe_fused_gate_ungrouped_kernel_small_token(
    float* input,
    float* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk,
    int64_t output_stride,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  static constexpr int WARPS_PER_TOKEN_SMALL = NUM_EXPERTS / WARP_SIZE;
  static constexpr int MAX_TOPK = 8;

  int64_t row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  // Sigmoid weights (no bias) for final lookup, indexed by expert id.
  __shared__ float shared_original_scores[NUM_EXPERTS];
  __shared__ float warp_maxs[WARPS_PER_TOKEN_SMALL];
  __shared__ int warp_experts[WARPS_PER_TOKEN_SMALL];
  __shared__ int selected_experts[MAX_TOPK];

  // Keep biased_val in register; mask the winner in-place each iteration to
  // avoid round-tripping through shared memory.
  float input_val = input[row_idx * NUM_EXPERTS + tid];
  float bias_val = bias[tid];
  float sigmoid_val = 1.0f / (1.0f + expf(-input_val));
  float biased_val = sigmoid_val + bias_val;
  shared_original_scores[tid] = sigmoid_val;

  __syncthreads();

  // Lane 0 of warp 0 accumulates the renorm sum as it picks each winner,
  // saving a second pass over selected_experts during writeback.
  float sum_for_renorm = 0.0f;

  for (int k = 0; k < topk; k++) {
    // Stage 1: per-warp argmax.
    float warp_max_val = biased_val;
    int warp_max_expert = tid;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, warp_max_val, offset);
      int other_expert = __shfl_down_sync(0xFFFFFFFF, warp_max_expert, offset);
      if (other_val > warp_max_val) {
        warp_max_val = other_val;
        warp_max_expert = other_expert;
      }
    }
    if (lane_id == 0) {
      warp_maxs[warp_id] = warp_max_val;
      warp_experts[warp_id] = warp_max_expert;
    }
    __syncthreads();

    // Stage 2: warp 0 merges warp-leaders into a single winner.
    if (warp_id == 0) {
      float final_max = (lane_id < WARPS_PER_TOKEN_SMALL) ? warp_maxs[lane_id] : -FLT_MAX;
      int final_expert = (lane_id < WARPS_PER_TOKEN_SMALL) ? warp_experts[lane_id] : -1;
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, final_max, offset);
        int other_expert = __shfl_down_sync(0xFFFFFFFF, final_expert, offset);
        if (other_val > final_max) {
          final_max = other_val;
          final_expert = other_expert;
        }
      }
      if (lane_id == 0) {
        selected_experts[k] = final_expert;
        if (renormalize && final_expert >= 0 && final_expert < NUM_EXPERTS) {
          sum_for_renorm += shared_original_scores[final_expert];
        }
      }
    }
    __syncthreads();

    int selected = selected_experts[k];
    if (tid == selected) biased_val = -FLT_MAX;
  }

  // Lane 0 of warp 0 writes the output. sum_for_renorm was accumulated
  // during the topk loop, so we just fold it into rcp.
  if (warp_id == 0 && lane_id == 0) {
    float rcp = 1.0f;
    if (renormalize && sum_for_renorm > 0.0f) {
      rcp = 1.0f / sum_for_renorm;
      if (apply_routed_scaling_factor_on_output) {
        rcp *= static_cast<float>(routed_scaling_factor);
      }
    }

    for (int k = 0; k < topk; k++) {
      int expert_id = selected_experts[k];
      bool valid = (expert_id >= 0 && expert_id < NUM_EXPERTS);
      output_ptr[row_idx * output_stride + k] = valid ? shared_original_scores[expert_id] * rcp : 0.0f;
      indices_ptr[row_idx * output_stride + k] = valid ? expert_id : 0;
    }
  }
}

// Large token kernel: Each warp handles one token with vectorized loads
// output_stride: row stride in output/indices tensors (>= topk, allows reserved slots for shared experts)
template <int NUM_EXPERTS>
__global__ void moe_fused_gate_ungrouped_kernel(
    float* input,
    float* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk,
    int64_t output_stride,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  static constexpr int VPT = NUM_EXPERTS / WARP_SIZE;
  static constexpr int VEC_PER_LANE = VPT / VEC_SIZE;
  static constexpr int MAX_TOPK = 8;

  int64_t row_idx = blockIdx.x * WARPS_PER_CTA + threadIdx.y;
  // Each warp owns one token; all 32 lanes of a warp share row_idx, so an early
  // return is safe (no inter-warp sync below — only __syncwarp).
  if (row_idx >= num_rows) return;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;

  __shared__ float shared_scores[NUM_EXPERTS * WARPS_PER_CTA];
  __shared__ float shared_original_scores[NUM_EXPERTS * WARPS_PER_CTA];
  float* warp_scores = shared_scores + warp_id * NUM_EXPERTS;
  float* warp_original_scores = shared_original_scores + warp_id * NUM_EXPERTS;
  float4* warp_scores_v4 = reinterpret_cast<float4*>(warp_scores);
  float4* warp_original_scores_v4 = reinterpret_cast<float4*>(warp_original_scores);

  float4* input_vec = reinterpret_cast<float4*>(input + row_idx * NUM_EXPERTS);
  float4* bias_vec = reinterpret_cast<float4*>(bias);

  // Lane-strided vec_idx (each lane k stores at vec_idx k, k+32, k+64, ...) so each
  // iteration's STS.128 is lane-contiguous, avoiding shared-mem bank conflicts.
#pragma unroll
  for (int i = 0; i < VEC_PER_LANE; i++) {
    int vec_idx = lane_id + i * WARP_SIZE;
    float4 input_val = input_vec[vec_idx];
    float4 bias_val = bias_vec[vec_idx];

    float4 sigmoid_v4;
    float4 biased_v4;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float inp = ((float*)&input_val)[j];
      float b = ((float*)&bias_val)[j];
      float sigmoid_val = 1.0f / (1.0f + expf(-inp));
      ((float*)&sigmoid_v4)[j] = sigmoid_val;
      ((float*)&biased_v4)[j] = sigmoid_val + b;
    }
    warp_original_scores_v4[vec_idx] = sigmoid_v4;
    warp_scores_v4[vec_idx] = biased_v4;
  }

  __syncwarp();

  // Lane 0 records the picked expert ids and accumulates the renorm sum as
  // it goes; the global write is a single pass after the loop.
  int top_indices[MAX_TOPK];
  float sum_for_renorm = 0.0f;

  for (int k = 0; k < topk; k++) {
    float max_val = -FLT_MAX;
    int max_expert = -1;

    for (int expert = lane_id; expert < NUM_EXPERTS; expert += WARP_SIZE) {
      if (warp_scores[expert] > max_val) {
        max_val = warp_scores[expert];
        max_expert = expert;
      }
    }

    // warp shfl reduce; tie-break by lower expert id
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
      int other_expert = __shfl_down_sync(0xFFFFFFFF, max_expert, offset);
      if (other_val > max_val || (other_val == max_val && other_expert < max_expert)) {
        max_val = other_val;
        max_expert = other_expert;
      }
    }

    if (lane_id == 0) {
      bool valid = (max_expert >= 0 && max_expert < NUM_EXPERTS);
      top_indices[k] = valid ? max_expert : -1;
      if (renormalize && valid) {
        sum_for_renorm += warp_original_scores[max_expert];
      }
      if (valid) warp_scores[max_expert] = -FLT_MAX;
    }
    __syncwarp();
  }

  if (lane_id == 0) {
    float rcp = 1.0f;
    if (renormalize && sum_for_renorm > 0.0f) {
      rcp = 1.0f / sum_for_renorm;
      if (apply_routed_scaling_factor_on_output) {
        rcp *= static_cast<float>(routed_scaling_factor);
      }
    }

    for (int k = 0; k < topk; k++) {
      int e = top_indices[k];
      bool valid = (e >= 0);
      output_ptr[row_idx * output_stride + k] = valid ? warp_original_scores[e] * rcp : 0.0f;
      indices_ptr[row_idx * output_stride + k] = valid ? e : 0;
    }
  }
}

}  // namespace moe

namespace {

template <int NUM_EXPERTS>
struct MoeFusedGateUngroupedKernel {
  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView bias,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView indices,
      int64_t topk,
      bool renormalize,
      double routed_scaling_factor,
      bool apply_routed_scaling_factor_on_output) {
    using namespace host;

    auto device = input.device();
    const cudaStream_t stream = LaunchKernel::resolve_device(device);

    int64_t num_rows = input.size(0);
    int64_t output_stride = output.size(1);

    float* input_ptr = static_cast<float*>(input.data_ptr());
    float* bias_ptr = static_cast<float*>(bias.data_ptr());
    float* output_ptr = static_cast<float*>(output.data_ptr());
    int32_t* indices_ptr = static_cast<int32_t*>(indices.data_ptr());

    if (num_rows <= moe::SMALL_TOKEN_THRESHOLD) {
      LaunchKernel(dim3(num_rows), dim3(NUM_EXPERTS), stream)(
          moe::moe_fused_gate_ungrouped_kernel_small_token<NUM_EXPERTS>,
          input_ptr,
          bias_ptr,
          output_ptr,
          indices_ptr,
          num_rows,
          topk,
          output_stride,
          renormalize,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output);
    } else {
      int64_t num_blocks = (num_rows + moe::WARPS_PER_CTA - 1) / moe::WARPS_PER_CTA;
      LaunchKernel(dim3(num_blocks), dim3(WARP_SIZE, moe::WARPS_PER_CTA), stream)(
          moe::moe_fused_gate_ungrouped_kernel<NUM_EXPERTS>,
          input_ptr,
          bias_ptr,
          output_ptr,
          indices_ptr,
          num_rows,
          topk,
          output_stride,
          renormalize,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output);
    }
  }
};

}  // namespace
