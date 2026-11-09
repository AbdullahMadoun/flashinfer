// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <flashinfer/attention/sparse_mla_sm120/kernels/cute_decode/decode_fp8.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/cute_prefill/prefill_fp8.cuh>

#include "attention_descriptor.h"
#include "attention_dispatch.h"

namespace flashinfer::sparse_mla_sm120 {

cudaError_t dispatch_cute_decode(const execution::AttentionParams& p,
                                 const execution::ExecutionPlan& plan, cudaStream_t stream) {
  const auto& m = plan.metadata;
  cute_decode::Params a{};
  a.q = p.q;
  a.kv = p.kv;
  a.extra_kv = p.extra_kv;
  a.indices = p.indices;
  a.extra_indices = p.extra_indices;
  a.lengths = p.topk_length;
  a.extra_lengths = p.extra_topk_length;
  a.positions = p.mega.positions;
  a.cos_sin = p.mega.cos_sin;
  a.sink = p.attn_sink;
  a.output = p.mega.output;
  a.scales = p.mega.scales;
  a.bf16_output = p.output;
  a.lse = p.out_lse;
  a.tokens = m.tokens;
  a.heads = m.heads;
  a.topk = m.topk;
  a.extra_topk = m.extra_topk;
  a.kv_stride = m.row_stride_bytes;
  a.extra_stride = m.extra_row_stride_bytes;
  a.sm_scale = p.sm_scale;
  a.lse_stride = m.lse_stride;
  a.lse_scale = p.lse_scale;
  a.partial = p.mid_out;
  a.partial_lse = p.mid_lse;
  a.splits = plan.active_splits;
  auto launch = [&](auto warps, auto fused, auto split) {
    constexpr int W = decltype(warps)::value;
    constexpr bool Mega = decltype(fused)::value;
    auto kernel = cute_decode::decode<W, Mega, decltype(split)::value>;
    static thread_local bool configured[32] = {};
    auto status =
        configure_dynamic_smem_per_device(kernel, sizeof(cute_decode::Shared<W>), configured);
    if (!status.supported || status.error != cudaSuccess) return status.error;
    static thread_local bool carveout_set[32] = {};
    int device = 0;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    const bool cacheable = device >= 0 && device < 32;
    if (!cacheable || !carveout_set[device]) {
      error = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
      if (error != cudaSuccess) return error;
      if (cacheable) carveout_set[device] = true;
    }
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(m.tokens, plan.head_blocks, a.splits);
    config.blockDim = dim3((W + 1) * 32);
    config.dynamicSmemBytes = sizeof(cute_decode::Shared<W>);
    config.stream = stream;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {1, 1, unsigned(a.splits)};
    if (a.splits > 1) {
      config.attrs = &attr;
      config.numAttrs = 1;
    }
    return cudaLaunchKernelEx(&config, kernel, a);
  };
  auto heads = [&](auto fused) {
    if (m.heads == 8) return launch(cute::Int<1>{}, fused, cute::Int<0>{});
    if (m.heads == 16) return launch(cute::Int<2>{}, fused, cute::Int<0>{});
    if (m.heads == 64) {
      return a.splits == 8 ? launch(cute::Int<3>{}, fused, cute::Int<8>{})
                           : launch(cute::Int<3>{}, fused, cute::Int<0>{});
    }
    return launch(cute::Int<4>{}, fused, cute::Int<0>{});
  };
  return p.mega.output ? heads(std::true_type{}) : heads(std::false_type{});
}

namespace execution {
// Runs the existing metadata validation first; this resolver only changes scheduling.
ffi::Module resolve_cute_descriptor(ffi::Array<int64_t> values, int64_t mode, int64_t sm_count,
                                    int64_t max_shared) {
  auto m = unpack_metadata(values);
  TVM_FFI_ICHECK(mode >= 0 && mode <= 2 && sm_count > 0 && sm_count <= INT_MAX && max_shared > 0);
  m.variant = 0;
  auto p = resolve_attention(m, NumericRoute::FP8, 1, {int(sm_count), size_t(max_shared)});
  TVM_FFI_ICHECK(m.model == int(ModelType::DSV4_1_INLINE) && !m.extra_fp4 &&
                 (m.heads == 8 || m.heads == 16 || m.heads == 32 || m.heads == 64) &&
                 m.page_stride_bytes == size_t(m.page_size) * m.row_stride_bytes &&
                 m.indices_stride == size_t(m.topk) &&
                 (!m.extra_topk || (m.extra_page_stride_bytes ==
                                        size_t(m.extra_page_size) * m.extra_row_stride_bytes &&
                                    m.extra_indices_stride == size_t(m.extra_topk))))
      << "CuTe scheduling requires flat inline FP8 KV, contiguous indices and H=8/16/32/64";
  const int warps = m.heads == 64 ? 3 : std::min(m.heads / 8, 4);
  const int latency_heads = (m.heads + warps * 8 - 1) / (warps * 8);
  const int throughput_heads = (m.heads + 31) / 32;
  const int64_t tasks = int64_t(m.tokens) * latency_heads;
  // The 32-head prefill CTA uses one SM; the smaller decode CTA can reside
  // twice per SM except for the four-compute-warp specialization. Model whole
  // waves so crossing the prefill grid boundary does not select a slower path.
  // The measured unsplit decode/prefill wave-cost ratio is approximately 4/3.
  const int resident_decode = warps == 4 ? 1 : 2;
  const int64_t prefill_tasks = int64_t(m.tokens) * throughput_heads;
  const int64_t prefill_waves = (prefill_tasks + sm_count - 1) / sm_count;
  const int64_t decode_capacity = sm_count * resident_decode;
  const int64_t decode_waves = (tasks + decode_capacity - 1) / decode_capacity;
  const bool throughput = mode == 2 || (mode == 0 && prefill_tasks * 2 >= sm_count &&
                                        prefill_waves * 3 <= decode_waves * 4);
  p.implementation =
      throughput ? Implementation::CuteHighThroughput : Implementation::CuteLowLatency;
  p.metadata.variant = throughput ? 1 : 0;
  p.head_blocks = throughput ? throughput_heads : latency_heads;
  const int tiles = (m.topk + 31) / 32 + (m.extra_topk + 31) / 32;
  int splits = 1;
  while (splits < 8 && tasks * splits * 2 <= decode_capacity) splits *= 2;
  p.active_splits = throughput ? 1 : std::max(1, std::min(splits, tiles));
  p.chunk_capacity = p.scratch_split_stride = p.active_splits;
  p.scratch_heads = m.heads;
  p.merge = Merge::Direct;  // Split finalization is inside the attention launch.
  p.cpb = 0;
  p.block_threads = throughput ? 384 : (warps + 1) * 32;
  p.shared_bytes = throughput   ? sizeof(cute_prefill::integrated::Shared)
                   : warps == 1 ? sizeof(cute_decode::Shared<1>)
                   : warps == 2 ? sizeof(cute_decode::Shared<2>)
                   : warps == 3 ? sizeof(cute_decode::Shared<3>)
                                : sizeof(cute_decode::Shared<4>);
  const size_t slots = p.active_splits > 1 ? size_t(m.tokens) * m.heads * p.active_splits : 0;
  p.partial_bytes = slots * 512 * sizeof(bf16);
  p.lse_bytes = slots * sizeof(float);
  TVM_FFI_ICHECK_LE(p.shared_bytes, max_shared);
  return pack_plan(p);
}
}  // namespace execution
}  // namespace flashinfer::sparse_mla_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(resolve_cute_attention,
                              flashinfer::sparse_mla_sm120::execution::resolve_cute_descriptor);
