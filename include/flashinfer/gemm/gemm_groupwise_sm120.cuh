/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_GEMM_GEMM_GROUPWISE_SM120_CUH_
#define FLASHINFER_GEMM_GEMM_GROUPWISE_SM120_CUH_

#include <cuda_runtime.h>
#include <flashinfer/allocator.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <typeinfo>

#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_detail.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"

namespace flashinfer {
namespace gemm {

namespace detail {

inline cutlass::KernelHardwareInfo MakeCutlassHardwareInfo() {
  cutlass::KernelHardwareInfo hw_info{};
  // Query both the active device ID and its SM count per launch (multi-GPU safe).
  if (cudaGetDevice(&hw_info.device_id) == cudaSuccess) {
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  }
  return hw_info;
}

inline bool HandleTrivialGroupwiseGemmDims(void* D_ptr, int m, int n, int k, int l,
                                           size_t element_d_size, cudaStream_t stream,
                                           cudaError_t& err) {
  if (m < 0 || n < 0 || k < 0 || l < 0) {
    err = cudaErrorInvalidValue;
    return true;
  }
  if (m == 0 || n == 0 || l == 0) {
    err = cudaSuccess;
    return true;
  }
  if (k == 0) {
    err = cudaMemsetAsync(D_ptr, 0, static_cast<size_t>(m) * n * l * element_d_size, stream);
    return true;
  }
  return false;
}

template <typename Gemm>
inline cudaError_t RunGroupwiseGemmOperator(typename Gemm::Arguments& arguments, void* float_buffer,
                                            size_t float_buffer_size_in_bytes,
                                            cudaStream_t stream) {
  Gemm gemm;

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorNotSupported;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  if (workspace_size > float_buffer_size_in_bytes) {
    return cudaErrorInsufficientDriver;
  }

  void* kernel_workspace = nullptr;
  if (workspace_size > 0) {
    kernel_workspace = float_buffer;
  }

  status = gemm.initialize(arguments, kernel_workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorNotSupported;
  }

  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

}  // namespace detail

// SM120 uses Cooperative schedule with 128x128x128 tile shape.
// UseStreamK selects CUTLASS's StreamKScheduler (default Heuristic decomposition) for grids whose
// last wave would leave most SMs idle.
template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          bool UseStreamK = false, typename DTypeIn, typename DTypeOut>
cudaError_t CutlassGroupwiseScaledGEMMSM120(void* float_buffer, size_t float_buffer_size_in_bytes,
                                            DTypeIn* A_ptr, DTypeIn* B_ptr, float* SFA_ptr,
                                            float* SFB_ptr, DTypeOut* D_ptr, int m, int n, int k,
                                            int l, cudaStream_t stream) {
  // SM120/SM121 only supports these specific scale granularities
  static_assert(ScaleGranularityM == 1 || ScaleGranularityM == 128,
                "SM120/SM121 only supports ScaleGranularityM = 1 or 128");
  static_assert(ScaleGranularityN == 128, "SM120/SM121 only supports ScaleGranularityN = 128");
  static_assert(ScaleGranularityK == 128, "SM120/SM121 only supports ScaleGranularityK = 128");
  cudaError_t trivial_err = cudaSuccess;
  if (detail::HandleTrivialGroupwiseGemmDims(D_ptr, m, n, k, l, sizeof(DTypeOut), stream,
                                             trivial_err)) {
    return trivial_err;
  }
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  using namespace cute;

  using ElementA = DTypeIn;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = DTypeIn;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<DTypeOut>::value;

  using ElementD = DTypeOut;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = AlignmentC;

  using ElementAccumulator = float;
  using ElementCompute = float;

  // SM120 uses fixed 128x128x128 tile shape (Cooperative schedule)
  // TODO (yongwww): add PingPong schedule (64x128x128)
  using MmaTileShape_MNK = Shape<_128, _128, _128>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  // SM120's Sm120BlockwiseScaleConfig takes UMMA::Major parameters based on ScaleMajorK
  using ScaleConfig = std::conditional_t<
      ScaleMajorK,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK, UMMA::Major::K, UMMA::Major::K>,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK, UMMA::Major::MN,
                                                 UMMA::Major::MN>>;

  // Use decltype like SM100 does for consistency
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
      LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  // SM120 uses automatic stage count with epilogue carveout
  using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK, StageCount,
      cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

  using TileScheduler = cute::conditional_t<UseStreamK, cutlass::gemm::StreamKScheduler, void>;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, TileScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, l));

  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, l));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, l));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {A_ptr, stride_A, B_ptr, stride_B, SFA_ptr, layout_SFA, SFB_ptr, layout_SFB},
      {{}, nullptr, stride_C, D_ptr, stride_D},
      detail::MakeCutlassHardwareInfo()};

  // Set alpha and beta for the epilogue
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  return detail::RunGroupwiseGemmOperator<Gemm>(arguments, float_buffer, float_buffer_size_in_bytes,
                                                stream);
#else
  return cudaErrorNotSupported;
#endif
}

// Low-latency swap-AB kernel for small M (e.g., decode / small-batch prefill):
// Computes D^T = B * A^T with shape (n, m, k, l) and ColumnMajor output layout,
// which has the exact same memory layout as RowMajor D of shape (m, n, l).
// Placing M on the MMA N-axis enables narrow TileN (32, 64, or 96), reducing per-stage
// shared memory to (128 + TileN) * 128 bytes and eliminating CTA row padding waste.
template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          int TileN, bool UseStreamK = false, typename DTypeIn, typename DTypeOut>
cudaError_t CutlassGroupwiseScaledGEMMSM120LowLatency(
    void* float_buffer, size_t float_buffer_size_in_bytes, DTypeIn* A_ptr, DTypeIn* B_ptr,
    float* SFA_ptr, float* SFB_ptr, DTypeOut* D_ptr, int m, int n, int k, int l,
    cudaStream_t stream, int splits = 1, bool force_stream_k = false) {
  static_assert(ScaleGranularityM == 1,
                "SM120 low-latency swap-AB kernel requires ScaleGranularityM = 1");
  static_assert(ScaleGranularityN == 128, "SM120/SM121 only supports ScaleGranularityN = 128");
  static_assert(ScaleGranularityK == 128, "SM120/SM121 only supports ScaleGranularityK = 128");
  static_assert(TileN == 32 || TileN == 64 || TileN == 96,
                "SM120 low-latency swap-AB kernel only supports TileN = 32, 64, or 96");
  cudaError_t trivial_err = cudaSuccess;
  if (detail::HandleTrivialGroupwiseGemmDims(D_ptr, m, n, k, l, sizeof(DTypeOut), stream,
                                             trivial_err)) {
    return trivial_err;
  }
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  using namespace cute;

  using ElementA = DTypeIn;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = DTypeIn;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementC = void;
  using LayoutC = cutlass::layout::ColumnMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<DTypeOut>::value;

  using ElementD = DTypeOut;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = AlignmentC;

  using ElementAccumulator = float;
  using ElementCompute = float;

  using MmaTileShape_MNK = Shape<_128, Int<TileN>, _128>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;
  using EpilogueTileType = cute::conditional_t<TileN == 64, Shape<_128, _32>,
                                               cutlass::epilogue::collective::EpilogueTileAuto>;

  // Operands A and B are swapped (M' = n, N' = m), so ScaleGranularityM and ScaleGranularityN swap.
  using ScaleConfig = std::conditional_t<
      ScaleMajorK,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityN, ScaleGranularityM,
                                                 ScaleGranularityK, UMMA::Major::K, UMMA::Major::K>,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityN, ScaleGranularityM,
                                                 ScaleGranularityK, UMMA::Major::MN,
                                                 UMMA::Major::MN>>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      EpilogueTileType, ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC, ElementD,
      LayoutD, AlignmentD, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK, StageCount,
      cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

  using TileScheduler = cute::conditional_t<UseStreamK, cutlass::gemm::StreamKScheduler, void>;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, TileScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(n, k, l));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(m, k, l));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, l));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, l));

  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(n, m, k, l));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(n, m, k, l));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {n, m, k, l},
      {B_ptr, stride_A, A_ptr, stride_B, SFB_ptr, layout_SFA, SFA_ptr, layout_SFB},
      {{}, nullptr, stride_C, D_ptr, stride_D},
      detail::MakeCutlassHardwareInfo()};
  if constexpr (UseStreamK) {
    arguments.scheduler.splits = std::max(1, splits);
    if (force_stream_k) {
      arguments.scheduler.decomposition_mode =
          cutlass::gemm::kernel::detail::DecompositionMode::StreamK;
    } else if (arguments.scheduler.splits > 1) {
      arguments.scheduler.decomposition_mode =
          cutlass::gemm::kernel::detail::DecompositionMode::SplitK;
    }
  }

  return detail::RunGroupwiseGemmOperator<Gemm>(arguments, float_buffer, float_buffer_size_in_bytes,
                                                stream);
#else
  return cudaErrorNotSupported;
#endif
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_GEMM_GROUPWISE_SM120_CUH_
