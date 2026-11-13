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
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <flashinfer/cutlass_utils.cuh>
#include <type_traits>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

#define DISPATCH_DLPACK_INPUT_OUTPUT_DTYPE(input_dtype, output_dtype, c_type_in, c_type_out, ...) \
  [&]() -> bool {                                                                                 \
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(output_dtype, c_type_out, [&] {                    \
      return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(input_dtype, c_type_in,                           \
                                                [&] { return __VA_ARGS__(); });                   \
    });                                                                                           \
  }()

#define DISPATCH_SCALE_GRANULARITY(scale_granularity_m, scale_granularity_n, scale_granularity_k, \
                                   SCALE_GRANULARITY_M, SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, \
                                   ...)                                                           \
  [&]() -> bool {                                                                                 \
    /* SM120/SM121 Cooperative schedule uses 128x128x128 tile shape */                            \
    /* TODO (yongwww): PingPong schedule (64x128x128) will need additional dispatch logic */      \
    constexpr int SCALE_GRANULARITY_K = 128;                                                      \
    if (scale_granularity_k != 128) {                                                             \
      TVM_FFI_ICHECK(false)                                                                       \
          << "SM120/SM121 requires scale_granularity_k=128. CUTLASS enforces ScaleGranularityK "  \
             "must "                                                                              \
             "equal tile shape K dimension (128 for both Cooperative and PingPong schedules).";   \
      return false;                                                                               \
    }                                                                                             \
    /* Support (1,128,128) and (128,128,128) as per SM100's approach */                           \
    if (scale_granularity_m == 1 && scale_granularity_n == 128) {                                 \
      constexpr int SCALE_GRANULARITY_M = 1;                                                      \
      constexpr int SCALE_GRANULARITY_N = 128;                                                    \
      return __VA_ARGS__();                                                                       \
    } else if (scale_granularity_m == 128 && scale_granularity_n == 128) {                        \
      constexpr int SCALE_GRANULARITY_M = 128;                                                    \
      constexpr int SCALE_GRANULARITY_N = 128;                                                    \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    TVM_FFI_ICHECK(false) << "SM120: Unsupported scale granularity combination ("                 \
                          << scale_granularity_m << "," << scale_granularity_n << ","             \
                          << scale_granularity_k << ")";                                          \
    return false;                                                                                 \
  }()

#define DISPATCH_SCALE_MAJOR_K(scale_major_mode, SCALE_MAJOR_K, ...) \
  [&]() -> bool {                                                    \
    if (scale_major_mode == "K") {                                   \
      constexpr bool SCALE_MAJOR_K = true;                           \
      return __VA_ARGS__();                                          \
    } else if (scale_major_mode == "MN") {                           \
      constexpr bool SCALE_MAJOR_K = false;                          \
      return __VA_ARGS__();                                          \
    }                                                                \
    TVM_FFI_ICHECK(false) << "Unsupported Scale Major Mode";         \
    return false;                                                    \
  }()

namespace flashinfer {
namespace gemm {

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          typename DTypeIn, typename DTypeOut>
cudaError_t CutlassGroupwiseScaledGEMMSM120(void* float_buffer, size_t float_buffer_size_in_bytes,
                                            DTypeIn* A_ptr, DTypeIn* B_ptr, float* SFA_ptr,
                                            float* SFB_ptr, DTypeOut* C_ptr, int m, int n, int k,
                                            int l, cudaStream_t stream);

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          int TileN, bool UseStreamK = false, typename DTypeIn, typename DTypeOut>
cudaError_t CutlassGroupwiseScaledGEMMSM120LowLatency(
    void* float_buffer, size_t float_buffer_size_in_bytes, DTypeIn* A_ptr, DTypeIn* B_ptr,
    float* SFA_ptr, float* SFB_ptr, DTypeOut* C_ptr, int m, int n, int k, int l,
    cudaStream_t stream, int splits = 1, bool force_stream_k = false);

inline int QueryActiveDeviceMultiProcessorCount() {
  int device_id = 0;
  int sm_count = 0;
  if (cudaGetDevice(&device_id) == cudaSuccess &&
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id) == cudaSuccess &&
      sm_count > 0) {
    return sm_count;
  }
  return 1;
}

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          typename DTypeIn, typename DTypeOut>
cudaError_t SelectAndLaunchSM120Groupwise(void* float_buffer, size_t workspace_size_in_bytes,
                                          DTypeIn* A_ptr, DTypeIn* B_ptr, float* SFA_ptr,
                                          float* SFB_ptr, DTypeOut* C_ptr, int m, int n, int k,
                                          int l, cudaStream_t stream) {
  auto launch_baseline = [&]() {
    return CutlassGroupwiseScaledGEMMSM120<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
                                           ScaleMajorK>(float_buffer, workspace_size_in_bytes,
                                                        A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, m, n,
                                                        k, l, stream);
  };

  if constexpr (ScaleGranularityM != 1) {
    return launch_baseline();
  } else {
    if (m <= 0 || n <= 0 || k <= 0 || l <= 0) {
      return launch_baseline();
    }

    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    const size_t sm = static_cast<size_t>(QueryActiveDeviceMultiProcessorCount());
    const size_t l_nt = static_cast<size_t>(l) * ceil_div(n, 128);
    const int kt = ceil_div(k, 128);

    auto grid_for = [&](int tile_m) { return l_nt * static_cast<size_t>(ceil_div(m, tile_m)); };
    auto waves_for = [&](size_t g) { return (g + sm - 1) / sm; };
    auto fits_streamk_ws = [&](size_t g, int tile_n) {
      // Deterministic Stream-K / Split-K needs 1 FP32 tile accumulator per output tile + barriers.
      const size_t per_tile_bytes = static_cast<size_t>(128) * tile_n * sizeof(float) + 256;
      return workspace_size_in_bytes >= (g * per_tile_bytes + 65536);
    };
    auto pick_split_factor = [&](size_t g) {
      // Fixed Split-K (2..4) only when a 4-way split fills at least half the SMs. Otherwise
      // return 1 and let CUTLASS's Heuristic decomposition choose data-parallel or Stream-K.
      // Both use the scheduler's default deterministic reduction.
      if (g * 8 < sm) {
        return 1;
      }
      int max_splits = std::min(4, static_cast<int>(sm / g));
      max_splits = std::min(max_splits, kt / 12);
      return std::max(1, max_splits);
    };

    const size_t g32 = grid_for(32);
    const size_t g64 = grid_for(64);
    const size_t g96 = grid_for(96);
    const size_t g128 = grid_for(128);
    const size_t w32 = waves_for(g32);
    const size_t w64 = waves_for(g64);
    const size_t w96 = waves_for(g96);
    const size_t w128 = waves_for(g128);
    const size_t tail128 = g128 % sm;

    // 1. Underfilled GPU regime: use Swap-AB with Split-K / Stream-K to fill idle SMs.
    if (kt >= 16) {
      auto try_launch_stream_k = [&](auto tile_n_tag, size_t g) -> cudaError_t {
        if (!fits_streamk_ws(g, decltype(tile_n_tag)::value)) {
          return cudaErrorInsufficientDriver;
        }
        const int splits = pick_split_factor(g);
        const bool force_sk = (splits == 1 && g > sm / 2);
        return CutlassGroupwiseScaledGEMMSM120LowLatency<ScaleGranularityM, ScaleGranularityN,
                                                         ScaleGranularityK, ScaleMajorK,
                                                         decltype(tile_n_tag)::value, true>(
            float_buffer, workspace_size_in_bytes, A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, m, n, k,
            l, stream, splits, force_sk);
      };

      const bool use_sw32_sk = (m <= 96) ? (g32 <= sm / 16) : (g32 <= sm / 2);
      if (use_sw32_sk) {
        const cudaError_t status = try_launch_stream_k(std::integral_constant<int, 32>{}, g32);
        if (status != cudaErrorInsufficientDriver) {
          return status;
        }
      }

      const bool use_sw64_sk =
          (m > 96) && (m <= 256 ? g64 <= sm / 2 : m <= 384 && g64 <= (sm * 2) / 3);
      if (use_sw64_sk) {
        const cudaError_t status = try_launch_stream_k(std::integral_constant<int, 64>{}, g64);
        if (status != cudaErrorInsufficientDriver) {
          return status;
        }
      }
    }

    auto launch_persistent = [&](auto tile_n_tag) {
      return CutlassGroupwiseScaledGEMMSM120LowLatency<ScaleGranularityM, ScaleGranularityN,
                                                       ScaleGranularityK, ScaleMajorK,
                                                       decltype(tile_n_tag)::value, false>(
          float_buffer, workspace_size_in_bytes, A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, m, n, k, l,
          stream);
    };

    // 2. Persistent Swap-AB when it completes in <= waves of baseline 128x128.
    if (m <= 96 && w32 <= w128) {
      return launch_persistent(std::integral_constant<int, 32>{});
    }
    if (m <= 128 && w32 <= w128 && w32 == 1) {
      return launch_persistent(std::integral_constant<int, 32>{});
    }
    if (m <= 256 && w64 <= w128) {
      return launch_persistent(std::integral_constant<int, 64>{});
    }
    if (m <= 384 && w96 <= w128) {
      return launch_persistent(std::integral_constant<int, 96>{});
    }

    // 3. Multi-wave tail-wave smoothing when 128x128 leaves a <= 50% tail wave.
    if (w128 >= 2 && tail128 > 0 && tail128 <= sm / 2) {
      if (m <= 256) {
        const size_t t64 = g64 % sm;
        const size_t t96 = g96 % sm;
        if (m > 128 && !(t96 > 0 && t96 < sm / 4 && (t64 == 0 || t64 >= sm / 2))) {
          return launch_persistent(std::integral_constant<int, 96>{});
        }
        return launch_persistent(std::integral_constant<int, 64>{});
      }
      if (m <= 384) {
        return launch_persistent(std::integral_constant<int, 96>{});
      }
    }

    return launch_baseline();
  }
}

}  // namespace gemm
}  // namespace flashinfer

void CutlassGemmGroupwiseScaledSM120(TensorView float_workspace_buffer, TensorView A, TensorView B,
                                     TensorView SFA, TensorView SFB, TensorView C,
                                     int64_t scale_granularity_m, int64_t scale_granularity_n,
                                     int64_t scale_granularity_k, std::string scale_major_mode) {
  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  auto stream = get_stream(C.device());

  // Ensure scales are contiguous
  // Note: We keep the original shape and let the kernel's layout handle interpretation
  CHECK_CONTIGUOUS(SFA);
  CHECK_CONTIGUOUS(SFB);

  DISPATCH_SCALE_MAJOR_K(scale_major_mode, SCALE_MAJOR_K, [&] {
    return DISPATCH_DLPACK_INPUT_OUTPUT_DTYPE(A.dtype(), C.dtype(), c_type_in, c_type_out, [&] {
      return DISPATCH_SCALE_GRANULARITY(
          scale_granularity_m, scale_granularity_n, scale_granularity_k, SCALE_GRANULARITY_M,
          SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, [&] {
            using cutlass_t_in = cutlass_dtype_t<c_type_in>;
            using cutlass_t_out = cutlass_dtype_t<c_type_out>;

            // Handle both 2D and 3D tensors (BMM)
            int m, n, k, l;
            if (A.ndim() == 2) {
              // 2D case: simple matrix multiplication
              m = A.size(0);
              k = A.size(1);
              n = B.size(0);
              l = 1;  // no batch dimension
            } else if (A.ndim() == 3) {
              // 3D case: batch matrix multiplication
              l = A.size(0);  // batch size
              m = A.size(1);  // per-batch m dimension
              k = A.size(2);  // per-batch k dimension
              n = B.size(2);  // per-batch n dimension (B is [batch, k, n] column-major)
            } else {
              return false;  // Unsupported tensor dimension
            }

            const size_t workspace_size_in_bytes =
                get_element_size(float_workspace_buffer) * float_workspace_buffer.numel();
            const cudaError_t status = flashinfer::gemm::SelectAndLaunchSM120Groupwise<
                SCALE_GRANULARITY_M, SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, SCALE_MAJOR_K>(
                static_cast<void*>(float_workspace_buffer.data_ptr()), workspace_size_in_bytes,
                static_cast<cutlass_t_in*>(A.data_ptr()), static_cast<cutlass_t_in*>(B.data_ptr()),
                static_cast<float*>(SFA.data_ptr()), static_cast<float*>(SFB.data_ptr()),
                static_cast<cutlass_t_out*>(C.data_ptr()), m, n, k, l, stream);
            TVM_FFI_ICHECK_EQ(status, cudaSuccess)
                << "Failed to run cutlass gemm groupwise scaled sm120: "
                << cudaGetErrorString(status);
            return true;
          });
    });
  });
}
