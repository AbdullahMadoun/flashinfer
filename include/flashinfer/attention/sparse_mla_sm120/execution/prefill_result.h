// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>

namespace flashinfer::sparse_mla_sm120 {

struct PrefillLaunchResult {
  bool supported;
  cudaError_t error;
  const char* operation;

  PrefillLaunchResult(bool supported)
      : supported(supported), error(cudaSuccess), operation(nullptr) {}
  PrefillLaunchResult(cudaError_t error, const char* operation)
      : supported(true), error(error), operation(operation) {}
};

constexpr int kMaxCachedCudaDevices = 32;

template <typename Kernel>
PrefillLaunchResult configure_dynamic_smem_per_device(Kernel kernel, size_t smem_bytes,
                                                      bool (&configured)[kMaxCachedCudaDevices]) {
  if (smem_bytes <= 48 * 1024) return true;

  int device = 0;
  const cudaError_t device_error = cudaGetDevice(&device);
  if (device_error != cudaSuccess) return {device_error, "cudaGetDevice"};
  const bool cacheable_device = device >= 0 && device < kMaxCachedCudaDevices;
  if (cacheable_device && configured[device]) return true;

  const cudaError_t rc = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(smem_bytes));
  if (rc == cudaSuccess && cacheable_device) configured[device] = true;
  return {rc, "cudaFuncSetAttribute"};
}

}  // namespace flashinfer::sparse_mla_sm120
