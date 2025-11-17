/******************************************************************************
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

#include <torch/python.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "cub/util_type.cuh"
#include <cub/cub.cuh>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "flash_topk.h"
#include "namespace_config.h"

namespace FLASH_TOPK_NAMESPACE {

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


/* --------------------------------------------
 * Segmented (variable-length) sort implementation
 * -------------------------------------------- */

template <typename scalar_t, bool Is_int64=false>
static void segmented_sort_impl(
    const at::Tensor& cu_seq_starts,
    const at::Tensor& cu_seq_ends,
    const at::Tensor& src,
    at::Tensor& dst)
{
    // using offset_t = int64_t;
    // using cub_key_t = int32_t;

    at::cuda::CUDAGuard device_guard(src.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int64_t total_elems = src.numel();
    int64_t seg_num = cu_seq_starts.numel();
    const int64_t* d_st_offsets = cu_seq_starts.data_ptr<int64_t>();
    const int64_t* d_ed_offsets   = cu_seq_ends.data_ptr<int64_t>();
    const scalar_t* d_keys_in  = reinterpret_cast<scalar_t*>(src.data_ptr<int32_t>());
    scalar_t* d_keys_out = reinterpret_cast<scalar_t*>(dst.data_ptr<int32_t>());
    // cudaStream_t stream = 0;

    // void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    if (Is_int64) {
        cub::DeviceSegmentedSort::SortKeys(
            nullptr,
            temp_storage_bytes,
            d_keys_in,
            d_keys_out,
            total_elems,
            seg_num,
            d_st_offsets,
            d_ed_offsets,
            stream
        );
    } else {
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr,
            temp_storage_bytes,
            d_keys_in,
            d_keys_out,
            total_elems,
            seg_num,
            d_st_offsets,
            d_ed_offsets,
            0,
            8 * sizeof(scalar_t),
            stream
        );
    }

    // cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // temp_storage_bytes is size_t (unsigned long). Cast to int64_t to avoid narrowing conversion when constructing the size list
    at::Tensor d_temp_storage = torch::empty({static_cast<int64_t>(temp_storage_bytes)}, src.options().dtype(at::kByte));
    // Tensor d_temp_storage = at::empty({temp_storage_bytes}, src.options());

    if (Is_int64) {
        cub::DeviceSegmentedSort::SortKeys(
            d_temp_storage.data_ptr(),
            temp_storage_bytes,
            d_keys_in,
            d_keys_out,
            total_elems,
            seg_num,
            d_st_offsets,
            d_ed_offsets,
            stream
        );
    } else {
        cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp_storage.data_ptr(),
            temp_storage_bytes,
            d_keys_in,
            d_keys_out,
            total_elems,
            seg_num,
            d_st_offsets,
            d_ed_offsets,
            0,
            8 * sizeof(scalar_t),
            stream
        );
    }

    // Ensure the sorting kernels have finished before freeing temporary storage
    // C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    // cudaFree(d_temp_storage);
    // C10_CUDA_KERNEL_LAUNCH_CHECK();
    // printf("err0: %d, err1: %d\n", err0, err1);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "DeviceSegmentedSort kernel failed");

    // return dst;
}

// Public C++/CUDA entry – dispatch on dtype
at::Tensor varlen_sort(
    at::Tensor cu_seq_starts,
    at::Tensor cu_seq_ends,
    at::Tensor src)
{
    // TORCH_CHECK(src.numel() < INT_MAX);
    TORCH_CHECK(cu_seq_starts.scalar_type() == torch::kInt64, "only support int64 cu_seq_starts data type");
    TORCH_CHECK(cu_seq_ends.scalar_type() == torch::kInt64, "only support int64 cu_seq_ends data type");
    TORCH_CHECK(src.scalar_type() == torch::kInt32, "only support int32 src data type");

    CHECK_DEVICE(cu_seq_starts);
    CHECK_DEVICE(cu_seq_ends);
    CHECK_DEVICE(src);

    // Ensure we have 1-D tensors
    TORCH_CHECK(cu_seq_starts.dim() == 1 && cu_seq_ends.dim() == 1,
                "cu_seq_starts / cu_seq_lens must be 1-D");

    TORCH_CHECK(cu_seq_starts.numel() == cu_seq_ends.numel(),
                "seq_starts and seq_lens must have the same length");

    // Contiguity checks
    CHECK_CONTIGUOUS(cu_seq_starts);
    CHECK_CONTIGUOUS(cu_seq_ends);
    CHECK_CONTIGUOUS(src);

    // at::Tensor dst = at::empty_like(src);
    at::Tensor dst = at::zeros_like(src);
    // if (src.numel() < INT_MAX) {
    //     segmented_sort_impl<int32_t, false>(cu_seq_starts, cu_seq_ends, src, dst);
    // } else {
    //     segmented_sort_impl<int32_t, true>(cu_seq_starts, cu_seq_ends, src, dst);
    // }
    segmented_sort_impl<int32_t, true>(cu_seq_starts, cu_seq_ends, src, dst);
    return dst;
}

} // namespace FLASH_TOPK_NAMESPACE